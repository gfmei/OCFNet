"""
Scripts for pairwise registration demo

Author: Shengyu Huang
Last modified: 22.02.2021

Ported to the sparse-convolutional (spconv) pipeline: the demo now builds the same
voxelised inputs as datasets/indoor.py and runs the ResUNet backbone of models/resunet.py.
Without a display (e.g. on a compute node) it writes the result to disk instead of
opening open3d windows.
"""
import os, torch, copy, sys, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
import open3d as o3d

cwd = os.getcwd()
sys.path.append(cwd)
from datasets.dataloader import collate_pair_fn
from lib.spconv_utils import sparse_quantize, make_sparse_tensor
from lib.utils import setup_seed, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor
from models import load_model
setup_seed(0)


class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,config, src_path, tgt_path):
        super(ThreeDMatchDemo,self).__init__()
        self.config = config
        self.voxel_size = config.voxel_size
        self.src_path = src_path
        self.tgt_path = tgt_path

    def __len__(self):
        return 1

    def __getitem__(self,item):
        # get pointcloud
        src_pcd = torch.load(self.src_path, weights_only=False).astype(np.float32)
        tgt_pcd = torch.load(self.tgt_path, weights_only=False).astype(np.float32)

        # voxelise, one point per voxel
        _, sel_src = sparse_quantize(np.ascontiguousarray(src_pcd) / self.voxel_size, return_index=True)
        _, sel_tgt = sparse_quantize(np.ascontiguousarray(tgt_pcd) / self.voxel_size, return_index=True)
        src_xyz, tgt_xyz = src_pcd[sel_src], tgt_pcd[sel_tgt]
        src_coords, tgt_coords = np.floor(src_xyz / self.voxel_size), np.floor(tgt_xyz / self.voxel_size)

        src_feats = np.ones((src_coords.shape[0],1),dtype=np.float32)
        tgt_feats = np.ones((tgt_coords.shape[0],1),dtype=np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3,1)).astype(np.float32)
        correspondences = torch.ones(1,2).long()

        return to_tensor(src_xyz).float(), to_tensor(tgt_xyz).float(), src_coords, tgt_coords, \
               src_feats, tgt_feats, correspondences, to_tensor(rot), to_tensor(trans), 1


def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent


def build_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, tsfm):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    tgt_pcd_before.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    ########################################
    # 2. overlap colors
    src_overlap = src_overlap[:,None].repeat(1,3).numpy()
    tgt_overlap = tgt_overlap[:,None].repeat(1,3).numpy()
    src_overlap_color = lighter(get_yellow(), 1 - src_overlap)
    tgt_overlap_color = lighter(get_blue(), 1 - tgt_overlap)
    src_pcd_overlap = copy.deepcopy(src_pcd_before)
    src_pcd_overlap.transform(tsfm)
    tgt_pcd_overlap = copy.deepcopy(tgt_pcd_before)
    src_pcd_overlap.colors = o3d.utility.Vector3dVector(src_overlap_color)
    tgt_pcd_overlap.colors = o3d.utility.Vector3dVector(tgt_overlap_color)

    ########################################
    # 3. registered source
    src_pcd_after = copy.deepcopy(src_pcd_before)
    src_pcd_after.transform(tsfm)

    return src_pcd_before, tgt_pcd_before, src_pcd_overlap, tgt_pcd_overlap, src_pcd_after


def draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm):
    src_pcd_before, tgt_pcd_before, src_pcd_overlap, tgt_pcd_overlap, src_pcd_after = \
        build_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, tsfm)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='Inferred overlap region', width=960, height=540, left=0, top=600)
    vis2.add_geometry(src_pcd_overlap)
    vis2.add_geometry(tgt_pcd_overlap)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name ='Our registration', width=960, height=540, left=960, top=0)
    vis3.add_geometry(src_pcd_after)
    vis3.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        vis3.update_geometry(tgt_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(src_pcd_overlap)
        vis2.update_geometry(tgt_pcd_overlap)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        vis3.update_geometry(src_pcd_after)
        vis3.update_geometry(tgt_pcd_before)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()


def save_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, tsfm, out_dir):
    """Headless fallback: dump the coloured clouds and the estimated pose next to each other."""
    src_pcd_before, tgt_pcd_before, src_pcd_overlap, tgt_pcd_overlap, src_pcd_after = \
        build_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, tsfm)
    os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(f'{out_dir}/src_input.ply', src_pcd_before)
    o3d.io.write_point_cloud(f'{out_dir}/tgt_input.ply', tgt_pcd_before)
    o3d.io.write_point_cloud(f'{out_dir}/src_overlap.ply', src_pcd_overlap)
    o3d.io.write_point_cloud(f'{out_dir}/tgt_overlap.ply', tgt_pcd_overlap)
    o3d.io.write_point_cloud(f'{out_dir}/src_registered.ply', src_pcd_after)
    np.savetxt(f'{out_dir}/tsfm.txt', tsfm)
    print(f'no display found, wrote the point clouds and the estimated pose to {out_dir}')


def main(config, demo_loader):
    config.model.eval()
    c_loader_iter = demo_loader.__iter__()
    with torch.no_grad():
        inputs = next(c_loader_iter)

        ###############################################
        # forward pass
        sinput_src = make_sparse_tensor(inputs['src_C'], inputs['src_F'], device=config.device)
        sinput_tgt = make_sparse_tensor(inputs['tgt_C'], inputs['tgt_F'], device=config.device)
        src_feats, tgt_feats, scores_overlap, scores_saliency = config.model(sinput_src, sinput_tgt)

        src_pcd, tgt_pcd = inputs['pcd_src'], inputs['pcd_tgt']
        len_src = src_pcd.size(0)
        src_raw = copy.deepcopy(src_pcd)
        tgt_raw = copy.deepcopy(tgt_pcd)
        src_feats, tgt_feats = src_feats.detach().cpu(), tgt_feats.detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

        ########################################
        # do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        if(src_pcd.size(0) > config.n_points):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if(tgt_pcd.size(0) > config.n_points):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

        ########################################
        # run ransac and draw registration
        tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)
        if os.environ.get('DISPLAY'):
            draw_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, src_saliency, tgt_saliency, tsfm)
        else:
            save_registration_result(src_raw, tgt_raw, src_overlap, tgt_overlap, tsfm,
                                     os.path.join(config.snapshot_dir, 'demo') if 'snapshot_dir' in config else 'demo_output')


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # model initialization
    Model = load_model(config.model)
    config.model = Model(config, D=3).to(config.device)

    # create dataset and dataloader
    demo_set = ThreeDMatchDemo(config, config.src_pcd, config.tgt_pcd)
    demo_loader = torch.utils.data.DataLoader(demo_set,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1,
                                        collate_fn=collate_pair_fn)

    # load pretrained weights
    assert config.pretrain != None
    state = torch.load(config.pretrain, weights_only=False)
    config.model.load_state_dict(state['state_dict'])

    # do pose estimation
    main(config, demo_loader)
