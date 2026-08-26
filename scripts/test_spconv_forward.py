"""
Self-contained smoke test for the spconv port of the sparse Predator backbone.

    python scripts/test_spconv_forward.py configs/train/indoor.yaml

Checks, on one GPU and without any dataset:
  1. the model builds and runs a forward pass on a pair of point clouds
  2. the outputs keep one row per input voxel, in the input order -- the whole pipeline
     (correspondences, losses, RANSAC) indexes model outputs with point indices, so a
     reordering would silently corrupt training. This is verified by permuting the input
     rows and comparing against the permuted reference output
  3. gradients flow back to every parameter and are finite
  4. batch_size = 2 works (exercises the per-sample sparse instance norm)
"""
import argparse, os, sys, time
import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.append(os.getcwd())
from lib.utils import load_config, setup_seed
from lib.spconv_utils import sparse_quantize, sparse_collate, make_sparse_tensor
from models import load_model

setup_seed(0)


def voxelise(pcd, voxel_size):
    """datasets/indoor.py in a nutshell: one point per voxel + its integer coordinates."""
    _, sel = sparse_quantize(np.ascontiguousarray(pcd) / voxel_size, return_index=True)
    xyz = pcd[sel]
    coords = np.floor(xyz / voxel_size)
    feats = np.ones((coords.shape[0], 1), dtype=np.float32)
    return xyz, coords, feats


def load_pair(config):
    """Use the two demo fragments if they are around, otherwise synthesise a pair."""
    src_path, tgt_path = config.get('src_pcd', ''), config.get('tgt_pcd', '')
    if os.path.isfile(src_path) and os.path.isfile(tgt_path):
        src = torch.load(src_path, weights_only=False).astype(np.float32)
        tgt = torch.load(tgt_path, weights_only=False).astype(np.float32)
        print(f'using the demo fragments {src_path} / {tgt_path}')
    else:
        rng = np.random.default_rng(0)
        # two overlapping noisy planes, centred on the origin so that a good part of the
        # coordinates is negative (MinkowskiEngine allowed that, spconv needs the shift
        # that lib/spconv_utils.make_sparse_tensor applies)
        def sheet(n, offset):
            xy = rng.uniform(-1.5, 1.5, size=(n, 2))
            z = 0.3 * np.sin(3 * xy[:, :1]) + 0.02 * rng.standard_normal((n, 1))
            return (np.concatenate([xy, z], axis=1) + offset).astype(np.float32)
        src = sheet(40000, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        tgt = sheet(40000, np.array([0.4, 0.1, 0.0], dtype=np.float32))
        print('demo fragments not found, using synthetic clouds')
    return src, tgt


def build_inputs(src, tgt, voxel_size, device, repeat=1):
    src_xyz, src_coords, src_feats = voxelise(src, voxel_size)
    tgt_xyz, tgt_coords, tgt_feats = voxelise(tgt, voxel_size)
    src_C, src_F = sparse_collate([src_coords] * repeat, [src_feats] * repeat)
    tgt_C, tgt_F = sparse_collate([tgt_coords] * repeat, [tgt_feats] * repeat)
    return (make_sparse_tensor(src_C, src_F, device=device),
            make_sparse_tensor(tgt_C, tgt_F, device=device),
            src_xyz.shape[0], tgt_xyz.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/train/indoor.yaml')
    args = parser.parse_args()
    config = edict(load_config(args.config))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch {torch.__version__}, device {device}')
    if device.type == 'cuda':
        print(f'gpu: {torch.cuda.get_device_name(0)}')
    else:
        print('WARNING: no CUDA device, spconv only supports its implicit-GEMM kernels on GPU')
    import spconv
    print(f'spconv {spconv.__version__}')

    model = load_model(config.model)(config, D=3).to(device)
    n_params = sum(p.nelement() for p in model.parameters())
    print(f'model {config.model}: {n_params / 1e6:.3f} M parameters')

    src, tgt = load_pair(config)

    ###########################################################################
    # 1. forward pass
    stensor_src, stensor_tgt, n_src, n_tgt = build_inputs(src, tgt, config.voxel_size, device)
    print(f'voxels: src {n_src}, tgt {n_tgt} (from {src.shape[0]} / {tgt.shape[0]} points), '
          f'grid {stensor_src.spatial_shape}')

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        src_feats, tgt_feats, scores_overlap, scores_saliency = model(stensor_src, stensor_tgt)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print(f'forward: {time.time() - t0:.3f}s')

    assert src_feats.shape == (n_src, config.out_feats_dim), src_feats.shape
    assert tgt_feats.shape == (n_tgt, config.out_feats_dim), tgt_feats.shape
    assert scores_overlap.shape == (n_src + n_tgt,), scores_overlap.shape
    assert scores_saliency.shape == (n_src + n_tgt,), scores_saliency.shape
    assert torch.isfinite(src_feats).all() and torch.isfinite(tgt_feats).all()
    print(f'[ok] shapes: src_feats {tuple(src_feats.shape)}, tgt_feats {tuple(tgt_feats.shape)}, '
          f'scores {tuple(scores_overlap.shape)}')

    ###########################################################################
    # 2a. row order, exactly: the tensor coming out of the U-Net must carry the input
    #     indices, in the input order (skip connections and the inverse convolutions rely
    #     on it, and so does every point index used by the losses)
    captured = []
    handle = model.final.register_forward_hook(lambda m, i, o: captured.append(o.indices))
    with torch.no_grad():
        model(stensor_src, stensor_tgt)
    handle.remove()
    assert torch.equal(captured[0], stensor_src.indices), 'src output coordinates were reordered'
    assert torch.equal(captured[1], stensor_tgt.indices), 'tgt output coordinates were reordered'
    print('[ok] the U-Net returns exactly the input coordinates, in the input order')

    ###########################################################################
    # 2b. permuting the input voxels must permute the output rows the same way. The
    #     attention modules are permutation equivariant and the rotary position embedding
    #     travels with the points, so this holds for the whole model.
    perm = torch.randperm(n_src, device=device)
    permuted = make_sparse_tensor(stensor_src.indices[perm].clone(),
                                  stensor_src.features[perm].clone(), device=device)
    with torch.no_grad():
        src_feats_perm, _, scores_overlap_perm, _ = model(permuted, stensor_tgt)
    feat_diff = (src_feats_perm - src_feats[perm]).abs().max().item()
    score_diff = (scores_overlap_perm[:n_src] - scores_overlap[:n_src][perm]).abs().max().item()
    print(f'[{"ok" if max(feat_diff, score_diff) < 1e-3 else "FAIL"}] output rows follow the '
          f'input order (max |diff| feats {feat_diff:.2e}, overlap {score_diff:.2e})')
    assert max(feat_diff, score_diff) < 1e-3, 'model outputs are not aligned with the input voxels'

    ###########################################################################
    # 3. backward pass
    model.train()
    src_feats, tgt_feats, scores_overlap, scores_saliency = model(stensor_src, stensor_tgt)
    loss = src_feats.sum() + tgt_feats.sum() + scores_overlap.sum() + scores_saliency.sum()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    non_finite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f'no gradient for: {missing}'
    assert not non_finite, f'non-finite gradient for: {non_finite}'
    print(f'[ok] backward: all {len(list(model.parameters()))} parameters got a finite gradient')
    if device.type == 'cuda':
        print(f'peak gpu memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')

    ###########################################################################
    # 4. batch_size = 2
    model.zero_grad(set_to_none=True)
    stensor_src2, stensor_tgt2, _, _ = build_inputs(src, tgt, config.voxel_size, device, repeat=2)
    src_feats2, tgt_feats2, scores_overlap2, _ = model(stensor_src2, stensor_tgt2)
    assert src_feats2.shape[0] == 2 * n_src, src_feats2.shape
    assert torch.isfinite(src_feats2).all()
    src_feats2.sum().backward()
    print(f'[ok] batch_size=2: src_feats {tuple(src_feats2.shape)}')

    print('\nall checks passed')


if __name__ == '__main__':
    main()
