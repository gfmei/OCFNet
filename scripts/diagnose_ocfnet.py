"""
Why is OCFNet's inlier ratio low? Measure the ceiling before changing anything.

    python scripts/diagnose_ocfnet.py configs/test/ocfnet_indoor.yaml --batches 8

Three things can hold the point level back, and they need different fixes:

  1. patch sampling -- a patch keeps only `patch_size` of its points, so a source point whose
     true partner was dropped can never be matched. The reachable fraction is an upper bound
     on the inlier ratio, whatever the transport plan does
  2. the transport plan -- of the source points whose partner *is* in the patch, how many does
     the plan actually pick
  3. the coarse overlap head -- `coarse_overlap_loss` barely moves during training. Since the
     targets of Eq. (7) are soft, its cross entropy cannot go below the entropy of those
     targets; the floor is printed here so a stuck head can be told from a converged one
"""
import argparse, os, sys

import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.append(os.getcwd())
from datasets.dataloader import get_datasets, collate_pair_fn
from lib.spconv_utils import make_sparse_tensor
from lib.utils import load_config, remap_legacy_state_dict, setup_seed
from models.ocfnet import OCFLoss, OCFNet

setup_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/test/ocfnet_indoor.yaml')
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    config = edict(load_config(args.config))
    config.batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    model = OCFNet(config, D=3).to(device).eval()
    if config.pretrain:
        state = torch.load(config.pretrain, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(
            remap_legacy_state_dict(state['state_dict']), strict=False)
        if missing or unexpected:
            print(f'checkpoint predates {list(missing)}, ignoring {list(unexpected)}')
        print(f'checkpoint {config.pretrain} (epoch {state["epoch"]})')
    criterion = OCFLoss(config)
    radius = criterion.matching_radius

    _, val_set, _ = get_datasets(config)
    stats = {k: [] for k in ['reachable', 'picked', 'picked_of_reachable', 'patch_points',
                             'kept', 'target_entropy', 'coarse_bce', 'mu_mean', 'mu_std',
                             'target_mean', 'auc', 'confident_frac', 'confident_correct',
                             'emitted', 'emitted_correct']}
    for it in range(args.batches):
        batch = collate_pair_fn([val_set[(it * args.batch_size + i) % len(val_set)]
                                 for i in range(args.batch_size)])
        src = make_sparse_tensor(batch['src_C'], batch['src_F'], device=device)
        tgt = make_sparse_tensor(batch['tgt_C'], batch['tgt_F'], device=device)
        correspondence = batch['correspondences'].long().to(device)

        with torch.no_grad():
            out = model(src, tgt, correspondences=correspondence,
                        src_xyz=inputs['pcd_src'].to(device), tgt_xyz=inputs['pcd_tgt'].to(device))

            # ---- 1/2. what the patches make reachable, and what the plan picks ----------
            plan = out['patch_log_plan']
            if plan.shape[0]:
                pair = out['patch_batch']
                rot, trans = batch['rot'].to(device), batch['trans'].to(device)
                src_xyz = batch['pcd_src'].to(device)[out['patch_src_index']]
                tgt_xyz = batch['pcd_tgt'].to(device)[out['patch_tgt_index']]
                src_xyz = torch.einsum('pij,pkj->pki', rot[pair], src_xyz) + trans[pair].transpose(1, 2)
                hit = (torch.cdist(src_xyz, tgt_xyz) < radius)
                hit = hit & out['patch_src_valid'][..., None] & out['patch_tgt_valid'][:, None, :]

                valid = out['patch_src_valid']
                reachable = hit.any(-1) & valid            # partner survived the sampling
                masked = plan[:, :-1, :-1].masked_fill(
                    ~out['patch_tgt_valid'][:, None, :], -float('inf'))
                picked = masked.argmax(-1, keepdim=True)
                correct = hit.gather(-1, picked).squeeze(-1) & valid

                # would requiring the match to beat the slack entry help?
                confident = masked.gather(-1, picked).squeeze(-1) > plan[:, :-1, -1]
                stats['confident_frac'].append(float((confident & valid).sum() / valid.sum()))
                stats['confident_correct'].append(
                    float((correct & confident).sum() / (confident & valid).sum().clamp(min=1)))

                # what the readout actually emits, with the mutual check and dedup
                pairs, _ = model.point_correspondences(out)
                if pairs.shape[0]:
                    moved = batch['pcd_src'].to(device)[pairs[:, 0]]
                    moved = torch.einsum('ij,kj->ki', rot[0], moved) + trans[0].view(1, 3)
                    close = (moved - batch['pcd_tgt'].to(device)[pairs[:, 1]]).norm(dim=-1) < radius
                    stats['emitted'].append(pairs.shape[0] / args.batch_size)
                    stats['emitted_correct'].append(float(close.float().mean()))

                stats['reachable'].append(float(reachable.sum() / valid.sum()))
                stats['picked'].append(float(correct.sum() / valid.sum()))
                stats['picked_of_reachable'].append(
                    float(correct.sum() / reachable.sum().clamp(min=1)))
                stats['kept'].append(float(valid.float().sum(-1).mean()))

            # ---- how many points a patch has to choose from ---------------------------
            counts = torch.bincount(out['src_patch_id'])
            stats['patch_points'].append(float(counts.float().mean()))

            # ---- 3. is the coarse overlap head stuck, or at its floor? -----------------
            ratio, visibility = criterion.coarse_targets(
                correspondence, out['src_patch_id'], out['tgt_patch_id'],
                out['src_slot'], out['tgt_slot'], out['src_super_batch'],
                out['tgt_super_batch'], (src.batch_size, out['src_super_mask'].shape[1],
                                         out['tgt_super_mask'].shape[1]))
            mask = out['src_super_mask']
            target, predicted = visibility[mask], out['src_super_overlap'][mask]
            entropy = -(target * (target + 1e-8).log() + (1 - target) * (1 - target + 1e-8).log())
            stats['target_entropy'].append(float(entropy.mean()))
            stats['coarse_bce'].append(float(torch.nn.functional.binary_cross_entropy(
                predicted, target)))
            stats['mu_mean'].append(float(predicted.mean()))
            stats['mu_std'].append(float(predicted.std()))
            stats['target_mean'].append(float(target.mean()))
            # scale-free: how well does mu rank the patches that are really visible?
            positive = target > 0.5
            if positive.any() and (~positive).any():
                order = torch.argsort(predicted)
                ranks = torch.empty_like(order, dtype=torch.float)
                ranks[order] = torch.arange(order.numel(), device=order.device).float()
                n_pos, n_neg = int(positive.sum()), int((~positive).sum())
                auc = (ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
                stats['auc'].append(float(auc))

    def show(key, fmt='{:.3f}'):
        return fmt.format(np.mean(stats[key])) if stats[key] else 'n/a'

    print(f'\npatch level ({args.batches * args.batch_size} pairs, matching radius {radius} m)')
    print(f'  points per patch before sampling   {show("patch_points", "{:.0f}")}')
    print(f'  points kept per patch (patch_size) {show("kept", "{:.0f}")}')
    print(f'  source points whose true partner survived the sampling  {show("reachable")}'
          '   <- ceiling on the inlier ratio')
    print(f'  source points the transport plan matches correctly      {show("picked")}')
    print(f'  of the reachable ones, correctly matched                {show("picked_of_reachable")}')
    print(f'  points whose match beats their slack entry               {show("confident_frac")}')
    print(f'  of those, correctly matched                             {show("confident_correct")}')
    print(f'  after the mutual check and deduplication: {show("emitted", "{:.0f}")} correspondences '
          f'per pair, {show("emitted_correct")} correct')

    print('\ncoarse overlap head')
    print(f'  predicted mu: mean {show("mu_mean")}, std {show("mu_std")}; '
          f'target mean {show("target_mean")}')
    print(f'  cross entropy {show("coarse_bce")} against a floor of {show("target_entropy")} '
          '(the targets are soft)')
    print(f'  ranking quality (AUC, 0.5 = chance): {show("auc")}')


if __name__ == '__main__':
    main()
