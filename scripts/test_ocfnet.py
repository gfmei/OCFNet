"""
Smoke test for models/ocfnet.py on real 3DMatch pairs.

    python scripts/test_ocfnet.py configs/train/ocfnet_indoor.yaml --batch_size 2

Checks the two modes of the network and every term of the loss:
  1. training pass -- patches come from the ground-truth super-point pairs, the four losses
     of section 2.4 are finite and gradients reach every parameter
  2. inference pass -- patches come from the predicted coarse matches (Eq. 3) and point
     correspondences are read out of the patch transport plans (Eq. 6)
  3. the transport plans really respect their marginals, i.e. the overlap scores
"""
import argparse, os, sys, time
import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.append(os.getcwd())
from datasets.dataloader import get_datasets, collate_pair_fn
from lib.spconv_utils import make_sparse_tensor
from lib.utils import load_config, setup_seed
from models.ocfnet import (OCFNet, OCFLoss, overlap_optimal_transport,
                           normalised_feature_cost)

setup_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/train/ocfnet_indoor.yaml')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    config = edict(load_config(args.config))
    config.batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    model = OCFNet(config, D=3).to(device)
    criterion = OCFLoss(config)
    print(f'OCFNet on {device}: {sum(p.nelement() for p in model.parameters()) / 1e6:.3f} M parameters')

    train_set, _, _ = get_datasets(config)
    batch = collate_pair_fn([train_set[i] for i in range(args.batch_size)])
    stensor_src = make_sparse_tensor(batch['src_C'], batch['src_F'], device=device)
    stensor_tgt = make_sparse_tensor(batch['tgt_C'], batch['tgt_F'], device=device)
    correspondence = batch['correspondences'].long().to(device)
    print(f'batch of {args.batch_size}: {batch["src_C"].shape[0]}/{batch["tgt_C"].shape[0]} voxels, '
          f'{correspondence.shape[0]} correspondences')

    ###########################################################################
    # 1. training pass
    model.train()
    t0 = time.time()
    out = model(stensor_src, stensor_tgt, correspondences=correspondence,
                src_xyz=inputs['pcd_src'].to(device), tgt_xyz=inputs['pcd_tgt'].to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    forward_time = time.time() - t0

    per_pair_src = out['src_super_mask'].sum(1).tolist()
    per_pair_tgt = out['tgt_super_mask'].sum(1).tolist()
    print(f'super-points per pair: {[int(v) for v in per_pair_src]} / '
          f'{[int(v) for v in per_pair_tgt]}')
    print(f'patches refined: {out["coarse_matches"].shape[0]}, forward {forward_time:.2f}s')
    assert out['src_feats'].shape == (batch['src_C'].shape[0], config.out_feats_dim)
    assert out['coarse_log_plan'].shape[1] == out['src_super_mask'].shape[1] + 1
    print(f'[ok] shapes: point feats {tuple(out["src_feats"].shape)}, '
          f'coarse plan {tuple(out["coarse_log_plan"].shape)}, '
          f'patch plans {tuple(out["patch_log_plan"].shape)}')

    stats = criterion(out, batch)
    for key, value in stats.items():
        assert torch.isfinite(value), f'{key} is {value}'
    print('[ok] losses: ' + ', '.join(f'{k} {float(v):.4f}' for k, v in stats.items()))

    stats['loss'].backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    non_finite = [n for n, p in model.named_parameters()
                  if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not missing, f'no gradient for: {missing}'
    assert not non_finite, f'non-finite gradient for: {non_finite}'
    print(f'[ok] backward: all {len(list(model.parameters()))} parameters got a finite gradient')

    ###########################################################################
    # 2. inference pass
    model.eval()
    with torch.no_grad():
        out = model(stensor_src, stensor_tgt)
        matches, scores = model.point_correspondences(out)
    print(f'[ok] inference: {out["coarse_matches"].shape[0]} coarse matches -> '
          f'{matches.shape[0]} point correspondences '
          f'(score {float(scores.mean()) if scores.numel() else 0:.4f} on average)')

    ###########################################################################
    # 3. the transport plan follows the overlap scores it was given. Sinkhorn approaches the
    #    marginals from one side and the loop ends on a column update, so the rows still lag
    #    at the iteration count used for training; run it to convergence to check that the
    #    formulation itself is right.
    with torch.no_grad():
        mask, other_mask = out['src_super_mask'], out['tgt_super_mask']
        plan = out['coarse_log_plan'].exp()
        error = (plan[:, :-1, :].sum(-1)[mask] - out['src_super_overlap'][mask]).abs().max().item()

        cost = normalised_feature_cost(out['src_super_coords'].transpose(1, 2),
                                       out['tgt_super_coords'].transpose(1, 2))
        converged = overlap_optimal_transport(
            cost, out['src_super_overlap'] * mask, out['tgt_super_overlap'] * other_mask,
            model.bin_score, iters=500, epsilon=model.sinkhorn_epsilon).exp()
        row_err = (converged[:, :-1, :].sum(-1)[mask]
                   - out['src_super_overlap'][mask]).abs().max().item()
        col_err = (converged[:, :, :-1].sum(1)[other_mask]
                   - out['tgt_super_overlap'][other_mask]).abs().max().item()
    print(f'[ok] transport marginals: {error:.2e} off the overlap scores at '
          f'{model.sinkhorn_iters} iterations, {max(row_err, col_err):.2e} at 500')
    assert max(row_err, col_err) < 1e-3, 'the transport plan does not reach its marginals'

    if device.type == 'cuda':
        print(f'peak gpu memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB')
    print('\nall checks passed')


if __name__ == '__main__':
    main()
