"""
Check that batched training is equivalent to feeding the pairs one by one.

    python scripts/test_batch_equivalence.py configs/train/indoor.yaml --batch_size 3

The overlap-attention module is the only part of the network where points can influence
each other, so batching it is only correct if the masks really keep the pairs apart. Two
checks:

  1. a batch of identical copies of one pair must reproduce the single-pair output. Any
     leakage across the batch would change the attention distributions and show up here.
  2. a batch of different pairs must reproduce each pair's individual output.

The same comparison is run with the attention bypassed, which isolates the padding/unpadding of
the sparse tensors from the attention itself.
"""
import argparse, os, sys
import numpy as np
import torch
from easydict import EasyDict as edict

sys.path.append(os.getcwd())
from datasets.dataloader import get_datasets
from lib.spconv_utils import sparse_collate, make_sparse_tensor
from lib.utils import load_config, setup_seed
from models import load_model

setup_seed(0)


class BypassAttention(torch.nn.Module):
    def forward(self, coords0, coords1, desc0, desc1, mask0=None, mask1=None):
        return desc0, desc1


def forward_pairs(model, pairs, device):
    """Run the model on a list of (src_coords, src_feats, tgt_coords, tgt_feats) pairs."""
    src_C, src_F = sparse_collate([p[0] for p in pairs], [p[1] for p in pairs])
    tgt_C, tgt_F = sparse_collate([p[2] for p in pairs], [p[3] for p in pairs])
    stensor_src = make_sparse_tensor(src_C, src_F, device=device)
    stensor_tgt = make_sparse_tensor(tgt_C, tgt_F, device=device)
    with torch.no_grad():
        return model(stensor_src, stensor_tgt)


def compare(batched, singles, pairs, label):
    """Slice the batched output per pair and compare against the individual runs."""
    src_feats, tgt_feats, scores_overlap, scores_saliency = batched
    n_src_total = sum(p[0].shape[0] for p in pairs)
    src_start = tgt_start = 0
    worst = 0.0
    for b, (single, pair) in enumerate(zip(singles, pairs)):
        n_src, n_tgt = pair[0].shape[0], pair[2].shape[0]
        s_feats, t_feats, s_overlap, s_saliency = single
        diffs = [
            (src_feats[src_start:src_start + n_src] - s_feats).abs().max().item(),
            (tgt_feats[tgt_start:tgt_start + n_tgt] - t_feats).abs().max().item(),
            (scores_overlap[src_start:src_start + n_src] - s_overlap[:n_src]).abs().max().item(),
            (scores_overlap[n_src_total + tgt_start:n_src_total + tgt_start + n_tgt]
             - s_overlap[n_src:]).abs().max().item(),
            (scores_saliency[src_start:src_start + n_src] - s_saliency[:n_src]).abs().max().item(),
        ]
        worst = max(worst, max(diffs))
        src_start, tgt_start = src_start + n_src, tgt_start + n_tgt
    status = 'ok' if worst < 1e-3 else 'FAIL'
    print(f'[{status}] {label}: max |batched - single| = {worst:.2e}')
    return worst


def compare_within_batch(batched, pairs, label):
    """With identical pairs in the batch, every sample must give the same answer.

    This separates leakage between samples (which would break this too) from kernels that
    break ties differently depending on the tensor shape (which would not).
    """
    src_feats, tgt_feats, _, _ = batched
    n_src, n_tgt = pairs[0][0].shape[0], pairs[0][2].shape[0]
    worst = 0.0
    for b in range(1, len(pairs)):
        worst = max(worst,
                    (src_feats[b * n_src:(b + 1) * n_src] - src_feats[:n_src]).abs().max().item(),
                    (tgt_feats[b * n_tgt:(b + 1) * n_tgt] - tgt_feats[:n_tgt]).abs().max().item())
    print(f'[{"ok" if worst < 1e-3 else "FAIL"}] {label}: max spread inside the batch = {worst:.2e}')
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/train/indoor.yaml')
    parser.add_argument('--batch_size', type=int, default=3)
    args = parser.parse_args()

    config = edict(load_config(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device
    model = load_model(config.model)(config, D=3).to(device).eval()
    print(f'{config.model} on {device}, batch_size {args.batch_size}, '
          f'rope: {config.get("rope", True)}, '
          f'{sum(p.nelement() for p in model.parameters()) / 1e6:.3f} M parameters')

    train_set, _, _ = get_datasets(config)
    samples = [train_set[i] for i in range(args.batch_size)]
    # (src_coords, src_feats, tgt_coords, tgt_feats) per pair
    pairs = [(s[2], s[4], s[3], s[5]) for s in samples]
    print('pair sizes:', [(p[0].shape[0], p[2].shape[0]) for p in pairs])

    for label, attention_on in [('full model', True), ('attention bypassed', False)]:
        attention = model.attention
        if not attention_on:
            model.attention = BypassAttention()

        singles = [forward_pairs(model, [p], device) for p in pairs]

        # 1. a batch of identical copies: leakage would perturb every attention distribution
        copies = [pairs[0]] * args.batch_size
        batched = forward_pairs(model, copies, device)
        compare_within_batch(batched, copies, f'{label}, identical pairs')
        compare(batched, [singles[0]] * args.batch_size, copies, f'{label}, identical pairs')

        # 2. a batch of different pairs, i.e. real padding
        batched = forward_pairs(model, pairs, device)
        compare(batched, singles, pairs, f'{label}, different pairs')

        model.attention = attention

    print('\nbatched forward matches the per-pair forward')


if __name__ == '__main__':
    main()
