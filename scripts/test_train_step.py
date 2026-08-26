"""
End-to-end check: a few real training iterations on 3DMatch with the spconv backbone.

    python scripts/test_train_step.py configs/train/indoor.yaml --iters 3

Exercises the whole path -- dataset -> collate -> spconv tensors -> model -> circle /
overlap / saliency losses -> backward -> optimiser step -- and prints the stats so the
numbers can be compared against a MinkowskiEngine run.
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch import optim
from easydict import EasyDict as edict

sys.path.append(os.getcwd())
from datasets.dataloader import get_datasets, collate_pair_fn
from lib.loss import MetricLoss
from lib.spconv_utils import make_sparse_tensor
from lib.utils import load_config, setup_seed
from models import load_model

setup_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/train/indoor.yaml')
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=None, help='overrides the config')
    args = parser.parse_args()

    config = edict(load_config(args.config))
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {config.device}, dataset root {config.root}')

    model = load_model(config.model)(config, D=3).to(config.device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    desc_loss = MetricLoss(config)

    train_set, _, _ = get_datasets(config)
    loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                                         num_workers=args.num_workers, collate_fn=collate_pair_fn,
                                         drop_last=False)
    print(f'{len(train_set)} training pairs')

    model.train()
    loader_iter = iter(loader)
    for it in range(args.iters):
        t0 = time.time()
        inputs = next(loader_iter)

        sinput_src = make_sparse_tensor(inputs['src_C'], inputs['src_F'], device=config.device)
        sinput_tgt = make_sparse_tensor(inputs['tgt_C'], inputs['tgt_F'], device=config.device)
        src_feats, tgt_feats, scores_overlap, scores_saliency = model(sinput_src, sinput_tgt)

        src_pcd = inputs['pcd_src'].to(config.device)
        tgt_pcd = inputs['pcd_tgt'].to(config.device)
        rot = inputs['rot'].to(config.device)
        trans = inputs['trans'].to(config.device)
        correspondence = inputs['correspondences'].long().to(config.device)

        # the losses index the model outputs with point indices, which only works if the
        # backbone returns one row per input voxel, in the input order
        assert src_feats.shape[0] == src_pcd.shape[0], (src_feats.shape, src_pcd.shape)
        assert tgt_feats.shape[0] == tgt_pcd.shape[0], (tgt_feats.shape, tgt_pcd.shape)

        stats = desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans,
                          scores_overlap, scores_saliency, inputs['scale'], inputs['len_batch'])
        loss = (stats['circle_loss'] * config.w_circle_loss
                + stats['overlap_loss'] * config.w_overlap_loss
                + stats['saliency_loss'] * config.w_saliency_loss)
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(loss), f'loss is {loss.item()}'
        optimizer.step()

        if config.device.type == 'cuda':
            torch.cuda.synchronize()
        printable = {k: round(float(v.detach()) if torch.is_tensor(v) else float(v), 4)
                     for k, v in stats.items()}
        print(f'iter {it}: {src_feats.shape[0]}/{tgt_feats.shape[0]} voxels, '
              f'{correspondence.shape[0]} correspondences, loss {float(loss):.4f}, '
              f'{time.time() - t0:.2f}s\n        {printable}')

    if config.device.type == 'cuda':
        print(f'peak gpu memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    print('\ntraining loop runs')


if __name__ == '__main__':
    main()
