"""
How large a batch fits, and what it buys.

    python scripts/bench_batch_size.py configs/train/indoor.yaml --sizes 4 8 16 32

Runs a real forward+backward (model and loss) on several batches drawn from 3DMatch and
reports peak memory and time per pair. Several batches per size, because memory follows the
number of voxels and the clouds differ a lot in size -- the maximum is what has to fit.
"""
import argparse, os, sys, time
import torch
from easydict import EasyDict as edict

sys.path.append(os.getcwd())
from datasets.dataloader import get_datasets, collate_pair_fn
from lib.spconv_utils import make_sparse_tensor
from lib.utils import load_config, setup_seed
from models import load_model

setup_seed(0)


def run_predator(model, batch, device, config):
    from lib.loss import MetricLoss
    criterion = getattr(run_predator, 'criterion', None) or MetricLoss(config)
    run_predator.criterion = criterion
    src = make_sparse_tensor(batch['src_C'], batch['src_F'], device=device)
    tgt = make_sparse_tensor(batch['tgt_C'], batch['tgt_F'], device=device)
    src_feats, tgt_feats, overlap, saliency = model(src, tgt)
    stats = criterion(batch['pcd_src'].to(device), batch['pcd_tgt'].to(device),
                      src_feats, tgt_feats, batch['correspondences'].long().to(device),
                      batch['rot'].to(device), batch['trans'].to(device), overlap, saliency,
                      batch['scale'], batch['len_batch'])
    return (stats['circle_loss'] * config.w_circle_loss
            + stats['overlap_loss'] * config.w_overlap_loss
            + stats['saliency_loss'] * config.w_saliency_loss)


def run_ocfnet(model, batch, device, config):
    from models.ocfnet import OCFLoss
    criterion = getattr(run_ocfnet, 'criterion', None) or OCFLoss(config)
    run_ocfnet.criterion = criterion
    src = make_sparse_tensor(batch['src_C'], batch['src_F'], device=device)
    tgt = make_sparse_tensor(batch['tgt_C'], batch['tgt_F'], device=device)
    out = model(src, tgt, correspondences=batch['correspondences'].long().to(device))
    return criterion(out, batch)['loss']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, nargs='?', default='configs/train/indoor.yaml')
    parser.add_argument('--sizes', type=int, nargs='+', default=[4, 8, 16, 32])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()

    config = edict(load_config(args.config))
    device = torch.device('cuda')
    config.device = device
    model = load_model(config.model)(config, D=3).to(device).train()
    step = run_ocfnet if config.model == 'OCFNet' else run_predator
    print(f'{config.model} on {torch.cuda.get_device_name(0)} '
          f'({torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f} GiB)')

    train_set, _, _ = get_datasets(config)
    print(f'{"batch":>6} {"voxels":>9} {"peak GiB":>10} {"s/iter":>8} {"s/pair":>8}')
    index = 0
    for size in args.sizes:
        peak, seconds, voxels, done = 0.0, 0.0, 0, 0
        for _ in range(args.repeats):
            batch = collate_pair_fn([train_set[(index + i) % len(train_set)] for i in range(size)])
            index += size
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.time()
            try:
                loss = step(model, batch, device, config)
                loss.backward()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'{size:6d} {"":>9} {"OUT OF MEMORY":>10}')
                done = -1
                break
            seconds = max(seconds, time.time() - start)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
            voxels = max(voxels, batch['src_C'].shape[0] + batch['tgt_C'].shape[0])
            done += 1
        if done > 0:
            print(f'{size:6d} {voxels:9d} {peak:10.2f} {seconds:8.2f} {seconds / size:8.3f}')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
