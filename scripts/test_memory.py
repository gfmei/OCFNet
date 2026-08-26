"""Largest tensors the forward and backward actually build.

Nothing here may scale with the product of two fine-level point counts: a pair of clouds is
tens of thousands of voxels each, so any [N_fine, M_fine] intermediate is hundreds of
millions of entries. Every quadratic term must live at the super-point or patch level.
"""
import sys, yaml, torch
from easydict import EasyDict as edict
sys.path.append('/leonardo_scratch/fast/AIFPT_agrifood/code/OCFNet')
from torch.utils.data import DataLoader
from lib.spconv_utils import make_sparse_tensor
from models.ocfnet import OCFNet, OCFLoss
from datasets.dataloader import get_datasets, collate_pair_fn

raw = yaml.safe_load(open('configs/train/ocfnet_indoor.yaml')); cfg = {}
for k, v in raw.items(): cfg.update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)
cfg = edict(cfg); dev = 'cuda'
_, val, _ = get_datasets(cfg)
loader = DataLoader(val, batch_size=cfg.batch_size, num_workers=0, shuffle=False,
                    collate_fn=collate_pair_fn)
inp = next(iter(loader))
model = OCFNet(cfg).to(dev); loss_fn = OCFLoss(cfg).to(dev)
src = make_sparse_tensor(inp['src_C'], inp['src_F'], device=dev)
tgt = make_sparse_tensor(inp['tgt_C'], inp['tgt_F'], device=dev)

big = []
real_cdist, real_einsum = torch.cdist, torch.einsum
def watch(name, fn):
    def wrapped(*a, **k):
        out = fn(*a, **k)
        if torch.is_tensor(out) and out.numel() > 5_000_000:
            big.append((name, tuple(out.shape), out.numel()))
        return out
    return wrapped
torch.cdist, torch.einsum = watch('cdist', real_cdist), watch('einsum', real_einsum)

torch.cuda.reset_peak_memory_stats()
out = model(src, tgt, correspondences=inp['correspondences'].long().to(dev),
            src_xyz=inp['pcd_src'].to(dev), tgt_xyz=inp['pcd_tgt'].to(dev))
stats = loss_fn(out, inp)
peak_fwd = torch.cuda.max_memory_allocated() / 2**30
stats['loss'].backward()
peak = torch.cuda.max_memory_allocated() / 2**30
torch.cdist, torch.einsum = real_cdist, real_einsum

n_src = torch.bincount(src.indices[:, 0].long()).max().item()
n_sup = out['src_super_mask'].shape[1]; m_sup = out['tgt_super_mask'].shape[1]
P, K = out['patch_src_index'].shape
print(f'batch {cfg.batch_size}: fine voxels/sample up to {n_src}, super-points {n_sup}x{m_sup}')
print(f'peak memory  forward {peak_fwd:.2f} GiB   forward+backward {peak:.2f} GiB\n')
print('quadratic intermediates:')
print(f'  coarse cost / plan   [{cfg.batch_size}, {n_sup}, {m_sup}]   '
      f'{cfg.batch_size*n_sup*m_sup/1e6:.1f}M entries')
print(f'  patch transport      [{P}, {K+1}, {K+1}]   {P*(K+1)**2/1e6:.1f}M entries')
print(f'  a fine-level pair    [{cfg.batch_size}, {n_src}, ...]  would be '
      f'{cfg.batch_size*n_src*n_src/1e9:.1f}G entries  <- never built')
print('\ntensors over 5M entries built by cdist/einsum:')
for name, shape, n in sorted(set(big), key=lambda x: -x[2])[:8]:
    print(f'  {name:8s} {shape}  {n/1e6:.1f}M')
if not big:
    print('  none')
