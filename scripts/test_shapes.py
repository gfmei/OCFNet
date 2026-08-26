"""Assert the shape of every tensor the OCFNet forward produces.

The dense batch is point-major ([B, N, C]); models/attention.py is the one channels-first
island, because the Predator backbone shares it. A silent transpose anywhere in between
would still typecheck and quietly corrupt the transport cost, so every intermediate is
pinned here.
"""
import sys, yaml, torch
from easydict import EasyDict as edict
sys.path.append('/leonardo_scratch/fast/AIFPT_agrifood/code/OCFNet')
from torch.utils.data import DataLoader
from lib.spconv_utils import make_sparse_tensor
from models.ocfnet import OCFNet, OCFLoss, normalised_feature_cost
from models.common import to_dense_batch, from_dense_batch
from datasets.dataloader import get_datasets, collate_pair_fn

raw = yaml.safe_load(open('configs/train/ocfnet_indoor.yaml')); cfg = {}
for k, v in raw.items(): cfg.update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)
cfg = edict(cfg); dev = 'cuda'
train, val, bench = get_datasets(cfg)
loader = DataLoader(val, batch_size=3, num_workers=0, shuffle=False, collate_fn=collate_pair_fn)
inp = next(iter(loader))

model = OCFNet(cfg).to(dev)
loss_fn = OCFLoss(cfg).to(dev)
src = make_sparse_tensor(inp['src_C'], inp['src_F'], device=dev)
tgt = make_sparse_tensor(inp['tgt_C'], inp['tgt_F'], device=dev)
corr = inp['correspondences'].long().to(dev)
sx, tx = inp['pcd_src'].to(dev), inp['pcd_tgt'].to(dev)
B, C = 3, cfg.out_feats_dim   # per-point descriptor width
CA = cfg.attention_feats_dim  # super-point attention width
fails = []

def check(name, got, want):
    ok = tuple(got) == tuple(want)
    print(f'  {"ok " if ok else "FAIL"} {name:26s} {tuple(got)}  expected {tuple(want)}')
    if not ok: fails.append(name)

for mode in ('train', 'eval'):
    model.train(mode == 'train')
    print(f'\n=== {mode} mode ===')
    with torch.set_grad_enabled(mode == 'train'):
        out = model(src, tgt, correspondences=corr if mode == 'train' else None,
                    src_xyz=sx, tgt_xyz=tx)
    N = out['src_super_mask'].shape[1]; M = out['tgt_super_mask'].shape[1]
    npts_s, npts_t = src.indices.shape[0], tgt.indices.shape[0]
    check('src_super_mask', out['src_super_mask'].shape, (B, N))
    check('src_super_coords', out['src_super_coords'].shape, (B, N, 3))
    check('tgt_super_coords', out['tgt_super_coords'].shape, (B, M, 3))
    check('src_super_overlap', out['src_super_overlap'].shape, (B, N))
    check('tgt_super_overlap', out['tgt_super_overlap'].shape, (B, M))
    check('coarse_log_plan', out['coarse_log_plan'].shape, (B, N + 1, M + 1))
    check('src_feats (per point)', out['src_feats'].shape, (npts_s, C))
    check('tgt_feats (per point)', out['tgt_feats'].shape, (npts_t, C))
    check('src_overlap (per point)', out['src_overlap'].shape, (npts_s,))
    check('src_patch_id', out['src_patch_id'].shape, (npts_s,))
    P, K = out['patch_src_index'].shape
    check('patch_src_index', out['patch_src_index'].shape, (P, K))
    check('patch_tgt_index', out['patch_tgt_index'].shape, (P, K))
    check('patch_src_valid', out['patch_src_valid'].shape, (P, K))
    check('patch_log_plan', out['patch_log_plan'].shape, (P, K + 1, K + 1))
    check('coarse_matches', out['coarse_matches'].shape, (P, 2))
    print(f'  .. {P} patch pairs of {K} points, {N}x{M} super-points')

    # The transport cost pairs *super-points*, so it must be [B, N, M] -- the same N and M
    # as coarse_log_plan minus its slack row and column. Building it from point-major
    # features at the super-point resolution is what the coarse stage does; feeding it
    # fine-level features would still typecheck and silently cost a 600M-entry tensor.
    sf = torch.randn(B, N, CA, device=dev)
    tf = torch.randn(B, M, CA, device=dev)
    cost = normalised_feature_cost(sf, tf)
    check('coarse feature cost', cost.shape, (B, N, M))
    check('  vs coarse_log_plan', tuple(x - 1 for x in out['coarse_log_plan'].shape[1:]), (N, M))

    m, s = model.point_correspondences(out)
    check('correspondences', m.shape, (m.shape[0], 2))
    check('confidence', s.shape, (m.shape[0],))
    assert m.shape[0] == s.shape[0]
    if m.numel():
        assert int(m[:, 0].max()) < npts_s and int(m[:, 1].max()) < npts_t, 'index out of range!'
        print(f'  ok  correspondence indices in range ({m.shape[0]} of them)')

    if mode == 'train':
        stats = loss_fn(out, inp)
        bad = [k for k, v in stats.items() if not torch.isfinite(torch.as_tensor(v))]
        print(f'  {"ok " if not bad else "FAIL"} loss terms finite ({len(stats)} terms)' +
              (f' -- non-finite: {bad}' if bad else ''))
        if bad: fails.append('loss')

# the Predator backbone must still work through the channels-first wrappers
from models.resunet import ResUNetBN2C
pred_cfg = edict({**cfg, 'model': 'ResUNetBN2C'})
pm = ResUNetBN2C(pred_cfg, D=3).to(dev).eval()
with torch.no_grad():
    f0, f1, ov, sal = pm(src, tgt)
print('\n=== Predator backbone through the compat wrappers ===')
check('predator src feats', f0.shape, (npts_s, cfg.out_feats_dim))
check('predator overlap', ov.shape, (npts_s + npts_t,))

print('\n' + ('ALL SHAPE CHECKS PASSED' if not fails else f'FAILURES: {fails}'))
sys.exit(1 if fails else 0)
