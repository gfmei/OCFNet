"""Which parameters does each loss term actually train?

A loss that reports a number but reaches no parameter is indistinguishable from a loss that
is learning slowly -- both show a flat curve. This backpropagates each term on its own and
reports the gradient norm arriving at the parts of the network it is supposed to teach.
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
inp = next(iter(DataLoader(val, batch_size=2, num_workers=0, shuffle=False, collate_fn=collate_pair_fn)))
model = OCFNet(cfg).to(dev); loss_fn = OCFLoss(cfg).to(dev)
src = make_sparse_tensor(inp['src_C'], inp['src_F'], device=dev)
tgt = make_sparse_tensor(inp['tgt_C'], inp['tgt_F'], device=dev)
corr = inp['correspondences'].long().to(dev)

groups = {
    'encoder':       [p for n, p in model.named_parameters() if n.startswith('encoder')],
    'decoder':       [p for n, p in model.named_parameters() if n.startswith('decoder')],
    'attention':     [p for n, p in model.named_parameters() if n.startswith('attention')],
    'overlap_head':  [p for n, p in model.named_parameters() if n.startswith('overlap_head')],
    'patch_refiner': [p for n, p in model.named_parameters() if n.startswith('patch_refiner')],
    'bin_score':     [p for n, p in model.named_parameters() if 'bin_score' in n],
}
terms = ['coarse_inlier_loss', 'inlier_loss', 'coarse_loss', 'fine_loss', 'coarse_overlap_loss', 'fine_overlap_loss',
         'descriptor_loss', 'infonce_loss', 'coarse_infonce_loss', 'pair_overlap_loss']

print(f'{"loss term":22s}' + ''.join(f'{g:>15s}' for g in groups))
for term in terms:
    out = model(src, tgt, correspondences=corr, src_xyz=inp['pcd_src'].to(dev),
                tgt_xyz=inp['pcd_tgt'].to(dev))
    stats = loss_fn(out, inp)
    if term not in stats:
        print(f'{term:22s}  <absent>'); continue
    value = stats[term]
    model.zero_grad(set_to_none=True)
    if not torch.is_tensor(value) or not value.requires_grad:
        print(f'{term:22s}' + '  NOT DIFFERENTIABLE'); continue
    value.backward(retain_graph=False)
    row = ''
    for g, params in groups.items():
        n = sum(float(p.grad.norm()**2) for p in params if p.grad is not None) ** 0.5
        row += f'{n:15.3e}' if n > 0 else f'{"ZERO":>15s}'
    print(f'{term:22s}{row}   (value {float(value):.3f})')
