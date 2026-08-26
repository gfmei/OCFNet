import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
  if norm_type == 'BN':
    return SparseBatchNorm(num_feats, momentum=bn_momentum)
  elif norm_type == 'IN':
    return SparseInstanceNorm(num_feats)
  else:
    raise ValueError(f'Type {norm_type}, not defined')


class SparseBatchNorm(spconv.SparseModule):
  """spconv counterpart of ME.MinkowskiBatchNorm: BN over all active voxels of the batch."""

  def __init__(self, num_feats, momentum=0.05, eps=1e-5):
    super().__init__()
    self.bn = nn.BatchNorm1d(num_feats, eps=eps, momentum=momentum)

  def forward(self, x):
    return x.replace_feature(self.bn(x.features))


class SparseInstanceNorm(spconv.SparseModule):
  """spconv counterpart of ME.MinkowskiInstanceNorm.

  Normalises the active voxels of every (sample, channel) pair to zero mean / unit
  variance, then applies a per-channel affine, exactly like the MinkowskiEngine module
  (which builds it out of global average poolings and uses eps=1e-8 under the sqrt).
  """

  def __init__(self, num_feats, eps=1e-8):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(1, num_feats))
    self.bias = nn.Parameter(torch.zeros(1, num_feats))
    self.eps = eps

  def forward(self, x):
    feats = x.features
    batch_ids = x.indices[:, 0].long()
    shape = (x.batch_size, feats.shape[1])

    counts = torch.zeros(x.batch_size, 1, dtype=feats.dtype, device=feats.device)
    counts = counts.index_add_(0, batch_ids, torch.ones_like(feats[:, :1])).clamp_(min=1)

    mean = torch.zeros(shape, dtype=feats.dtype, device=feats.device)
    mean = mean.index_add_(0, batch_ids, feats) / counts
    centered = feats - mean[batch_ids]

    var = torch.zeros(shape, dtype=feats.dtype, device=feats.device)
    var = var.index_add_(0, batch_ids, centered * centered) / counts

    out = centered * torch.rsqrt(var[batch_ids] + self.eps)
    return x.replace_feature(out * self.weight + self.bias)


def to_dense_batch(features, batch_ids, batch_size):
  """Scatter [N, C] sparse rows into a padded dense batch [B, N_max, C].

  Returns (dense, mask, index) where mask[b, n] marks the real points and `index` is the
  bookkeeping needed by `from_dense_batch` to put the rows back where they came from. No
  assumption is made about the rows being grouped by sample: spconv is free to reorder
  voxels when it downsamples.
  """
  order = torch.argsort(batch_ids, stable=True)
  sorted_ids = batch_ids[order]
  counts = torch.bincount(sorted_ids, minlength=batch_size)
  n_max = int(counts.max().item())
  starts = torch.cumsum(counts, dim=0) - counts
  pos = torch.arange(features.shape[0], device=features.device) - starts[sorted_ids]

  dense = features.new_zeros(batch_size, n_max, features.shape[1])
  dense[sorted_ids, pos] = features[order]
  mask = torch.zeros(batch_size, n_max, dtype=torch.bool, device=features.device)
  mask[sorted_ids, pos] = True
  return dense, mask, (order, sorted_ids, pos)


def from_dense_batch(dense, index):
  """Inverse of `to_dense_batch`: [B, N_max, C] -> [N, C] in the original row order."""
  order, sorted_ids, pos = index
  gathered = dense[sorted_ids, pos]
  out = torch.empty_like(gathered)
  out[order] = gathered
  return out


def masked_instance_norm(x, mask, eps=1e-5):
  """nn.InstanceNorm1d (affine=False) restricted to the valid entries of a padded batch.

  x:    [B, N, C], point-major
  mask: [B, N], True on real points
  Statistics are taken over the points of a sample, per channel. Padding must not enter
  them, otherwise the normalisation of a sample would depend on how long the other samples
  of the batch happen to be. Padded entries come back as zeros.
  """
  m = mask[..., None].to(x.dtype)
  count = m.sum(dim=-2, keepdim=True).clamp(min=1)
  mean = (x * m).sum(dim=-2, keepdim=True) / count
  centered = (x - mean) * m
  var = (centered * centered).sum(dim=-2, keepdim=True) / count
  return centered * torch.rsqrt(var + eps)


def sparse_relu(x):
  """spconv counterpart of MinkowskiEngine.MinkowskiFunctional.relu."""
  return x.replace_feature(F.relu(x.features))


def sparse_cat(*tensors):
  """spconv counterpart of ME.cat.

  Concatenates features along the channel dimension. The inputs must share their
  coordinates, which here is guaranteed by construction: a SparseInverseConv3d
  restores exactly the indices (same order) of the SparseConv3d it is keyed to.
  """
  out = tensors[0]
  return out.replace_feature(torch.cat([t.features for t in tensors], dim=1))
