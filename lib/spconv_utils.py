"""
spconv replacements for the MinkowskiEngine helpers this code base used to rely on.

    MinkowskiEngine                            ->  here
    ME.utils.sparse_quantize(coords, ...)      ->  sparse_quantize(coords, ...)
    ME.utils.sparse_collate(coords, feats)     ->  sparse_collate(coords, feats)
    ME.SparseTensor(feats, coordinates=coords) ->  make_sparse_tensor(coords, feats)

The only real semantic difference between the two libraries is the coordinate system:
MinkowskiEngine hashes coordinates and happily takes negative ones, whereas spconv
indexes a (virtual) dense grid, so coordinates must lie inside [0, spatial_shape).
Point clouds are centred around the origin, hence `make_sparse_tensor` shifts every
batch into the positive octant. The network only uses those coordinates to build a
attention (models/attention.py), which is invariant to a global translation, so the shift
does not change any output.
"""
import numpy as np
import torch
from spconv.pytorch import SparseConvTensor

# ResUNet2 halves the resolution three times, so the bottleneck lives on a 1/8 grid.
NUM_DOWNSAMPLES = 8


def sparse_quantize(coords, return_index=False, return_inverse=False):
    """Voxelise `coords` (already divided by the voxel size), keeping one point per voxel.

    Drop-in replacement for ME.utils.sparse_quantize: with return_index=True it returns
    (unique voxel coordinates, index of one representative input point per voxel).
    ME picks the representative by hash order, numpy by sorted order -- both are
    arbitrary, and everything downstream is indexed consistently with `sel`.
    """
    coords = np.floor(np.asarray(coords)).astype(np.int32)
    return np.unique(coords, axis=0,
                     return_index=return_index, return_inverse=return_inverse)


def sparse_collate(coords, feats, dtype=torch.int32):
    """Stack a list of clouds into one batch, prepending the batch index to the coords.

    Drop-in replacement for ME.utils.sparse_collate; returns (coords [N, 4], feats [N, C])
    with coords laid out as (batch_index, x, y, z), which is what spconv expects.
    """
    batched_coords, batched_feats = [], []
    for batch_id, (coord, feat) in enumerate(zip(coords, feats)):
        coord = torch.as_tensor(np.asarray(coord))
        feat = torch.as_tensor(np.asarray(feat))
        batch_col = torch.full((coord.shape[0], 1), batch_id, dtype=dtype)
        batched_coords.append(torch.cat([batch_col, coord.to(dtype)], dim=1))
        batched_feats.append(feat.float())
    return torch.cat(batched_coords, dim=0), torch.cat(batched_feats, dim=0)


def make_sparse_tensor(coords, feats, device=None, pad_multiple=NUM_DOWNSAMPLES):
    """Build the spconv equivalent of ME.SparseTensor(feats, coordinates=coords).

    coords: [N, 4] integer (batch_index, x, y, z), may contain negative coordinates
    feats:  [N, C] float features
    """
    coords = torch.as_tensor(coords)
    feats = torch.as_tensor(feats)
    if device is not None:
        coords, feats = coords.to(device), feats.to(device)
    assert coords.dim() == 2 and coords.shape[1] == 4, \
        f'expected [N, 4] (batch, x, y, z) coordinates, got {tuple(coords.shape)}'

    xyz = coords[:, 1:].int()
    # spconv grids start at the origin, so shift the batch into the positive octant. The
    # shift is rounded down to a multiple of the total stride: a stride-2 convolution maps a
    # voxel to floor(coord / 2), so an arbitrary shift would re-align the cloud on the grid
    # and change which voxels the encoder produces. Moving by a multiple of `pad_multiple`
    # translates every level by an exact integer instead, which convolutions are equivariant
    # to -- a cloud then yields the same features whether it is processed alone or batched.
    offset = torch.div(xyz.min(dim=0, keepdim=True).values, pad_multiple,
                       rounding_mode='floor') * pad_multiple
    xyz = xyz - offset
    indices = torch.cat([coords[:, :1].int(), xyz], dim=1).contiguous()

    # A grid whose side is (m * pad_multiple + 1) survives log2(pad_multiple) halvings
    # without losing the voxels on its upper border: a stride-2 conv maps the grid to
    # ceil(shape / 2) cells and the last input cell would fall outside an even shape.
    hi = xyz.max(dim=0).values.tolist()
    spatial_shape = [(int(h) // pad_multiple + 1) * pad_multiple + 1 for h in hi]
    batch_size = int(coords[:, 0].max().item()) + 1

    return SparseConvTensor(feats.float().contiguous(), indices, spatial_shape, batch_size)
