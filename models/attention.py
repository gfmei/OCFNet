"""
Overlap-attention module: masked multi-head self-attention with a 3D rotary position
embedding, alternating with cross-attention between the two clouds.

Differences to the original Predator implementation:

  * a 'self' layer is plain self-attention over the points of one cloud. Predator aggregated
    over a kNN graph there, which was the expensive part (it materialised a points-by-points
    tensor) and was also ill-defined under batching: voxels on a regular grid produce massive
    distance ties, which torch.topk resolved by storage position -- and spconv makes no
    promise about the order in which it emits voxels.
  * positions enter through RoPE-3D (following https://github.com/YilmazKadir/Volt): the
    query and key of every head are rotated by an angle that is linear in the voxel
    coordinate, so attention scores depend on the *relative* position of two points. Being
    relative, it is invariant to the shift that lib/spconv_utils.make_sparse_tensor applies.
  * every operation takes an optional mask [B, N] marking the real points of a padded batch,
    so several pairs can be trained at once without any value crossing between them:
    attention scores over padded keys are set to -1e9 before the softmax, and the instance
    norms only see valid entries (see models/common.masked_instance_norm).

Cross-attention deliberately gets no position embedding: the two clouds live in different
coordinate frames until they are registered, so a relative-position bias between them would
be meaningless.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import torch.utils.checkpoint as checkpoint
from models.common import masked_instance_norm


class RoPE3D(nn.Module):
    """Axial rotary position embedding for integer voxel coordinates.

    The head dimension is cut into `freq_split` groups of complex pairs, one group per axis,
    each rotated by (coordinate * frequency). Frequencies run from 1 down to 1/theta.
    """

    def __init__(self, head_dim: int, theta: float = 100.0, freq_split=None,
                 max_grid_size=(1024, 1024, 512)):
        super().__init__()
        assert head_dim % 2 == 0, f'RoPE needs an even head dimension, got {head_dim}'
        pairs = head_dim // 2
        if freq_split is None:
            # Volt's 3/8, 3/8, 2/8 split: indoor scenes vary less along z. With few pairs
            # that would leave an axis without a single frequency, so spread them evenly.
            n_xy = round(pairs * 3 / 8)
            freq_split = (n_xy, n_xy, pairs - 2 * n_xy)
            if min(freq_split) < 1:
                n_x = (pairs + 2) // 3
                n_y = (pairs - n_x + 1) // 2
                freq_split = (n_x, n_y, pairs - n_x - n_y)
        assert sum(freq_split) == pairs, f'{freq_split} does not add up to {pairs} pairs'
        self.freq_split = freq_split
        self.max_grid_size = max_grid_size

        for axis, (n_freqs, max_pos) in enumerate(zip(freq_split, max_grid_size)):
            freqs = 1.0 / theta ** torch.linspace(0, 1, n_freqs)
            angles = torch.outer(torch.arange(max_pos).float(), freqs)
            self.register_buffer(f'cis_{axis}', torch.polar(torch.ones_like(angles), angles),
                                 persistent=False)

    def forward(self, indices):
        """indices: [B, N, 3] voxel coordinates -> complex [B, N, head_dim // 2]."""
        indices = indices.long()
        cis = [getattr(self, f'cis_{axis}')[indices[..., axis].clamp(0, size - 1)]
               for axis, size in enumerate(self.max_grid_size)]
        return torch.cat(cis, dim=-1)


def apply_rope(x, freqs_cis):
    """Rotate [B, heads, N, head_dim] queries/keys by the per-point angles of `freqs_cis`."""
    x_ = torch.view_as_complex(x.float().contiguous().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_ * freqs_cis[:, None]).flatten(-2).type_as(x)


class PointNorm(nn.Module):
    """Instance norm over the points of a padded batch, [B, N, C].

    A marker layer: `apply_mlp` runs it through `masked_instance_norm`, which needs the
    mask that nn.Module.forward does not carry.
    """
    def forward(self, x):
        return masked_instance_norm(x, x.new_ones(x.shape[:-1], dtype=torch.bool))


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron over the channel axis of a [B, N, C] tensor.

    A 1x1 convolution over points is exactly a per-point linear map; writing it as one
    keeps the whole module point-major, so nothing has to be transposed.
    """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(PointNorm())
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def apply_mlp(mlp, x, mask=None):
    """Run an `MLP` sequential, giving its instance norms the padding mask."""
    if mask is None:
        mask = x.new_ones(x.shape[:-1], dtype=torch.bool)
    for layer in mlp:
        x = masked_instance_norm(x, mask) if isinstance(layer, PointNorm) else layer(x)
    return x


class MultiHeadedAttention(nn.Module):
    """Multi-head attention on top of torch's fused scaled_dot_product_attention.

    The projections stay explicit because RoPE has to rotate the queries and keys after
    they are projected, which nn.MultiheadAttention does not expose. The attention itself
    (softmax, masking, dropout-free) is torch's, so it picks the memory-efficient kernel
    and never materialises the [B, heads, N, M] score matrix.
    """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Linear(d_model, d_model)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, key_mask=None, freqs_cis=None):
        batch_dim = query.size(0)
        # [B, N, C] -> [B, heads, N, head_dim]. The channel axis splits head-dim first,
        # so channel c belongs to head c % num_heads -- keep that, or every projection
        # weight lands on a different head than it was trained on.
        query, key, value = [l(x).view(batch_dim, -1, self.dim, self.num_heads).permute(0, 3, 1, 2)
                             for l, x in zip(self.proj, (query, key, value))]
        if freqs_cis is not None:
            query, key = apply_rope(query, freqs_cis), apply_rope(key, freqs_cis)
        # a True entry takes part in the attention, so a query never sees the padding of the
        # batch, and therefore never sees another pair
        attn_mask = key_mask[:, None, None, :] if key_mask is not None else None
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        x = x.permute(0, 2, 3, 1).reshape(batch_dim, -1, self.dim * self.num_heads)
        return self.merge(x)


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, mask=None, source_mask=None, freqs_cis=None):
        message = self.attn(x, source, source, source_mask, freqs_cis)
        return apply_mlp(self.mlp, torch.cat([x, message], dim=-1), mask)


class SelfAttention(nn.Module):
    """Masked multi-head self-attention over one cloud, positions encoded with RoPE-3D."""

    def __init__(self, feature_dim: int, num_heads: int, rope: bool = True,
                 rope_theta: float = 100.0):
        super().__init__()
        self.attn = AttentionalPropagation(feature_dim, num_heads)
        self.rope = RoPE3D(feature_dim // num_heads, theta=rope_theta) if rope else None

    def forward(self, coords, features, mask=None):
        """
        Input:
            coords:     [B, N, 3] voxel coordinates of the bottleneck
            feats:      [B, N, C]
            mask:       [B, N] or None
        Output:
            feats:      [B, N, C]
        """
        freqs_cis = self.rope(coords) if self.rope is not None else None
        return features + self.attn(features, features, mask, mask, freqs_cis)


class OverlapAttention(nn.Module):
    """
        Alternate between self-attention and cross-attention
        Input:
            coords:     [B, N, 3] voxel coordinates, consumed by the rotary embedding
            feats:      [B, N, C]
            masks:      [B, N] or None, marks the real points of a padded batch
        Output:
            feats:      [B, N, C]
        """
    def __init__(self, num_head: int, feature_dim: int, layer_names: list,
                 rope: bool = True, rope_theta: float = 100.0):
        super().__init__()
        self.layers=[]
        for atten_type in layer_names:
            if atten_type == 'cross':
                self.layers.append(AttentionalPropagation(feature_dim,num_head))
            elif atten_type == 'self':
                self.layers.append(SelfAttention(feature_dim, num_head, rope, rope_theta))
            else:
                raise ValueError(f'unknown attention layer {atten_type}')
        self.layers = nn.ModuleList(self.layers)
        self.names = layer_names

    def forward(self, coords0, coords1, desc0, desc1, mask0=None, mask1=None):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                # desc0 = desc0 + checkpoint.checkpoint(layer, desc0, desc1)
                # desc1 = desc1 + checkpoint.checkpoint(layer, desc1, desc0)
                # note that desc1 is updated with the already updated desc0, as upstream
                desc0 = desc0 + layer(desc0, desc1, mask0, mask1)
                desc1 = desc1 + layer(desc1, desc0, mask1, mask0)
            elif name == 'self':
                desc0 = layer(coords0, desc0, mask0)
                desc1 = layer(coords1, desc1, mask1)
        return desc0, desc1
