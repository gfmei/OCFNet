"""
OCFNet -- Overlap-guided Coarse-to-Fine correspondence prediction (Mei et al., ICME 2022).

Pipeline (Fig. 1 of the paper), section by section:

  encoder            a sparse UNet encoder; its 1/8 voxels are the super-points P', Q' with
                     features F_p', F_q'
  overlap attention  geometry-aware positional encoding (Eq. 1) added to the super-point
                     features, then the self/cross attention layers of models/attention.py, then
                     the overlap scores mu_p', mu_q' of the g_alpha / g_beta heads
  coarse matching    optimal transport whose *marginals are the overlap scores*, i.e.
                     min <C', G'> s.t. G'1 = mu_p', G'^T 1 = mu_q' (Eq. 2), solved by a
                     log-domain Sinkhorn. Matches are the mutual maxima above tau_t (Eq. 3)
  decoder            sparse decoder fed with [F_p', mu_p'], giving per-point features F_p
                     and overlap scores mu_p
  point matching     every point is assigned to its nearest super-point, forming patches; a
                     patch keeps its K highest scoring points; each coarse match then solves
                     a small OT problem over its two patches (Eq. 5) and the correspondences
                     are read off row-wise (Eq. 6)

`OCFLoss` implements L = L_C + L_F + L_CO + L_FO of section 2.4. Its ground truth comes from
the correspondence list the data loader already produces (pairs closer than `overlap_radius`
under the ground-truth transform), so the ratios r(p'_i, q'_j) and r(p'_i) of Eq. (7) are
counted from that list instead of a fresh neighbour search.
"""
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from models.common import (get_norm, sparse_relu, sparse_cat, to_dense_batch,
                           from_dense_batch, masked_instance_norm)
from lib.utils import square_distance
from models.attention import AttentionalPropagation, OverlapAttention, RoPE3D
from models.residual_block import get_block


# ---------------------------------------------------------------------------------------
# optimal transport
# ---------------------------------------------------------------------------------------

def log_sinkhorn(scores, log_mu, log_nu, iters):
    """Sinkhorn-Knopp in log space (SuperGlue / CoFiNet formulation).

    scores: [B, N, M] log kernel, log_mu: [B, N], log_nu: [B, M] log marginals.
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def overlap_optimal_transport(cost, mu, nu, bin_cost, iters=20, epsilon=0.05, eps=1e-9,
                              dustbin=True):
    """Eq. (2) / Eq. (5): transport plan whose marginals are the overlap scores.

    cost:     [B, N, M] transport cost, here the distance between normalised features
    mu, nu:   [B, N], [B, M] overlap scores in [0, 1]. A padded entry passes 0 and then
              carries no mass, which is what keeps the pairs of a batch apart
    bin_cost: scalar parameter of the slack row/column, which absorbs the mass the two
              marginals do not have in common (they rarely sum to the same total)

    Returns the log transport plan, [B, N+1, M+1], slack included.
    """
    B, N, M = cost.shape
    bins = bin_cost.expand(B, 1, 1)
    cost = torch.cat([torch.cat([cost, bins.expand(B, N, 1)], dim=-1),
                      torch.cat([bins.expand(B, 1, M), torch.zeros_like(bins)], dim=-1)], dim=1)

    if dustbin:
        # the slack of one side takes the whole mass of the other, so both totals agree
        mu_total, nu_total = mu.sum(-1, keepdim=True), nu.sum(-1, keepdim=True)
        mu = torch.cat([mu, nu_total], dim=-1)
        nu = torch.cat([nu, mu_total], dim=-1)
        total = mu.sum(-1, keepdim=True).clamp(min=eps)
        log_mu = (mu / total).clamp(min=eps).log()
        log_nu = (nu / total).clamp(min=eps).log()
    else:
        # With overlap marginals the score already says how much a super-point has to match,
        # so a slack row would absorb that same mass a second time -- and it is not a small
        # effect: with the slack taking the other side's full total it holds half of all the
        # mass. Each side is normalised on its own instead, and the slack row and column are
        # kept only so the shape downstream is unchanged, at a mass the plan never uses.
        total = mu.sum(-1, keepdim=True).clamp(min=eps)
        mu = torch.cat([mu / total, torch.full_like(mu[..., :1], eps)], dim=-1)
        total = nu.sum(-1, keepdim=True).clamp(min=eps)
        nu = torch.cat([nu / total, torch.full_like(nu[..., :1], eps)], dim=-1)
        log_mu, log_nu = mu.clamp(min=eps).log(), nu.clamp(min=eps).log()
        total = torch.ones_like(total)

    log_plan = log_sinkhorn(cost if epsilon is None else -cost / epsilon, log_mu, log_nu, iters)
    return log_plan + total.log().unsqueeze(-1)  # undo the normalisation


def overlap_bias(cost, src_score, tgt_score, weight, epsilon=None, eps=1e-6):
    """Add lambda*(log o_i + log o_j) to the log-potential of a transport problem.

    The marginals cannot steer *selection*: Sinkhorn factorises as P_ij = u_i K_ij v_j, and a
    row-wise arg-max over j drops u_i entirely, so a source-side marginal only rescales a row
    and never changes which target that row picks. Putting the score in the cost instead puts
    it inside K_ij, where the target term varies along the row and does re-rank. The source
    term is still constant along a row -- o_i says nothing about which target is right -- but
    it now reaches the mutual check, which arg-maxes down the columns.

    `cost` is a log-potential when epsilon is None (the 'inner' route) and a distance
    otherwise, so the bias is added with the sign that makes a high score cheaper in both.
    """
    bias = (weight * (src_score.clamp(min=eps).log()[..., :, None]
                      + tgt_score.clamp(min=eps).log()[..., None, :]))
    return cost + bias if epsilon is None else cost - epsilon * bias


def feature_similarity(feats0, feats1, temperature=None):
    """A log-potential for Sinkhorn, CoFiNet Eq. (1) style.

    With `temperature` the features are L2-normalised first and the cosine similarity is
    scaled by it, which keeps the score bounded while leaving the sharpness learnable.
    Both fixed alternatives were measured at epoch 20 and both are useless: the distance
    between normalised features over a fixed epsilon gave the plan row entropy 4.959
    against 5.019 for a uniform row -- no information at all -- while the raw inner product
    over sqrt(d) spans 1e5 nats at these feature norms (127) and collapses it to 0.008.

    The alternative -- a distance between L2-normalised features, divided by a fixed
    epsilon -- caps how far apart the plan's entries can ever be. Measured at epoch 20 that
    cost had std 0.0367, so after dividing by epsilon = 0.05 the rows of the plan reached
    entropy 4.959 against 5.019 for a uniform row: the plan carried no information and the
    coarse inlier ratio sat at chance. Leaving the features unnormalised lets their
    magnitude act as a learnable temperature, so the model can sharpen its own plan.
    """
    if temperature is None:
        return torch.einsum('bnc,bmc->bnm', feats0, feats1) / feats0.shape[-1] ** 0.5
    return torch.einsum('bnc,bmc->bnm', F.normalize(feats0, p=2, dim=-1),
                        F.normalize(feats1, p=2, dim=-1)) * temperature


def normalised_feature_cost(feats0, feats1):
    """C_ij = || f_i/||f_i|| - f_j/||f_j|| ||_2, the transport cost of the paper."""
    return torch.cdist(F.normalize(feats0, p=2, dim=-1), F.normalize(feats1, p=2, dim=-1))


def mutual_top_matches(plan, threshold, src_mask=None, tgt_mask=None):
    """Eq. (3): mutual maxima of the transport plan above `threshold`.

    plan: [B, N, M]. Returns [P, 3] rows (batch, i, j) for the whole batch at once.
    """
    if src_mask is not None:
        plan = plan.masked_fill(~src_mask[:, :, None], -1.0).masked_fill(~tgt_mask[:, None, :], -1.0)
    keep = ((plan >= plan.max(dim=2, keepdim=True).values)
            & (plan >= plan.max(dim=1, keepdim=True).values)
            & (plan > threshold))
    return torch.nonzero(keep, as_tuple=False)


def top_k_matches(plan, k, src_mask=None, tgt_mask=None):
    """The k most likely super-point pairs of every batch element, [P, 3] rows (batch, i, j).

    A fallback for Eq. (3): a mutual maximum above tau_t is the right criterion once the
    transport plan is peaked, but an undertrained plan is flat and the threshold can leave a
    single match for the whole pair, starving the point level.
    """
    B, N, M = plan.shape
    if src_mask is not None:
        plan = (plan.masked_fill(~src_mask[:, :, None], -float('inf'))
                    .masked_fill(~tgt_mask[:, None, :], -float('inf')))
    scores, flat = plan.flatten(1).topk(min(k, N * M), dim=-1)
    batch = torch.arange(B, device=plan.device)[:, None].expand_as(flat)
    keep = torch.isfinite(scores)                       # samples smaller than k pad with -inf
    return torch.stack([batch[keep], flat[keep] // M, flat[keep] % M], dim=1)


def ground_truth_coarse_pairs(correspondence, src_patch_id, tgt_patch_id):
    """Super-point pairs whose patches share at least one ground-truth correspondence.

    Training refines these instead of the predicted matches (as CoFiNet does), so the point
    level always sees patches that really overlap, even while the coarse matcher is random.
    Patch ids are global, so the correspondences of the whole batch can be mapped at once.
    """
    if correspondence.numel() == 0:
        return correspondence.new_zeros(0, 2)
    pairs = torch.stack([src_patch_id[correspondence[:, 0]],
                         tgt_patch_id[correspondence[:, 1]]], dim=1)
    return torch.unique(pairs, dim=0)


# ---------------------------------------------------------------------------------------
# geometry aware overlap attention (section 2.2)
# ---------------------------------------------------------------------------------------

class GeometricPositionalEncoding(nn.Module):
    """f^e_i = phi(||p_i - p_c||) + max_{x in N_i} varphi(angle(p_i - p_c, p_x - p_c)).

    p_c is the centroid of the cloud and N_i the k nearest neighbours of p_i, so the code
    describes both how far a super-point sits from the centre and how its neighbourhood is
    laid out around that direction. phi and varphi are a linear layer plus a ReLU.
    """

    def __init__(self, feature_dim, k=5):
        super().__init__()
        self.k = k
        self.radial = nn.Sequential(nn.Linear(1, feature_dim), nn.ReLU())
        self.angular = nn.Sequential(nn.Linear(1, feature_dim), nn.ReLU())

    def forward(self, coords, mask=None):
        """coords: [B, N, 3] -> [B, N, C]."""
        xyz = coords                                                    # [B, N, 3]
        valid = mask if mask is not None else torch.ones(
            xyz.shape[:2], dtype=torch.bool, device=xyz.device)

        counts = valid.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        centre = (xyz * valid[..., None]).sum(dim=1, keepdim=True) / counts
        offset = xyz - centre                                           # p_i - p_c
        radius = offset.norm(dim=-1, keepdim=True)                      # [B, N, 1]

        dist = torch.cdist(xyz, xyz, compute_mode='donot_use_mm_for_euclid_dist')
        dist = dist.masked_fill(~valid[:, None, :], float('inf'))
        k = max(1, min(self.k, int(valid.sum(-1).min().item()) - 1))
        idx = dist.topk(k=k + 1, dim=-1, largest=False, sorted=True).indices[..., 1:]

        neighbour = torch.gather(offset.unsqueeze(1).expand(-1, xyz.shape[1], -1, -1), 2,
                                 idx[..., None].expand(-1, -1, -1, 3))  # [B, N, k, 3]
        cos = F.cosine_similarity(offset[:, :, None, :], neighbour, dim=-1).clamp(-1, 1)
        angle = torch.acos(cos).unsqueeze(-1)                           # [B, N, k, 1]

        encoding = self.radial(radius) + self.angular(angle).max(dim=2).values
        return encoding                                                 # [B, N, C]


class OverlapScoreHead(nn.Module):
    """Overlap score of section 2.2: mu_i = g_beta([f_i, w_i^T g_alpha(F_other)]).

    g_alpha scores every point of the other cloud, the attention weights w carry those
    scores over, and g_beta turns (own feature, borrowed score) into a probability. Both are
    a linear layer followed by an instance norm and a sigmoid, as in the paper.
    """

    def __init__(self, feature_dim, instance_norm=True, prior=0.08, rich=False):
        super().__init__()
        # A 1x1 convolution over points is a linear layer applied per point; saying so
        # directly keeps the whole head in the natural [B, N, C] layout, which is also what
        # the transport cost wants, so nothing has to be transposed on the way in or out.
        self.alpha = nn.Linear(feature_dim, 1)
        # Whether a super-point lies in the overlap is a question about how well it matches
        # the other cloud, and the paper's head only sees one borrowed scalar. `rich` adds
        # the evidence that answers it directly: the best similarity to the other cloud, the
        # mean of the ten best, and the entropy of the attention (how ambiguous the match is).
        self.rich = rich
        extra = 4 if rich else 1
        self.beta = (nn.Sequential(nn.Linear(feature_dim + extra, feature_dim), nn.ReLU(),
                                   nn.Linear(feature_dim, 1))
                     if rich else nn.Linear(feature_dim + extra, 1))
        self.instance_norm = instance_norm

        # The paper normalises before the sigmoid. That pins the mean and, worse, destroys
        # the absolute level: an instance norm can only express which super-points are more
        # visible than others *within one cloud*, while "is this patch visible at all" is an
        # absolute question. Measured after 149 epochs with the norm on, the learned affine
        # was scale 0.861 and bias +1.300, mapping a +/-3 sigma input to [0.22, 0.98] and
        # never predicting a confident zero -- against a target that is 53% exactly zero.
        # A near-constant score is a uniform marginal, which is why Eq. (5) changed nothing.
        # With `overlap_head_norm: False` the sigmoid sees the raw projection and can span
        # the range; the affine then only sets the starting point.
        bias = float(np.log(prior / (1 - prior)))
        self.alpha_affine = nn.Parameter(torch.tensor([1.0, bias]))
        self.beta_affine = nn.Parameter(torch.tensor([1.0, bias]))

    def _norm_sigmoid(self, x, mask, affine):
        """x: [B, N, 1], normalised over the points of each sample."""
        if self.instance_norm:
            if mask is None:
                mask = x.new_ones(x.shape[:-1], dtype=torch.bool)
            x = masked_instance_norm(x, mask)
        return torch.sigmoid(x * affine[0] + affine[1])

    def forward(self, feats, other_feats, mask=None, other_mask=None):
        """feats: [B, N, C], other_feats: [B, M, C] -> scores [B, N] in [0, 1]."""
        other_scores = self._norm_sigmoid(self.alpha(other_feats), other_mask,
                                          self.alpha_affine)                     # [B, M, 1]

        weights = torch.einsum('bnc,bmc->bnm', feats, other_feats)
        if other_mask is not None:
            weights = weights.masked_fill(~other_mask[:, None, :], -1e9)
        weights = torch.softmax(weights, dim=-1)

        borrowed = torch.einsum('bnm,bmc->bnc', weights, other_scores)           # [B, N, 1]
        evidence = [feats, borrowed]
        if self.rich:
            similarity = F.normalize(feats, dim=-1) @ F.normalize(other_feats, dim=-1).transpose(1, 2)
            if other_mask is not None:
                similarity = similarity.masked_fill(~other_mask[:, None, :], -1.0)
            best = similarity.max(dim=-1).values
            top = similarity.topk(min(10, similarity.shape[-1]), dim=-1).values.mean(-1)
            entropy = -(weights.clamp(min=1e-9) * weights.clamp(min=1e-9).log()).sum(-1)
            evidence += [best[..., None], top[..., None], entropy[..., None]]
        scores = self._norm_sigmoid(self.beta(torch.cat(evidence, dim=-1)), mask,
                                    self.beta_affine)
        return scores.squeeze(-1)


# voxel offsets inside a patch are small and signed; the rotary table is indexed from 0, so
# they are shifted into the middle of a span this wide (a rotation only sees differences)
PATCH_ROPE_SPAN = 64


class PatchRefiner(nn.Module):
    """Make the fine stage aware of *which* patch it is matching against.

    The marginal of Eq. (5) is the global overlap score mu_p -- whether a point has a partner
    anywhere in the other cloud. But whether it has one inside a given patch depends on the
    pair: a point of patch A may be matchable against B and not against C. This block lets
    the two patches exchange information (cross-attention over their points) and predicts a
    per-point score conditioned on that pair, which then serves as the transport marginal.
    The refined features also form the cost matrix, so both are pair-specific.
    """

    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.cross = AttentionalPropagation(feature_dim, num_heads)
        self.score = nn.Linear(feature_dim, 1)
        # CoFiNet's l_final_proj, the fine-level counterpart of final_proj
        self.final_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        # Points inside a patch lie a few centimetres apart in a 20 cm neighbourhood, and the
        # transport cost is otherwise pure descriptor distance -- the matcher would be blind
        # to where in the patch a point sits. Their offsets from the super-point enter as a
        # rotation of the queries and keys, so a score depends on the *relative* offset of
        # the two points, the same way RoPE works in the coarse attention.
        self.rope = RoPE3D(feature_dim // num_heads, theta=10.0,
                           max_grid_size=(PATCH_ROPE_SPAN,) * 3)

    def forward(self, src_feats, tgt_feats, src_mask, tgt_mask,
                src_offsets=None, tgt_offsets=None):
        """[P, K, C] features, [P, K] masks, [P, K, 3] voxel offsets from the super-point."""
        src_cis = self.rope(src_offsets + PATCH_ROPE_SPAN // 2) if src_offsets is not None else None
        tgt_cis = self.rope(tgt_offsets + PATCH_ROPE_SPAN // 2) if tgt_offsets is not None else None
        src = src_feats + self.cross(src_feats, tgt_feats, src_mask, tgt_mask, src_cis)
        tgt = tgt_feats + self.cross(tgt_feats, src_feats, tgt_mask, src_mask, tgt_cis)
        src, tgt = self.final_proj(src), self.final_proj(tgt)
        return (src, tgt,
                torch.sigmoid(self.score(src)).squeeze(-1),
                torch.sigmoid(self.score(tgt)).squeeze(-1))


# ---------------------------------------------------------------------------------------
# patches (section 2.3)
# ---------------------------------------------------------------------------------------

def assign_to_super_points(points, point_batch, super_points, super_batch, budget=4_000_000):
    """Nearest super-point of every point (Eq. 4), for the whole batch in one go.

    Pairs are kept apart by masking the distances of other samples, not by appending a large
    batch coordinate: torch.cdist expands ||a-b||^2 as ||a||^2 - 2ab + ||b||^2, so a
    coordinate of 1e4 makes ||a||^2 ~ 1e8 and the metre-scale term vanishes in float32. That
    version picked the correct super-point for 0.5% of points, which left half the cells
    empty and one holding 40% of the cloud.

    `budget` caps how many entries of the distance matrix exist at once, so the transient
    stays put as the batch or the super-point count grows; it is a memory guard, not a
    per-sample loop.
    """
    chunk = max(1, budget // max(1, super_points.shape[0]))
    out = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    for start in range(0, points.shape[0], chunk):
        stop = start + chunk
        # The fast matmul expansion, in float32: 38 ms per cloud against 982 ms for the
        # exact kernel, agreeing with it on 99.79% of points -- the rest are boundary ties
        # between two near-equidistant super-points, where either answer is defensible.
        # This is only well conditioned because the 1e4 batch coordinate is gone; with it,
        # ||a||^2 reached 1e8 and the correct super-point came out 0.5% of the time.
        distance = torch.cdist(points[start:stop], super_points)
        distance = distance.masked_fill(
            point_batch[start:stop, None] != super_batch[None, :], float('inf'))
        out[start:stop] = distance.argmin(dim=1)
    return out


def pack_indices(indices, span=8192):
    """[N, 4] (batch, x, y, z) voxel indices -> one int64 key each, for set membership."""
    a = indices.long()
    return ((a[:, 0] * span + a[:, 1]) * span + a[:, 2]) * span + a[:, 3]


def super_point_positions(fine_indices, fine_xyz, super_indices, stride, sample=True):
    """Real coordinate of each super-point: the coordinate of one of its own points.

    A super-point sits at the *corner* of its coarse voxel, which at stride 8 is a 20 cm
    cell, so the corner misses the cell's own point mass by a median 8.8 cm -- more than the
    3.75 cm that defines a correspondence. Every Voronoi cell and every patch built around
    such a point is displaced.

    The replacement is one of the points in the cell, so the super-point always lies on the
    surface; a barycentre would float off it wherever a cell straddles a corner or a thin
    structure. Which point is drawn uniformly at random while training, a mild augmentation
    of the patch layout, and is the first in index order at inference so that evaluation
    stays reproducible. A fine voxel belongs to the coarse voxel `index // stride`, exact
    for the k=2 stride-2 encoder, so the members are known without a search.
    """
    key = torch.cat([fine_indices[:, :1],
                     torch.div(fine_indices[:, 1:], stride, rounding_mode='floor')], dim=1)
    fine_key, super_key = pack_indices(key), pack_indices(super_indices)
    order = torch.argsort(super_key)
    rank = torch.searchsorted(super_key[order], fine_key).clamp(max=super_key.shape[0] - 1)
    row = order[rank]
    hit = super_key[row] == fine_key
    member, row = torch.nonzero(hit, as_tuple=True)[0], row[hit]

    if sample:                       # uniformly random member: shuffle, then a stable regroup
        shuffle = torch.randperm(row.shape[0], device=row.device)
        member, row = member[shuffle], row[shuffle]
    inside = torch.argsort(row, stable=True)
    member, row = member[inside], row[inside]
    counts = torch.bincount(row, minlength=super_indices.shape[0])
    starts = torch.cumsum(counts, 0) - counts
    if not bool((counts > 0).all()):
        raise RuntimeError('a super-point has no points in its cell; the encoder is no '
                           'longer a chain of k=2 stride-2 convolutions, so index // stride '
                           'does not give cell membership')
    return fine_xyz[member[starts]]


def true_visibility(xyz, point_batch, centre, super_batch, seen, index, batch_size, n_padded):
    """Fraction of each super-point's own points that have a ground-truth correspondence.

    This is the target the overlap head is trained against (Eq. 7), computed directly so it
    can be fed to the transport marginals in place of the prediction.
    """
    pid = assign_to_super_points(xyz, point_batch, centre, super_batch)
    n_super = centre.shape[0]
    size = torch.bincount(pid, minlength=n_super).clamp(min=1).float()
    vis = torch.zeros(n_super, device=xyz.device).index_add_(0, pid, seen) / size
    order, sorted_ids, pos = index
    dense = torch.zeros(batch_size, n_padded, device=xyz.device)
    dense[sorted_ids, pos] = vis[order]
    return dense


def true_pair_score(take, other_patch_of_pair, correspondence, other_patch_id, is_source):
    """o_ik: 1 where the point at `take[p, k]` has a correspondence inside the paired patch.

    This is the pair-conditional quantity the global overlap score cannot express -- the same
    point is 1 against the patch holding its partner and 0 against every other. It needs only
    the correspondence list and the point-to-super-point assignment, not the transform.
    """
    src_col, tgt_col = (0, 1) if is_source else (1, 0)
    keys = (correspondence[:, src_col].long() * (int(other_patch_id.max()) + 1)
            + other_patch_id[correspondence[:, tgt_col]].long())
    keys = torch.unique(keys)
    want = (take.long() * (int(other_patch_id.max()) + 1)
            + other_patch_of_pair[:, None].long())
    order = torch.argsort(keys)
    pos = torch.searchsorted(keys[order], want.reshape(-1)).clamp(max=keys.numel() - 1)
    hit = keys[order][pos] == want.reshape(-1)
    return hit.reshape(take.shape).float()


def voronoi_patches(patch_id, n_patches, patch_size, scores=None):
    """CoFiNet's grouping: the Voronoi cell of a super-point, truncated by point index.

    Mirrors their `grouping_wrapper` -- each super-point takes the points assigned to it by
    `assign_to_super_points`, and pads the rest out with a validity mask, so patches are
    disjoint: a point belongs to exactly one of them.

    With `scores`, an overflowing cell keeps its highest-scoring members rather than the
    first by point index, and the straight-through gate carries d/d(score) back, so the
    overlap head still hears from the fine stage. Without them the truncation is CoFiNet's,
    by point index, and the gate is constant.
    """
    device = patch_id.device
    if scores is None:
        order = torch.argsort(patch_id, stable=True)
    else:
        by_score = torch.argsort(scores, descending=True)
        order = by_score[torch.argsort(patch_id[by_score], stable=True)]
    sorted_id = patch_id[order]
    counts = torch.bincount(sorted_id, minlength=n_patches)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(patch_id.shape[0], device=device) - starts[sorted_id]
    keep = rank < patch_size
    index = torch.zeros(n_patches, patch_size, dtype=torch.long, device=device)
    valid = torch.zeros(n_patches, patch_size, dtype=torch.bool, device=device)
    index[sorted_id[keep], rank[keep]] = order[keep]
    valid[sorted_id[keep], rank[keep]] = True
    if scores is None:
        return index, valid, torch.ones(n_patches, patch_size, device=device)
    chosen = scores[index]
    return index, valid, chosen / chosen.detach().clamp(min=1e-6)


def dense_index_maps(scatter_index, batch_size, n_padded):
    """Translate between sparse rows and their slot in a padded [B, n_padded] tensor.

    Returns (position of every row, row living at every slot).
    """
    order, sorted_ids, pos = scatter_index
    row_position = torch.empty_like(order)
    row_position[order] = pos
    slot_row = torch.zeros(batch_size, n_padded, dtype=torch.long, device=order.device)
    slot_row[sorted_ids, pos] = order
    return row_position, slot_row


# ---------------------------------------------------------------------------------------
# sparse backbone
# ---------------------------------------------------------------------------------------

class SparseEncoder(nn.Module):
    """Stem plus three stride-2 stages: raw voxels -> super-points on the 1/8 grid."""

    def __init__(self, in_channels, channels, norm_type, block_norm_type, bn_momentum,
                 stem_kernel_size, D=3):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(in_channels, channels[0], stem_kernel_size, bias=False,
                                       indice_key='subm_s1_stem')
        self.norm1 = get_norm(norm_type, channels[0], bn_momentum=bn_momentum, D=D)
        self.block1 = get_block(block_norm_type, channels[0], channels[0],
                                bn_momentum=bn_momentum, D=D, indice_key='subm_s1')

        # kernel 2 stride 2 maps a voxel to exactly floor(c / 2), so three stages give the
        # 1/8 grid of the paper -- a few hundred super-points per cloud. A kernel-3 stride-2
        # convolution (what models/resunet.py uses, to stay faithful to Predator) also emits
        # the output voxels whose receptive field merely touches an active input, which
        # compounds to about twice as many super-points over three stages.
        self.downs, self.norms, self.blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for level, (cin, cout) in enumerate(zip(channels[:-1], channels[1:]), start=1):
            self.downs.append(spconv.SparseConv3d(cin, cout, 2, stride=2, bias=False,
                                                  indice_key=f'down_s{2 ** level}'))
            self.norms.append(get_norm(norm_type, cout, bn_momentum=bn_momentum, D=D))
            self.blocks.append(get_block(block_norm_type, cout, cout, bn_momentum=bn_momentum,
                                         D=D, indice_key=f'subm_s{2 ** level}'))

    def forward(self, x):
        skips = []
        out = self.block1(self.norm1(self.conv1(x)))
        for down, norm, block in zip(self.downs, self.norms, self.blocks):
            skips.append(out)
            out = block(norm(down(sparse_relu(out))))
        return sparse_relu(out), skips


class SparseDecoder(nn.Module):
    """Three inverse-convolution stages back to the input voxels, then the output head."""

    def __init__(self, in_channels, channels, tr_channels, out_channels, norm_type,
                 block_norm_type, bn_momentum, D=3):
        super().__init__()
        self.ups, self.norms, self.blocks = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cin = in_channels
        for level in range(len(tr_channels), 0, -1):
            # keyed to the encoder convolution whose coordinates it has to restore
            self.ups.append(spconv.SparseInverseConv3d(cin, tr_channels[level - 1], 2, bias=False,
                                                       indice_key=f'down_s{2 ** level}'))
            self.norms.append(get_norm(norm_type, tr_channels[level - 1],
                                       bn_momentum=bn_momentum, D=D))
            self.blocks.append(get_block(block_norm_type, tr_channels[level - 1],
                                         tr_channels[level - 1], bn_momentum=bn_momentum, D=D,
                                         indice_key=f'subm_s{2 ** (level - 1)}'))
            cin = tr_channels[level - 1] + channels[level - 1]

        self.conv1_tr = spconv.SubMConv3d(cin, tr_channels[0], 1, bias=False,
                                          indice_key='subm_s1_1x1')
        self.final = spconv.SubMConv3d(tr_channels[0], out_channels, 1, bias=True,
                                       indice_key='subm_s1_final')

    def forward(self, x, skips):
        for up, norm, block, skip in zip(self.ups, self.norms, self.blocks, reversed(skips)):
            x = sparse_cat(sparse_relu(block(norm(up(x)))), skip)
        return self.final(sparse_relu(self.conv1_tr(x)))


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------

class OCFNet(nn.Module):
    """Overlap-guided coarse-to-fine correspondence prediction."""

    NORM_TYPE = 'IN'
    BLOCK_NORM_TYPE = 'IN'
    CHANNELS = [32, 64, 128, 256]
    TR_CHANNELS = [64, 64, 64]
    BOTTLENECK_STRIDE = 8

    def __init__(self, config, D=3):
        super().__init__()
        self.voxel_size = config.voxel_size
        feature_dim = config.attention_feats_dim

        self.encoder = SparseEncoder(config.in_feats_dim, self.CHANNELS, self.NORM_TYPE,
                                     self.BLOCK_NORM_TYPE, config.bn_momentum,
                                     config.conv1_kernel_size, D)
        # the decoder is fed with the attended super-point feature plus its overlap score,
        # and emits a descriptor plus a point overlap logit
        self.decoder = SparseDecoder(feature_dim + 1, self.CHANNELS, self.TR_CHANNELS,
                                     config.out_feats_dim + 1, self.NORM_TYPE,
                                     self.BLOCK_NORM_TYPE, config.bn_momentum, D)

        self.bottle = nn.Linear(self.CHANNELS[-1], feature_dim, bias=True)
        # CoFiNet's final_proj: one more projection after the attention, before the
        # similarity is formed. Without it the similarity reads the attention's
        # residual stream directly, whose scale nothing controls (norm 127 measured).
        self.final_proj = nn.Linear(feature_dim, feature_dim, bias=True)
        # two ways to tell the attention where a super-point is: the paper's geometric
        # encoding added to the features, and/or a rotary embedding inside the attention
        self.pos_encoding = (GeometricPositionalEncoding(feature_dim, k=config.get('pos_enc_k', 5))
                             if config.get('geometric_pos_enc', True) else None)
        self.attention = OverlapAttention(config.num_head, feature_dim, config.nets,
                       rope=config.get('rope', False), rope_theta=config.get('rope_theta', 100.0))
        # Running without any position was copied from CoFiNet (ape: False) and does work,
        # but it is not the better setting here: with RoPE-3D on the super-point voxel
        # indices the coarse inlier ratio is about +0.085 at matched epochs, in both the
        # overlap and the no-overlap condition, and the fine inlier ratio +0.089. CoFiNet's
        # choice was made for a KPConv backbone and does not transfer.
        #
        # The paper's own geometric encoding is a third option and is disabled by default for
        # a different reason: it measures each point's radius from its *own* cloud's centroid,
        # and under partial overlap the two centroids are different physical points, so
        # corresponding super-points receive different encodings.
        if self.pos_encoding is None and not config.get('rope', False):
            print('coarse attention runs without any positional encoding '
                  '(CoFiNet ape: False; rope: True measures better)')
        self.patch_refiner = (PatchRefiner(config.out_feats_dim, config.num_head)
                              if config.get('patch_attention', True) else None)
        # The overlap score is this paper's contribution over CoFiNet, so it is optional:
        # with it off the model is a plain coarse-to-fine matcher and any gap to CoFiNet's
        # numbers is attributable. Off, nothing may depend on the score -- an unsupervised
        # head would still steer patch membership through `voronoi_scored`.
        self.use_overlap_head = config.get('use_overlap_head', True)
        # Teacher forcing: during training, feed the true visibility in place of the
        # predicted score. It bounds what any overlap head could contribute -- if the model
        # cannot exploit a perfect score, the mechanism is the limit, not the prediction.
        # At inference the head's own output is used, since the truth is not available.
        self.gt_overlap = config.get('gt_overlap', False)
        if self.use_overlap_head:
            self.overlap_head = OverlapScoreHead(
                feature_dim, instance_norm=config.get('overlap_head_norm', True),
                prior=config.get('overlap_prior', 0.08),
                rich=config.get('overlap_head_rich', False))
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        # cosine similarities among these features have std ~0.02, so the scale that
        # makes a row of the plan informative over a few hundred candidates is of
        # order 50. It is learned from there rather than fixed.
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(50.0))))

        self.sinkhorn_iters = config.get('sinkhorn_iters', 20)
        self.sinkhorn_epsilon = config.get('sinkhorn_epsilon', 0.05)
        self.coarse_threshold = config.get('coarse_threshold', 0.05)
        self.patch_size = config.get('patch_size', 32)
        # Patches are always disjoint. 'voronoi_scored' keeps an overflowing cell's
        # highest-overlap members and stays differentiable through the gate; 'voronoi' is
        # CoFiNet's truncation by point index. Overlapping candidate sets are gone: they
        # existed only to paper over super-points that sat on their lattice corner.
        self.patch_grouping = config.get('patch_grouping', 'voronoi_scored')
        if self.patch_grouping not in ('voronoi_scored', 'voronoi'):
            raise ValueError(f'patch_grouping must be voronoi_scored or voronoi, '
                             f'got {self.patch_grouping!r}: patches must be disjoint')
        self.fine_marginals = config.get('fine_marginals', 'uniform')
        # which overlap score feeds the point-level marginals when the patch refiner runs:
        # 'pair'  -- o_ik from the refiner, "visible in *this* patch" (Eq. 5 as written)
        # 'point' -- the decoder's per-point score, "visible at all". The pair head plateaus
        # near the chance level of the balanced BCE (0.62 against log 2) while the per-point
        # head reaches 0.38, so the pair score can be the weaker signal of the two.
        # lambda of the cost-side overlap bias; 0 disables it and keeps the plain cost
        self.overlap_cost_weight = float(config.get('overlap_cost_weight', 0.0))
        self.fine_overlap_source = config.get('fine_overlap_source', 'pair')
        if self.fine_overlap_source not in ('pair', 'point'):
            raise ValueError(f'fine_overlap_source must be pair or point, '
                             f'got {self.fine_overlap_source!r}')
        # 'uniform' -- every super-point carries one unit of mass, as CoFiNet does.
        # 'overlap' -- Eq. (5), mass proportional to the overlap score. The latter makes a
        # row's total plan mass scale with its score, so selecting the global top-k favours
        # high-scoring super-points over genuinely matching ones while that head is still
        # miscalibrated: at epoch 1 the plan ranked true pairs at the 96th percentile yet
        # the top 128 of 250k contained none of them.
        # 'inner' is CoFiNet's unnormalised inner product / sqrt(d) as a log-potential;
        # 'cosine' the distance between normalised features divided by sinkhorn_epsilon
        # With overlap marginals the score already encodes matchability, so the slack row
        # would absorb it twice; it holds half the total mass. Kept switchable to
        # separate the two effects.
        self.coarse_dustbin = config.get('coarse_dustbin', False)
        self.overlap_sharpen = float(config.get('overlap_sharpen', 1.0))
        self.overlap_threshold = float(config.get('overlap_threshold', 0.0))
        self.readout_mutual = config.get('readout_mutual', True)
        # CoFiNet keeps the *union* of the row and the column arg-max
        # (`logical_or(row_map, col_map)`), which covers points of both clouds; `mutual` is
        # the intersection and covers only points both clouds agree on. 'row' is neither.
        self.readout_select = config.get('readout_select', 'row')
        if self.readout_select not in ('row', 'union'):
            raise ValueError(f"readout_select must be row or union, got {self.readout_select!r}")
        # scale the fine confidence by the patch pair's coarse confidence, as CoFiNet does
        self.readout_scale_by_coarse = config.get('readout_scale_by_coarse', False)
        self.readout_deduplicate = config.get('readout_deduplicate', True)
        self.readout_beat_slack = config.get('readout_beat_slack', False)
        self.fine_dustbin = config.get('fine_dustbin', False)
        self.similarity = config.get('similarity', 'inner')
        if self.similarity not in ('inner', 'scaled_cosine', 'cosine'):
            raise ValueError(
                f'similarity must be inner, scaled_cosine or cosine, got '
                f'{self.similarity!r}. A typo used to fall through to the cosine\n'
                f'branch silently, which is the flat-plan regime.')
        self.coarse_marginals = config.get('coarse_marginals', 'uniform')
        if not self.use_overlap_head:
            # Only the *coarse* marginals come from the global head. The fine marginals come
            # from PatchRefiner's pair-conditional score, which is a different quantity: it
            # answers "does this point have a partner inside *this* patch", so it can be
            # 1 against patch K and 0 against patch L where a global score cannot.
            if self.coarse_marginals == 'overlap':
                raise ValueError('coarse overlap marginals need use_overlap_head: True')
            if self.fine_marginals == 'overlap' and not config.get('patch_attention', True):
                raise ValueError('fine overlap marginals need patch_attention: True, '
                                 'which is what predicts the pair-conditional score')
            # CoFiNet's grouping: the first patch_size members by point index, no score
            self.patch_grouping = 'voronoi'
        self.max_coarse_matches = config.get('max_coarse_matches', 128)
        self.min_coarse_matches = config.get('min_coarse_matches', 128)

    def _super_points(self, stensor, centre):
        """Dense view of a bottleneck: features, coordinates, mask, scatter index.

        Coordinates come in both flavours because the two position encodings need different
        ones: the geometric encoding of Eq. (1) measures distances and angles, so it wants
        metres, while the rotary embedding indexes a table of voxel positions.
        """
        batch_ids = stensor.indices[:, 0].long()
        feats, mask, index = to_dense_batch(stensor.features, batch_ids, stensor.batch_size)
        voxels, _, _ = to_dense_batch(stensor.indices[:, 1:].float(), batch_ids, stensor.batch_size)
        coords, _, _ = to_dense_batch(centre, batch_ids, stensor.batch_size)
        return feats, coords, voxels, mask, index

    def forward(self, stensor_src, stensor_tgt, correspondences=None,
                src_xyz=None, tgt_xyz=None, overlap_correspondences=None):
        """
        src_xyz / tgt_xyz: the real coordinates of the input voxels, in the order they were
        given to `make_sparse_tensor`. Required -- the sparse tensors carry only integer grid
        indices, and snapping points and super-points to their lattice corners displaces
        every patch from the geometry it is meant to describe.

        correspondences: the ground-truth point pairs of the batch, indices into the
        concatenated clouds (as produced by datasets.dataloader.collate_pair_fn). When given
        -- i.e. during training -- the patches to refine are the super-point pairs those
        correspondences fall into, so the point level learns from patches that really overlap
        even while the coarse matcher is still random. Leave it out at inference and the
        predicted matches of Eq. (3) are refined instead.
        """
        ################################
        # 1. encoder: raw voxels -> super-points
        src_bottleneck, src_skips = self.encoder(stensor_src)
        tgt_bottleneck, tgt_skips = self.encoder(stensor_tgt)

        if src_xyz is None or tgt_xyz is None:
            raise ValueError(
                'src_xyz and tgt_xyz are required: the sparse tensors carry only grid '
                'indices, and grouping on lattice corners displaces every patch from the '
                'geometry it describes. Pass the real point coordinates.')
        src_centre = super_point_positions(
            stensor_src.indices, src_xyz, src_bottleneck.indices, self.BOTTLENECK_STRIDE,
            sample=self.training)
        tgt_centre = super_point_positions(
            stensor_tgt.indices, tgt_xyz, tgt_bottleneck.indices, self.BOTTLENECK_STRIDE,
            sample=self.training)

        src_feats, src_coords, src_voxels, src_mask, src_index = self._super_points(src_bottleneck, src_centre)
        tgt_feats, tgt_coords, tgt_voxels, tgt_mask, tgt_index = self._super_points(tgt_bottleneck, tgt_centre)

        ################################
        # 2. geometry aware overlap attention
        src_feats, tgt_feats = self.bottle(src_feats), self.bottle(tgt_feats)
        if self.pos_encoding is not None:
            src_feats = src_feats + self.pos_encoding(src_coords, src_mask)
            tgt_feats = tgt_feats + self.pos_encoding(tgt_coords, tgt_mask)
        # the rotary embedding of the attention reads these as voxel indices
        src_feats, tgt_feats = self.attention(src_voxels, tgt_voxels, src_feats, tgt_feats,
                                              src_mask, tgt_mask)
        src_feats, tgt_feats = self.final_proj(src_feats), self.final_proj(tgt_feats)

        if self.use_overlap_head:
            src_super_overlap = self.overlap_head(src_feats, tgt_feats, src_mask, tgt_mask)
            tgt_super_overlap = self.overlap_head(tgt_feats, src_feats, tgt_mask, src_mask)
            gt_corr = correspondences if correspondences is not None else overlap_correspondences
            if self.gt_overlap and gt_corr is not None:
                # keep the head's own output in the graph so its supervision still trains it,
                # but hand the true score to the marginals and the patch selection
                seen_s = torch.zeros(src_xyz.shape[0], device=src_xyz.device)
                seen_t = torch.zeros(tgt_xyz.shape[0], device=tgt_xyz.device)
                seen_s[gt_corr[:, 0]] = 1.0
                seen_t[gt_corr[:, 1]] = 1.0
                true_s = true_visibility(src_xyz, stensor_src.indices[:, 0].long(), src_centre,
                                         src_bottleneck.indices[:, 0].long(), seen_s, src_index,
                                         src_mask.shape[0], src_mask.shape[1])
                true_t = true_visibility(tgt_xyz, stensor_tgt.indices[:, 0].long(), tgt_centre,
                                         tgt_bottleneck.indices[:, 0].long(), seen_t, tgt_index,
                                         tgt_mask.shape[0], tgt_mask.shape[1])
                src_super_overlap = true_s + 0 * src_super_overlap
                tgt_super_overlap = true_t + 0 * tgt_super_overlap
        else:
            # the decoder keeps its extra input channel so its shape does not change
            src_super_overlap = src_feats.new_zeros(src_mask.shape)
            tgt_super_overlap = tgt_feats.new_zeros(tgt_mask.shape)

        ################################
        # 3. coarse matching: OT over the feature cost
        if self.similarity == 'inner':          # CoFiNet: F_X F_Y^T / sqrt(d), unnormalised
            cost, epsilon = feature_similarity(src_feats, tgt_feats), None
        elif self.similarity == 'scaled_cosine':
            cost, epsilon = feature_similarity(src_feats, tgt_feats,
                                               self.log_temperature.exp()), None
        else:                                    # 'cosine': distance over a fixed epsilon
            cost, epsilon = normalised_feature_cost(src_feats, tgt_feats), self.sinkhorn_epsilon
        if self.coarse_marginals == 'overlap':
            # The marginal wants the *shape* of the true visibility, not just its ordering.
            # Measured at epoch 16 the head ranks well (AUC 0.881) but 52% of super-points
            # should carry no mass and only 10% do, so invisible patches stay in the plan.
            # A monotone power sharpens the score without touching the ranking: x^3 takes the
            # average invisible super-point from 0.219 to 0.177 of a unit share and triples
            # the fraction near zero. It cannot delete a patch the way the truth does, which
            # is the remaining gap to the oracle.
            def shape(score, mask):
                s = score.pow(self.overlap_sharpen)
                if self.overlap_threshold > 0:
                    # The oracle's whole advantage is deletion: it gives 52% of super-points
                    # exactly no mass, while a sigmoid never reaches zero and a power only
                    # suppresses. A straight-through gate makes the forward pass an exact
                    # zero and leaves the backward pass smooth, so the head still trains.
                    keep = (score > self.overlap_threshold).float()
                    # forward: s * keep (exact zeros). backward: d/ds, so a super-point the
                    # gate wrongly suppressed still receives gradient and can come back.
                    s = s + (s * keep - s).detach()
                return s * mask
            src_marginal, tgt_marginal = shape(src_super_overlap, src_mask), shape(tgt_super_overlap, tgt_mask)
        else:
            src_marginal, tgt_marginal = src_mask.float(), tgt_mask.float()
        if self.overlap_cost_weight > 0 and self.use_overlap_head:
            cost = overlap_bias(cost, src_super_overlap * src_mask, tgt_super_overlap * tgt_mask,
                                self.overlap_cost_weight, epsilon)
        coarse_log_plan = overlap_optimal_transport(
            cost, src_marginal, tgt_marginal, self.bin_score,
            iters=self.sinkhorn_iters, epsilon=epsilon,
            dustbin=self.coarse_marginals != 'overlap' or self.coarse_dustbin)

        ################################
        # 4. decoder: super-points (+ their overlap score) -> points
        src_bottleneck = src_bottleneck.replace_feature(from_dense_batch(
            torch.cat([src_feats, src_super_overlap.unsqueeze(-1)], dim=-1), src_index))
        tgt_bottleneck = tgt_bottleneck.replace_feature(from_dense_batch(
            torch.cat([tgt_feats, tgt_super_overlap.unsqueeze(-1)], dim=-1), tgt_index))
        src_out = self.decoder(src_bottleneck, src_skips)
        tgt_out = self.decoder(tgt_bottleneck, tgt_skips)

        output = {
            'src_super_coords': src_coords, 'tgt_super_coords': tgt_coords,
            'src_super_mask': src_mask, 'tgt_super_mask': tgt_mask,
            'src_super_overlap': src_super_overlap, 'tgt_super_overlap': tgt_super_overlap,
            'coarse_log_plan': coarse_log_plan,
            # super-point features, for a loss that works on feature distance directly
            # rather than on the transport plan
            'src_super_feats': src_feats, 'tgt_super_feats': tgt_feats,
            # normalised for the descriptor loss, raw for the patch similarity: CoFiNet
            # builds its local scores from the decoder output directly (src_final_f)
            'src_feats': F.normalize(src_out.features[:, :-1], p=2, dim=1),
            'src_feats_raw': src_out.features[:, :-1],
            'tgt_feats': F.normalize(tgt_out.features[:, :-1], p=2, dim=1),
            'tgt_feats_raw': tgt_out.features[:, :-1],
            'src_overlap': torch.sigmoid(src_out.features[:, -1]),
            'tgt_overlap': torch.sigmoid(tgt_out.features[:, -1]),
        }

        ################################
        # 5. point level: patches around the matched super-points
        output.update(self._patches(src_out, tgt_out, src_bottleneck, tgt_bottleneck,
                                    src_index, tgt_index, src_mask, tgt_mask, output,
                                    correspondences, src_xyz, tgt_xyz, src_centre, tgt_centre))
        return output

    def _patches(self, src_out, tgt_out, src_super, tgt_super, src_index, tgt_index,
                 src_mask, tgt_mask, output, correspondences,
                 src_xyz, tgt_xyz, src_centre, tgt_centre):
        """Group points into patches and solve one OT problem per coarse match, batched."""
        # patch assignment for the whole batch at once (Eq. 4), on real coordinates
        src_patch_id = assign_to_super_points(
            src_xyz, src_out.indices[:, 0].long(), src_centre, src_super.indices[:, 0].long())
        tgt_patch_id = assign_to_super_points(
            tgt_xyz, tgt_out.indices[:, 0].long(), tgt_centre, tgt_super.indices[:, 0].long())
        src_slot, src_slot_row = dense_index_maps(src_index, src_mask.shape[0], src_mask.shape[1])
        tgt_slot, tgt_slot_row = dense_index_maps(tgt_index, tgt_mask.shape[0], tgt_mask.shape[1])

        # which super-point pairs to refine
        if correspondences is not None:
            pairs = ground_truth_coarse_pairs(correspondences, src_patch_id, tgt_patch_id)
        else:
            plan = output['coarse_log_plan'][:, :-1, :-1].exp()
            found = mutual_top_matches(plan, self.coarse_threshold, src_mask, tgt_mask)
            # The fallback has to be decided per pair: summed over a batch the count looks
            # healthy while every individual pair is starved, which leaves the fine stage
            # with a handful of patches and RANSAC with a few hundred correspondences.
            counts = torch.bincount(found[:, 0], minlength=plan.shape[0])
            short = counts < self.min_coarse_matches
            if short.any():
                topk = top_k_matches(plan, self.min_coarse_matches, src_mask, tgt_mask)
                found = torch.cat([found[~short[found[:, 0]]], topk[short[topk[:, 0]]]], dim=0)
            pairs = torch.stack([src_slot_row[found[:, 0], found[:, 1]],
                                 tgt_slot_row[found[:, 0], found[:, 2]]], dim=1)
            # CoFiNet scales every fine score by the confidence of the patch pair it came
            # from (`fine_score * node_corr_conf`), so a match inside a doubtful patch pair
            # ranks below an equally strong match inside a confident one.
            pair_conf = plan[found[:, 0], found[:, 1], found[:, 2]]
        if self.training and pairs.shape[0] > self.max_coarse_matches:
            pick = torch.randperm(pairs.shape[0], device=pairs.device)[:self.max_coarse_matches]
            pairs = pairs[pick]
            if correspondences is None:
                pair_conf = pair_conf[pick]

        if correspondences is not None:
            pair_conf = pairs.new_ones(pairs.shape[0], dtype=torch.float32)
        fine = {'src_patch_id': src_patch_id, 'tgt_patch_id': tgt_patch_id,
                'src_slot': src_slot, 'tgt_slot': tgt_slot, 'coarse_matches': pairs,
                'patch_coarse_conf': pair_conf,
                'src_super_batch': src_super.indices[:, 0].long(),
                'tgt_super_batch': tgt_super.indices[:, 0].long()}
        if pairs.shape[0] == 0:
            device = src_xyz.device
            fine.update({'patch_log_plan': torch.zeros(0, 1, 1, device=device),
                         'patch_src_index': torch.zeros(0, 1, dtype=torch.long, device=device),
                         'patch_tgt_index': torch.zeros(0, 1, dtype=torch.long, device=device),
                         'patch_src_valid': torch.zeros(0, 1, dtype=torch.bool, device=device),
                         'patch_tgt_valid': torch.zeros(0, 1, dtype=torch.bool, device=device),
                         'patch_batch': torch.zeros(0, dtype=torch.long, device=device),
                         'patch_coarse_conf': torch.zeros(0, device=device)})
            return fine

        # candidates around every super-point, then the K best of them by overlap score
        # Patches are disjoint: a point belongs to exactly one of them. An overflowing cell
        # keeps its highest-overlap members, unless this is the CoFiNet-faithful control,
        # which truncates by point index instead.
        scores = None if self.patch_grouping == 'voronoi' else output
        src_patch, src_valid, src_gate = voronoi_patches(
            src_patch_id, src_super.indices.shape[0], self.patch_size,
            None if scores is None else scores['src_overlap'])
        tgt_patch, tgt_valid, tgt_gate = voronoi_patches(
            tgt_patch_id, tgt_super.indices.shape[0], self.patch_size,
            None if scores is None else scores['tgt_overlap'])

        src_take, tgt_take = src_patch[pairs[:, 0]], tgt_patch[pairs[:, 1]]
        src_ok = src_valid[pairs[:, 0]].float() * src_gate[pairs[:, 0]]
        tgt_ok = tgt_valid[pairs[:, 1]].float() * tgt_gate[pairs[:, 1]]

        source = 'src_feats' if self.similarity == 'cosine' else 'src_feats_raw'
        src_patch_feats = output[source][src_take]                          # [P, K, C]
        tgt_patch_feats = output[source.replace('src', 'tgt')][tgt_take]
        # Uniform marginals with a slack column, as CoFiNet does: every selected point has
        # one unit of mass to place, on a target or on the bin. The overlap score no longer
        # doubles as a marginal -- it decides *which* points enter the patch, and reaches the
        # head through the straight-through gate in `src_ok` / `tgt_ok`.
        src_marginal, tgt_marginal = src_ok, tgt_ok
        if self.fine_marginals == 'overlap':
            src_marginal = output['src_overlap'][src_take] * src_ok
            tgt_marginal = output['tgt_overlap'][tgt_take] * tgt_ok
        if self.patch_refiner is not None:
            # offsets of every patch point from its own super-point, in voxels
            src_voxels = src_out.indices[:, 1:].float()
            tgt_voxels = tgt_out.indices[:, 1:].float()
            # the rotary table is indexed by integers, so these stay in voxel units; RoPE
            # only ever sees query-minus-key, so the constant per-patch origin cancels
            src_origin = src_super.indices[:, 1:].float() * self.BOTTLENECK_STRIDE
            tgt_origin = tgt_super.indices[:, 1:].float() * self.BOTTLENECK_STRIDE
            src_offsets = src_voxels[src_take] - src_origin[pairs[:, 0]][:, None, :]
            tgt_offsets = tgt_voxels[tgt_take] - tgt_origin[pairs[:, 1]][:, None, :]
            src_patch_feats, tgt_patch_feats, src_pair, tgt_pair = self.patch_refiner(
                src_patch_feats, tgt_patch_feats,
                src_ok.detach() > 0, tgt_ok.detach() > 0, src_offsets, tgt_offsets)
            # the marginal now says "matchable against *this* patch", not "matchable at all"
            # -- but only when the pair score is the one asked for. `src_marginal` already
            # holds the per-point score from above, so 'point' simply leaves it alone.
            if self.fine_marginals != 'uniform' and self.fine_overlap_source == 'pair':
                src_marginal, tgt_marginal = src_pair * src_ok, tgt_pair * tgt_ok
            if self.gt_overlap and correspondences is not None:
                # teacher forcing at the point level too: the true o_ik replaces the
                # prediction in the marginal, while the prediction stays in the graph so its
                # own supervision keeps training it
                true_src = true_pair_score(src_take, tgt_patch_id[tgt_take[:, 0]],
                                           correspondences, tgt_patch_id, True)
                true_tgt = true_pair_score(tgt_take, src_patch_id[src_take[:, 0]],
                                           correspondences, src_patch_id, False)
                # only the marginal takes the truth. `patch_src_score` keeps the prediction,
                # because pair_overlap_loss is a BCE against the same target: handing it the
                # truth would compare the target with itself and evaluate log(0).
                if self.fine_marginals != 'uniform':
                    src_marginal, tgt_marginal = true_src * src_ok, true_tgt * tgt_ok
            fine['patch_src_score'], fine['patch_tgt_score'] = src_pair, tgt_pair

        # the same routing as the coarse stage: an unnormalised inner product over sqrt(d)
        # fed to Sinkhorn as a log-potential, rather than a distance between normalised
        # features over a fixed epsilon, which cannot spread far enough to peak a plan
        if self.similarity == 'inner':
            patch_cost, patch_eps = feature_similarity(src_patch_feats, tgt_patch_feats), None
        elif self.similarity == 'scaled_cosine':
            patch_cost, patch_eps = feature_similarity(
                src_patch_feats, tgt_patch_feats, self.log_temperature.exp()), None
        else:
            patch_cost = normalised_feature_cost(src_patch_feats, tgt_patch_feats)
            patch_eps = self.sinkhorn_epsilon
        if self.overlap_cost_weight > 0 and self.use_overlap_head:
            # the same scores the marginals would have used, so the two routes are comparable
            patch_cost = overlap_bias(patch_cost, src_marginal, tgt_marginal,
                                      self.overlap_cost_weight, patch_eps)
        fine.update({
            # same argument as the coarse stage: when the marginal is a matchability score
            # the slack would absorb that mass a second time
            'patch_log_plan': overlap_optimal_transport(
                patch_cost, src_marginal, tgt_marginal, self.bin_score,
                iters=self.sinkhorn_iters, epsilon=patch_eps,
                dustbin=self.fine_marginals != 'overlap' or self.fine_dustbin),
            'patch_src_index': src_take, 'patch_tgt_index': tgt_take,
            'patch_src_valid': src_ok.detach() > 0, 'patch_tgt_valid': tgt_ok.detach() > 0,
            'patch_batch': src_super.indices[pairs[:, 0], 0].long()})
        return fine

    @torch.no_grad()
    def point_correspondences(self, output, min_score=0.0, beat_slack=None,
                              mutual=None, deduplicate=None):
        """Eq. (6): inside every patch, match each source point to its best target point.

        Padded slots of a patch hold index 0, so they have to be masked before the argmax --
        their transport mass is ~0 but a flat row would still land on one of them and emit a
        correspondence to a point that was never in the patch.

        A source patch is paired with several target patches, so a point emits one
        correspondence per pair it takes part in and at most one of them can be right. That
        dilution is large: measured on the trained model, the plain arg-max reads out 26.5k
        correspondences at 0.356 inlier ratio, while `mutual` + `deduplicate` reads out 2.0k
        at 0.526. `deduplicate` keeps the most confident proposal per source point, `mutual`
        additionally requires the chosen target to choose the source back, and `beat_slack`
        keeps only points the plan considers matched at all.

        (An early note here said mutual and deduplicate measured *worse*. That was on a
        checkpoint whose coarse matcher was at chance, where agreement between two random
        matchers rejects nearly everything; it does not hold once the model works.)
        """
        beat_slack = self.readout_beat_slack if beat_slack is None else beat_slack
        mutual = self.readout_mutual if mutual is None else mutual
        deduplicate = self.readout_deduplicate if deduplicate is None else deduplicate
        plan = output['patch_log_plan']
        if plan.shape[0] == 0:
            return plan.new_zeros(0, 2, dtype=torch.long), plan.new_zeros(0)
        scores = plan[:, :-1, :-1].exp()                     # drop the slack row/column
        scores = scores.masked_fill(~output['patch_tgt_valid'][:, None, :], -1.0)
        best, col = scores.max(dim=-1)
        keep = output['patch_src_valid'] & (best > min_score)
        if beat_slack:
            keep = keep & (best > plan[:, :-1, -1].exp())
        if mutual:
            # the chosen target must choose this source back
            back = scores.argmax(dim=1)                       # best source per target
            keep = keep & (torch.gather(back, 1, col) ==
                           torch.arange(scores.shape[1], device=scores.device)[None, :])

        src = output['patch_src_index'][keep]
        tgt = torch.gather(output['patch_tgt_index'], 1, col)[keep]
        confidence = best[keep]
        # scale each part by its patch pair's coarse confidence *before* concatenating: a
        # boolean mask flattens pair by pair, so a mask concatenated across the two parts
        # would not line up with a confidence vector that is row-part then column-part.
        if self.readout_scale_by_coarse:
            confidence = confidence * output['patch_coarse_conf'][:, None].expand_as(keep)[keep]

        if self.readout_select == 'union' and not mutual:
            # CoFiNet's `logical_or(row_map, col_map)`: add, for every target point, the
            # source that chose it. The row pass above only covers source points, so a target
            # point nobody's arg-max lands on contributes nothing -- which is what narrows
            # the spatial spread of the correspondence set.
            back_best, back_row = scores.max(dim=1)           # best source per target point
            keep_t = output['patch_tgt_valid'] & (back_best > min_score)
            if beat_slack:
                keep_t = keep_t & (back_best > plan[:, -1, :-1].exp())
            src_t = torch.gather(output['patch_src_index'], 1, back_row)[keep_t]
            tgt_t = output['patch_tgt_index'][keep_t]
            conf_t = back_best[keep_t]
            if self.readout_scale_by_coarse:
                conf_t = conf_t * output['patch_coarse_conf'][:, None].expand_as(keep_t)[keep_t]
            src = torch.cat([src, src_t]); tgt = torch.cat([tgt, tgt_t])
            confidence = torch.cat([confidence, conf_t])

        if deduplicate and src.numel():
            # candidate sets overlap, so a point can be proposed by several patches with
            # different targets; keep the most confident proposal per source point
            order = torch.argsort(confidence, descending=True)
            src, tgt, confidence = src[order], tgt[order], confidence[order]
            _, first = np.unique(src.cpu().numpy(), return_index=True)
            first = torch.from_numpy(first).to(src.device)
            src, tgt, confidence = src[first], tgt[first], confidence[first]
        return torch.stack([src, tgt], dim=1), confidence


# ---------------------------------------------------------------------------------------
# loss (section 2.4)
# ---------------------------------------------------------------------------------------

class OCFLoss(nn.Module):
    """L = L_C + L_F + L_CO + L_FO.

    L_C  coarse matching, cross entropy weighted by the patch overlap ratio r(p'_i, q'_j)
    L_CO binary cross entropy on the super-point overlap scores, target r(p'_i)/max_j r(p'_j)
    L_F  point matching inside the patches of the coarse correspondences
    L_FO binary cross entropy on the point overlap scores
    """

    def __init__(self, config):
        super().__init__()
        # a ranked contrastive loss on the descriptors themselves. The paper supervises the
        # transport plan only (Eq. 9), which puts mass on the true pairs but never pushes a
        # high-scoring impostor down -- and the readout of Eq. (6) is an argmax, so ranking is
        # what decides the inlier ratio. The circle loss shapes the features the cost matrix
        # is built from, so it helps the coarse and the fine transport alike.
        from lib.loss import MetricLoss
        self.circle = MetricLoss(config)
        self.w_descriptor = config.get('w_descriptor_loss', 1.0)
        self.w_patch_infonce = config.get('w_patch_infonce_loss', 1.0)
        self.w_coarse_infonce = config.get('w_coarse_infonce_loss', 1.0)
        self.max_points = config.get('max_points', 256)
        self.w_coarse = config.get('w_coarse_loss', 1.0)
        # overlap-aware circle loss on the super-point features; 0 disables it
        self.w_coarse_circle = config.get('w_coarse_circle_loss', 0.0)
        self.circle_positive_overlap = float(config.get('circle_positive_overlap', 0.1))
        self.circle_pos_margin = float(config.get('circle_pos_margin', 0.1))
        self.circle_neg_margin = float(config.get('circle_neg_margin', 1.4))
        self.circle_pos_optimal = float(config.get('circle_pos_optimal', 0.1))
        self.circle_neg_optimal = float(config.get('circle_neg_optimal', 1.4))
        self.circle_log_scale = float(config.get('circle_log_scale', 24.0))
        self.w_fine = config.get('w_fine_loss', 1.0)
        # trains the reported inlier ratio directly, per row, so the dustbin
        # cannot satisfy it the way it satisfies the matching loss
        self.w_inlier = config.get('w_inlier_loss', 1.0)
        self.w_coarse_inlier = config.get('w_coarse_inlier_loss', 1.0)
        # must agree with the model: a target with slack entries the plan cannot reach
        # produces a loss floor that no amount of training removes
        self.coarse_dustbin = (config.get('coarse_marginals', 'uniform') != 'overlap'
                               or config.get('coarse_dustbin', False))
        self.fine_dustbin = (config.get('fine_marginals', 'uniform') != 'overlap'
                             or config.get('fine_dustbin', False))
        # With the head off its scores are all zeros, so a BCE against a non-zero
        # target diverges. It is multiplied by weight 0, but 0 * inf is nan, so the
        # terms are skipped outright rather than computed and discarded.
        self.use_overlap_head = config.get('use_overlap_head', True)
        self.w_coarse_overlap = config.get('w_coarse_overlap_loss', 1.0)
        self.w_fine_overlap = config.get('w_fine_overlap_loss', 1.0)
        # CoFiNet supervises the fine matcher at `pos_margin` = 0.1 m, the same radius the
        # inlier ratio is measured at. Training at the dataset's 0.0375 m correspondence
        # radius instead makes the positives 2.7x sparser than the metric rewards.
        self.matching_radius = config.get('matching_radius', 0.1)

    def descriptor_loss(self, output, batch, device):
        """Circle loss over the ground-truth correspondences, as in Predator."""
        correspondence = batch['correspondences'].long().to(device)
        if correspondence.shape[0] < 2:
            return output['src_feats'].new_zeros(())
        if correspondence.shape[0] > self.max_points:
            pick = torch.randperm(correspondence.shape[0], device=device)[:self.max_points]
            correspondence = correspondence[pick]

        # the batch is concatenated, so pair every point with its own transform
        rot, trans = batch['rot'].to(device), batch['trans'].to(device)
        lengths = torch.as_tensor([int(n) for n, _ in batch['len_batch']], device=device)
        bounds = torch.cumsum(lengths, 0)
        src_owner = torch.searchsorted(bounds, correspondence[:, 0].contiguous(), right=True)

        src_pcd = batch['pcd_src'].to(device)[correspondence[:, 0]]
        tgt_pcd = batch['pcd_tgt'].to(device)[correspondence[:, 1]]
        src_pcd = torch.einsum('pij,pj->pi', rot[src_owner], src_pcd) + trans[src_owner].squeeze(-1)

        coords_dist = torch.sqrt(square_distance(src_pcd[None], tgt_pcd[None]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(output['src_feats'][correspondence[:, 0]][None],
                                                output['tgt_feats'][correspondence[:, 1]][None],
                                                normalised=True)).squeeze(0)
        return self.circle.get_circle_loss(coords_dist, feats_dist, 1.0)

    @staticmethod
    def inlier_ratio_loss(log_plan, hit, src_valid, tgt_valid, eps=1e-8):
        """Maximise the inlier ratio itself, not the transport mass on true pairs.

        The reported ratio takes each source point's arg-max partner and asks whether it is
        within the radius; arg-max and a threshold have no gradient, but the expectation of
        that indicator under the plan does:

            soft_ir_i = sum_j P(j | i) * hit_ij

        the probability that the match this row would emit is an inlier. Each row is
        normalised on its own, so unlike `_matching_loss` the objective cannot be satisfied
        by the dustbin -- which is how the fine matcher stayed at chance (0.25 against a 0.24
        random floor) while the matching loss kept falling.
        """
        scores = log_plan[:, :-1, :-1].masked_fill(~tgt_valid[:, None, :], -float('inf'))
        row = torch.softmax(scores, dim=-1)
        soft_ir = (row * hit).sum(-1)
        rows = src_valid & (hit.sum(-1) > 0)          # a row with no inlier to find is not a
        if not bool(rows.any()):                      # failure of the matcher
            return log_plan.new_zeros(())
        return -(soft_ir[rows].clamp(min=eps).log()).mean()

    @staticmethod
    def patch_infonce(log_plan, hit, src_valid):
        """Row-wise classification inside a patch: the true partner must be the argmax.

        `L_F` maximises the transport mass on the true pairs; this maximises their *rank*,
        which is what the argmax readout of Eq. (6) actually consumes.
        """
        rows = hit.any(-1) & src_valid                    # source points with a true partner
        if rows.sum() == 0:
            return log_plan.new_zeros(())
        scores = log_plan[:, :-1, :-1][rows]              # [R, K] log scores over the patch
        target = hit[rows].float()
        log_probability = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
        return -(target * log_probability).sum() / target.sum().clamp(min=1)

    @staticmethod
    def _balanced_bce(prediction, target):
        """BCE that weights the two classes by their frequency.

        The targets of Eq. (7) are near zero for most super-points -- a patch is visible in
        the other cloud only inside the overlap -- so a plain cross entropy is dominated by
        easy negatives. Predator balances its overlap loss the same way.
        """
        loss = F.binary_cross_entropy(prediction, target, reduction='none')
        positive = target.mean()
        weight = torch.where(target >= 0.5, 1 - positive, positive)
        return (weight * loss).sum() / weight.sum().clamp(min=1e-6)

    def overlap_circle_loss(self, src_feats, tgt_feats, ratio, src_mask, tgt_mask):
        """Overlap-aware circle loss on the super-point features (GeoTransformer, Eq. 12).

        Our coarse loss is a weighted cross entropy on the transport plan, so it is purely
        competitive: it asks the true patch pair to score *higher* than the others and never
        asks a non-overlapping pair to score *low*. The measured failure is the other way
        round -- on 21% of 3DLoMatch pairs the true pair never reaches the top-128 because
        wrong pairs score too well -- so a term that pushes zero-overlap pairs apart with a
        margin addresses something the plan loss structurally cannot.

        Positives are weighted by sqrt of the overlap ratio, negatives are the pairs with no
        overlap at all; both weights are detached, as in the reference implementation.
        """
        valid = src_mask[:, :, None].bool() & tgt_mask[:, None, :].bool()
        pos = (ratio > self.circle_positive_overlap) & valid
        neg = (ratio == 0) & valid
        # distance between L2-normalised features, in [0, 2]
        d = torch.cdist(F.normalize(src_feats, dim=-1), F.normalize(tgt_feats, dim=-1))

        pos_w = (F.relu(d - self.circle_pos_optimal) * ratio.clamp(min=0).sqrt()).detach()
        neg_w = F.relu(self.circle_neg_optimal - d).detach()
        big = torch.finfo(d.dtype).max
        lse_p = torch.logsumexp(torch.where(
            pos, self.circle_log_scale * (d - self.circle_pos_margin) * pos_w, -big * torch.ones_like(d)), dim=-1)
        lse_n = torch.logsumexp(torch.where(
            neg, self.circle_log_scale * (self.circle_neg_margin - d) * neg_w, -big * torch.ones_like(d)), dim=-1)
        rows = pos.any(-1) & neg.any(-1)          # a row needs both to contribute
        if not bool(rows.any()):
            return d.new_zeros(())
        return (F.softplus(lse_p + lse_n)[rows] / self.circle_log_scale).mean()

    @staticmethod
    def _matching_loss(log_plan, weights):
        """-sum(w * log G) / sum(w), the form used at both levels."""
        total = weights.sum()
        if total <= 0:
            return log_plan.new_zeros(())
        return -(weights * log_plan).sum() / total

    @staticmethod
    def coarse_targets(correspondence, patch_id, other_patch_id, slot, other_slot,
                       batch_of_patch, other_batch_of_patch, shape):
        """Eq. (7), in the doubly normalised form CoFiNet uses (lib/utils.point2node_correspondences).

        A patch pair is weighted by min(row-normalised, column-normalised) share of the
        correspondences, each scaled by how visible that patch is; the slack target of a
        patch is then simply the invisible fraction, 1 - visibility. Normalising by the
        patch size alone (what this did before) lets a row sum past one, because a point can
        have several correspondences, and the slack collapses to zero for patches that are
        barely visible at all.
        """
        device = patch_id.device
        n_patches, n_other = slot.shape[0], other_slot.shape[0]
        sizes = torch.bincount(patch_id, minlength=n_patches).clamp(min=1).float()
        other_sizes = torch.bincount(other_patch_id, minlength=n_other).clamp(min=1).float()

        counts = torch.zeros(shape, device=device)
        seen = torch.zeros(patch_id.shape[0], device=device)
        other_seen = torch.zeros(other_patch_id.shape[0], device=device)
        if correspondence.numel():
            src_patch = patch_id[correspondence[:, 0]]
            tgt_patch = other_patch_id[correspondence[:, 1]]
            counts.index_put_((batch_of_patch[src_patch], slot[src_patch], other_slot[tgt_patch]),
                              torch.ones(src_patch.shape[0], device=device), accumulate=True)
            seen[correspondence[:, 0]] = 1.0
            other_seen[correspondence[:, 1]] = 1.0

        def padded(values, batch_index, position, width):
            out = torch.zeros(shape[0], width, device=device)
            out[batch_index, position] = values
            return out

        visibility = padded(torch.zeros(n_patches, device=device).index_add_(
            0, patch_id, seen) / sizes, batch_of_patch, slot, shape[1])
        other_visibility = padded(torch.zeros(n_other, device=device).index_add_(
            0, other_patch_id, other_seen) / other_sizes, other_batch_of_patch,
            other_slot, shape[2])

        by_row = counts / counts.sum(dim=2, keepdim=True).clamp(min=1e-10) * visibility[..., None]
        by_column = counts / counts.sum(dim=1, keepdim=True).clamp(min=1e-10) * other_visibility[:, None, :]
        ratio = torch.minimum(by_row, by_column)
        return ratio, visibility, other_visibility

    def forward(self, output, batch):
        """`batch` is the dict produced by datasets.dataloader.collate_pair_fn."""
        device = output['src_feats'].device
        correspondence = batch['correspondences'].long().to(device)
        src_mask, tgt_mask = output['src_super_mask'], output['tgt_super_mask']
        B, n_super = src_mask.shape
        m_super = tgt_mask.shape[1]

        src_batch = output['src_super_batch']
        tgt_batch = output['tgt_super_batch']
        ratio, src_visibility, tgt_visibility = self.coarse_targets(
            correspondence, output['src_patch_id'], output['tgt_patch_id'],
            output['src_slot'], output['tgt_slot'], src_batch, tgt_batch, (B, n_super, m_super))

        # ---- coarse level -------------------------------------------------------------
        # A super-point outside the overlap has no counterpart, and the coarse plan has a
        # slack column for exactly that -- but supervising only the inner block leaves it
        # untrained, the same gap that cost the fine stage its ability to reject. The row
        # ratios sum to at most one; whatever is missing belongs in the bin.
        # the invisible fraction of a patch is what belongs in the slack
        # The slack targets say "this patch matches nothing". With a dustbin the plan can
        # place that mass; without one it cannot, and supervising it there costs a loss the
        # model can never reduce (measured 33.3 against the usual 1.5). When the transport
        # has no slack the target has none either: matchability is already in the marginal.
        if self.coarse_dustbin:
            coarse_row_slack = (1 - src_visibility) * src_mask
            coarse_col_slack = (1 - tgt_visibility) * tgt_mask
            coarse_target = torch.cat([
                torch.cat([ratio, coarse_row_slack[..., None]], dim=-1),
                torch.cat([coarse_col_slack[:, None, :],
                           torch.zeros_like(coarse_row_slack[:, :1, None])], dim=-1)], dim=1)
        else:
            coarse_target = torch.zeros_like(output['coarse_log_plan'])
            coarse_target[:, :-1, :-1] = ratio
        coarse_loss = self._matching_loss(output['coarse_log_plan'], coarse_target)
        if self.w_coarse_circle > 0:
            coarse_circle_loss = self.overlap_circle_loss(
                output['src_super_feats'], output['tgt_super_feats'], ratio, src_mask, tgt_mask)
        else:
            coarse_circle_loss = output['coarse_log_plan'].new_zeros(())
        # the coarse matches are read out by ranking (mutual maximum, or top-k), so the true
        # partner has to come first -- mass alone does not guarantee that
        coarse_infonce_loss = self.patch_infonce(output['coarse_log_plan'], ratio > 0, src_mask)
        # and the inlier ratio of the coarse transport itself: the patch pair each row would
        # emit has to be one that really overlaps. The matching loss above is mass-weighted
        # and its dustbin carries most of the target, so it falls to 0.29 while the plan
        # ranks true pairs at chance -- this term is normalised per row and cannot.
        coarse_inlier_loss = self.inlier_ratio_loss(
            output['coarse_log_plan'], (ratio > 0).float(), src_mask.bool(), tgt_mask.bool())
        with torch.no_grad():
            coarse_scores = output['coarse_log_plan'][:, :-1, :-1].masked_fill(
                ~tgt_mask.bool()[:, None, :], -float('inf'))
            coarse_pick = coarse_scores.argmax(dim=-1, keepdim=True)
            coarse_rows = src_mask.bool() & ((ratio > 0).sum(-1) > 0)
            coarse_ir = ((ratio > 0).float().gather(-1, coarse_pick).squeeze(-1)[coarse_rows].mean()
                         if bool(coarse_rows.any()) else output['coarse_log_plan'].new_zeros(()))
        if self.use_overlap_head:
            coarse_overlap_loss = 0.5 * (
                self._balanced_bce(output['src_super_overlap'][src_mask], src_visibility[src_mask])
                + self._balanced_bce(output['tgt_super_overlap'][tgt_mask], tgt_visibility[tgt_mask]))
        else:
            coarse_overlap_loss = output['coarse_log_plan'].new_zeros(())

        # ---- point level overlap, Eq. (10) --------------------------------------------
        src_gt = torch.zeros_like(output['src_overlap'])
        tgt_gt = torch.zeros_like(output['tgt_overlap'])
        src_gt[correspondence[:, 0]] = 1.0
        tgt_gt[correspondence[:, 1]] = 1.0
        fine_overlap_loss = (0.5 * (F.binary_cross_entropy(output['src_overlap'], src_gt)
                                    + F.binary_cross_entropy(output['tgt_overlap'], tgt_gt))
                             if self.use_overlap_head
                             else output['coarse_log_plan'].new_zeros(()))

        # ---- point matching inside the patches, Eq. (8) and (9) -----------------------
        plan = output['patch_log_plan']
        if plan.shape[0] > 0:
            pair = output['patch_batch']
            rot, trans = batch['rot'].to(device), batch['trans'].to(device)
            src_xyz = batch['pcd_src'].to(device)[output['patch_src_index']]
            tgt_xyz = batch['pcd_tgt'].to(device)[output['patch_tgt_index']]
            src_xyz = torch.einsum('pij,pkj->pki', rot[pair], src_xyz) + trans[pair].transpose(1, 2)
            hit = (torch.cdist(src_xyz, tgt_xyz) < self.matching_radius).float()
            hit = hit * output['patch_src_valid'][..., None] * output['patch_tgt_valid'][:, None, :]
            # A point whose partner sits in another patch belongs in the slack column --
            # about three quarters of them do (scripts/diagnose_ocfnet.py). Supervising the
            # augmented plan teaches the model to reject those instead of inventing a match,
            # which is what CoFiNet's local_scores_gt does.
            if 'patch_src_score' in output:
                # the pair-conditional score is exactly "has a partner in this patch"
                src_here = (hit.sum(-1) > 0).float()
                tgt_here = (hit.sum(-2) > 0).float()
                pair_overlap_loss = 0.5 * (
                    self._balanced_bce(output['patch_src_score'][output['patch_src_valid']],
                                       src_here[output['patch_src_valid']])
                    + self._balanced_bce(output['patch_tgt_score'][output['patch_tgt_valid']],
                                         tgt_here[output['patch_tgt_valid']]))
            else:
                pair_overlap_loss = plan.new_zeros(())

            row_slack = (1 - hit.sum(-1)).clamp(min=0) * output['patch_src_valid']
            col_slack = (1 - hit.sum(-2)).clamp(min=0) * output['patch_tgt_valid']
            if not self.fine_dustbin:
                row_slack = torch.zeros_like(row_slack); col_slack = torch.zeros_like(col_slack)
            target = torch.cat([
                torch.cat([hit, row_slack[..., None]], dim=-1),
                torch.cat([col_slack[:, None, :], torch.zeros_like(row_slack[:, :1, None])],
                          dim=-1)], dim=1)
            fine_loss = self._matching_loss(plan, target)
            infonce_loss = self.patch_infonce(plan, hit.bool(), output['patch_src_valid'])
            inlier_loss = self.inlier_ratio_loss(plan, hit, output['patch_src_valid'],
                                                 output['patch_tgt_valid'])
            with torch.no_grad():
                # inlier ratio of the correspondences Eq. (6) would read off these plans:
                # how often the best target of a source point really is a true match.
                # `matching_radius` is the radius the losses use (0.0375 m); the benchmark
                # calls a correspondence an inlier at 0.1 m, so report that one too -- it is
                # the number scripts/evaluate_ocfnet.py prints and the one to select on.
                picked = plan[:, :-1, :-1].masked_fill(
                    ~output['patch_tgt_valid'][:, None, :], -float('inf')).argmax(dim=-1, keepdim=True)
                valid = output['patch_src_valid']
                correct = hit.gather(-1, picked).squeeze(-1)[valid]
                inlier_ratio = correct.mean() if correct.numel() else plan.new_zeros(())

                # The ceiling: points whose partner is actually inside the patch they are
                # matched against. (`patch_recall` used to count patch *pairs* holding a
                # correspondence, which is ~1 by construction -- training picks the pairs
                # because they hold one.) And the floor: what a matcher that learned
                # nothing scores, since a patch is a 20 cm cell and the radius is 10 cm, so
                # a random guess inside the right patch is an inlier about a quarter of the
                # time. `inlier_ratio` is only meaningful between the two.
                reachable = (hit.sum(-1) > 0).float()[valid].mean()
                chance = (hit.sum(-1) / output['patch_tgt_valid'].sum(-1, keepdim=True)
                          .clamp(min=1))[valid].mean()
        else:
            fine_loss = infonce_loss = pair_overlap_loss = plan.new_zeros(())
            inlier_loss = plan.new_zeros(())
            inlier_ratio = reachable = chance = plan.new_zeros(())

        descriptor_loss = self.descriptor_loss(output, batch, device)

        loss = (self.w_coarse * coarse_loss + self.w_fine * fine_loss
                + self.w_coarse_overlap * coarse_overlap_loss
                + self.w_fine_overlap * fine_overlap_loss
                + self.w_descriptor * descriptor_loss
                + self.w_patch_infonce * infonce_loss
                + self.w_coarse_infonce * coarse_infonce_loss
                + self.w_coarse_circle * coarse_circle_loss
                + self.w_coarse_inlier * coarse_inlier_loss
                + self.w_fine_overlap * pair_overlap_loss
                + self.w_inlier * inlier_loss)
        return {'loss': loss, 'coarse_loss': coarse_loss, 'fine_loss': fine_loss,
                'coarse_circle_loss': coarse_circle_loss,
                'coarse_infonce_loss': coarse_infonce_loss,
                'coarse_inlier_loss': coarse_inlier_loss, 'coarse_ir': coarse_ir,
                'coarse_overlap_loss': coarse_overlap_loss, 'fine_overlap_loss': fine_overlap_loss,
                'descriptor_loss': descriptor_loss, 'infonce_loss': infonce_loss,
                'pair_overlap_loss': pair_overlap_loss, 'inlier_loss': inlier_loss,
                'inlier_ratio': inlier_ratio, 'ir_ceiling': reachable,
                'ir_chance': chance}
