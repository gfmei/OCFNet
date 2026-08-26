"""
Sparse-convolutional backbone of Predator, ported from MinkowskiEngine to spconv 2.x.

Layer-by-layer mapping (see also models/common.py and lib/spconv_utils.py):

	ME.MinkowskiConvolution(k, stride=1)          -> spconv.SubMConv3d(k)
	    a stride-1 Minkowski convolution writes to the input coordinates only,
	    i.e. it is a submanifold convolution
	ME.MinkowskiConvolution(k=3, stride=2)        -> spconv.SparseConv3d(3, stride=2, padding=1)
	ME.MinkowskiConvolutionTranspose(k=3, s=2)    -> spconv.SparseInverseConv3d(3, indice_key=<paired down conv>)
	    keying the transposed convolution to its encoder counterpart restores exactly
	    the encoder coordinates (same order), so the skip connections become a plain
	    torch.cat over features instead of ME.cat
	ME.SparseTensor(f, coordinate_map_key=..)     -> x.replace_feature(f)
	ME.MinkowskiBatchNorm / MinkowskiInstanceNorm -> models.common.SparseBatchNorm / SparseInstanceNorm

Convolutions that share an `indice_key` share their rulebook, so a key encodes
(resolution, kernel size). Pushing both src and tgt through the same modules is safe
because spconv caches rulebooks on the tensor, not on the layer.

One deliberate difference: for a stride-2 kernel-3 convolution MinkowskiEngine emits
one output voxel per floor(coord / 2), while spconv emits every output voxel whose
receptive field contains an active input, which is a slightly larger set on odd
coordinates. Architecture, parameter shapes and parameter count are unchanged, but
the two libraries are not numerically identical, so MinkowskiEngine checkpoints
cannot be loaded here -- the model has to be trained with this code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from models.common import get_norm, sparse_relu, sparse_cat, to_dense_batch, from_dense_batch
from models.attention import OverlapAttention
from models.residual_block import get_block


class ResUNet2(nn.Module):
	NORM_TYPE = None
	BLOCK_NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 32, 64, 64, 128]
	# the encoder halves the resolution three times, so the bottleneck is a 1/8 grid
	BOTTLENECK_STRIDE = 8

	def __init__(self,config,D=3):
		super(ResUNet2, self).__init__()
		NORM_TYPE = self.NORM_TYPE
		BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
		CHANNELS = self.CHANNELS
		TR_CHANNELS = self.TR_CHANNELS
		bn_momentum = config.bn_momentum
		self.normalize_feature = config.normalize_feature
		self.voxel_size = config.voxel_size
		feature_dim = config.attention_feats_dim

		self.conv1 = spconv.SubMConv3d(
			in_channels=config.in_feats_dim,
			out_channels=CHANNELS[1],
			kernel_size=config.conv1_kernel_size,
			bias=False,
			indice_key='subm_s1_stem')
		self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

		self.block1 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s1')

		self.conv2 = spconv.SparseConv3d(
			in_channels=CHANNELS[1],
			out_channels=CHANNELS[2],
			kernel_size=3,
			stride=2,
			padding=1,
			dilation=1,
			bias=False,
			indice_key='down_s2')
		self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.block2 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s2')

		self.conv3 = spconv.SparseConv3d(
			in_channels=CHANNELS[2],
			out_channels=CHANNELS[3],
			kernel_size=3,
			stride=2,
			padding=1,
			dilation=1,
			bias=False,
			indice_key='down_s4')
		self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.block3 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s4')

		self.conv4 = spconv.SparseConv3d(
			in_channels=CHANNELS[3],
			out_channels=CHANNELS[4],
			kernel_size=3,
			stride=2,
			padding=1,
			dilation=1,
			bias=False,
			indice_key='down_s8')
		self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.block4 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s8')

		# adapt input tensor here
		self.conv4_tr = spconv.SparseInverseConv3d(
			in_channels = feature_dim + 2,
			out_channels=TR_CHANNELS[4],
			kernel_size=3,
			bias=False,
			indice_key='down_s8')
		self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.block4_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s4')

		self.conv3_tr = spconv.SparseInverseConv3d(
			in_channels=CHANNELS[3] + TR_CHANNELS[4],
			out_channels=TR_CHANNELS[3],
			kernel_size=3,
			bias=False,
			indice_key='down_s4')
		self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.block3_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s2')

		self.conv2_tr = spconv.SparseInverseConv3d(
			in_channels=CHANNELS[2] + TR_CHANNELS[3],
			out_channels=TR_CHANNELS[2],
			kernel_size=3,
			bias=False,
			indice_key='down_s2')
		self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.block2_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D,
			indice_key='subm_s1')

		self.conv1_tr = spconv.SubMConv3d(
			in_channels=CHANNELS[1] + TR_CHANNELS[2],
			out_channels=TR_CHANNELS[1],
			kernel_size=1,
			bias=False,
			indice_key='subm_s1_1x1')

		self.final = spconv.SubMConv3d(
			in_channels=TR_CHANNELS[1],
			out_channels=config.out_feats_dim + 2,
			kernel_size=1,
			bias=True,
			indice_key='subm_s1_final')


		#############
		# Overlap attention module
		self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
		self.bottle = nn.Linear(CHANNELS[4], feature_dim, bias=True)
		self.attention = OverlapAttention(config.num_head, feature_dim, config.nets,
			rope=config.get('rope', True), rope_theta=config.get('rope_theta', 100.0))
		self.proj_feats = nn.Linear(feature_dim, feature_dim, bias=True)
		self.proj_score = nn.Linear(feature_dim, 1, bias=True)




	def forward(self, stensor_src, stensor_tgt):
		################################
		# encode src
		src_s1 = self.conv1(stensor_src)
		src_s1 = self.norm1(src_s1)
		src_s1 = self.block1(src_s1)
		src = sparse_relu(src_s1)

		src_s2 = self.conv2(src)
		src_s2 = self.norm2(src_s2)
		src_s2 = self.block2(src_s2)
		src = sparse_relu(src_s2)

		src_s4 = self.conv3(src)
		src_s4 = self.norm3(src_s4)
		src_s4 = self.block3(src_s4)
		src = sparse_relu(src_s4)

		src_s8 = self.conv4(src)
		src_s8 = self.norm4(src_s8)
		src_s8 = self.block4(src_s8)
		src = sparse_relu(src_s8)


		################################
		# encode tgt
		tgt_s1 = self.conv1(stensor_tgt)
		tgt_s1 = self.norm1(tgt_s1)
		tgt_s1 = self.block1(tgt_s1)
		tgt = sparse_relu(tgt_s1)

		tgt_s2 = self.conv2(tgt)
		tgt_s2 = self.norm2(tgt_s2)
		tgt_s2 = self.block2(tgt_s2)
		tgt = sparse_relu(tgt_s2)

		tgt_s4 = self.conv3(tgt)
		tgt_s4 = self.norm3(tgt_s4)
		tgt_s4 = self.block3(tgt_s4)
		tgt = sparse_relu(tgt_s4)

		tgt_s8 = self.conv4(tgt)
		tgt_s8 = self.norm4(tgt_s8)
		tgt_s8 = self.block4(tgt_s8)
		tgt = sparse_relu(tgt_s8)


		################################
		# overlap attention module
		# The batch is laid out as one padded block [B, N_max, C] plus a validity mask, and
		# every attention in here is masked, so the pairs of a batch never mix. The original
		# implementation concatenated them and could only be trained with batch_size = 1.
		src_batch, tgt_batch = src.indices[:,0].long(), tgt.indices[:,0].long()
		src_feats, src_mask, src_index = to_dense_batch(src.features, src_batch, src.batch_size)
		tgt_feats, tgt_mask, tgt_index = to_dense_batch(tgt.features, tgt_batch, tgt.batch_size)
		# the attention consumes the bottleneck voxel coordinates (1/8 grid) through the
		# rotary embedding of models/attention.py, which only depends on coordinate differences --
		# so the shift that lib/spconv_utils.make_sparse_tensor applies cancels out
		src_pcd, _, _ = to_dense_batch(src.indices[:,1:].float(), src_batch, src.batch_size)
		tgt_pcd, _, _ = to_dense_batch(tgt.indices[:,1:].float(), tgt_batch, tgt.batch_size)

		# 1. project the bottleneck feature
		src_feats, tgt_feats = self.bottle(src_feats), self.bottle(tgt_feats)

		# 2. let the two clouds exchange information and score the overlap
		src_feats, tgt_feats= self.attention(src_pcd, tgt_pcd, src_feats, tgt_feats, src_mask, tgt_mask)

		src_feats, src_scores = self.proj_feats(src_feats), self.proj_score(src_feats)  #[B, N, 1]
		tgt_feats, tgt_scores = self.proj_feats(tgt_feats), self.proj_score(tgt_feats)


		# 3. get cross-overlap scores
		src_feats_norm = F.normalize(src_feats, p=2, dim=-1)  #[B, N, C]
		tgt_feats_norm = F.normalize(tgt_feats, p=2, dim=-1)
		inner_products = torch.matmul(src_feats_norm, tgt_feats_norm.transpose(1,2))  #[B, N, M]
		temperature = torch.exp(self.epsilon) + 0.03
		pair_mask = src_mask[:,:,None] & tgt_mask[:,None,:]
		src_attention = F.softmax((inner_products / temperature).masked_fill(~pair_mask, -1e9), dim=2)
		tgt_attention = F.softmax((inner_products.transpose(1,2) / temperature).masked_fill(~pair_mask.transpose(1,2), -1e9), dim=2)
		src_scores_x = torch.matmul(src_attention, tgt_scores)
		tgt_scores_x = torch.matmul(tgt_attention, src_scores)

		# 4. update sparse tensor, dropping the padding on the way back
		src_feats = torch.cat([src_feats, src_scores, src_scores_x], dim=-1)
		tgt_feats = torch.cat([tgt_feats, tgt_scores, tgt_scores_x], dim=-1)
		src = src.replace_feature(from_dense_batch(src_feats, src_index))
		tgt = tgt.replace_feature(from_dense_batch(tgt_feats, tgt_index))


		################################
		# decoder src
		src = self.conv4_tr(src)
		src = self.norm4_tr(src)
		src = self.block4_tr(src)
		src_s4_tr = sparse_relu(src)

		src = sparse_cat(src_s4_tr, src_s4)

		src = self.conv3_tr(src)
		src = self.norm3_tr(src)
		src = self.block3_tr(src)
		src_s2_tr = sparse_relu(src)

		src = sparse_cat(src_s2_tr, src_s2)

		src = self.conv2_tr(src)
		src = self.norm2_tr(src)
		src = self.block2_tr(src)
		src_s1_tr = sparse_relu(src)

		src = sparse_cat(src_s1_tr, src_s1)
		src = self.conv1_tr(src)
		src = sparse_relu(src)
		src = self.final(src)

		################################
		# decoder tgt
		tgt = self.conv4_tr(tgt)
		tgt = self.norm4_tr(tgt)
		tgt = self.block4_tr(tgt)
		tgt_s4_tr = sparse_relu(tgt)

		tgt = sparse_cat(tgt_s4_tr, tgt_s4)

		tgt = self.conv3_tr(tgt)
		tgt = self.norm3_tr(tgt)
		tgt = self.block3_tr(tgt)
		tgt_s2_tr = sparse_relu(tgt)

		tgt = sparse_cat(tgt_s2_tr, tgt_s2)

		tgt = self.conv2_tr(tgt)
		tgt = self.norm2_tr(tgt)
		tgt = self.block2_tr(tgt)
		tgt_s1_tr = sparse_relu(tgt)

		tgt = sparse_cat(tgt_s1_tr, tgt_s1)
		tgt = self.conv1_tr(tgt)
		tgt = sparse_relu(tgt)
		tgt = self.final(tgt)

		################################
		# output features and scores
		sigmoid = nn.Sigmoid()
		src_feats, src_overlap, src_saliency = src.features[:,:-2], src.features[:,-2], src.features[:,-1]
		tgt_feats, tgt_overlap, tgt_saliency = tgt.features[:,:-2], tgt.features[:,-2], tgt.features[:,-1]

		src_overlap= torch.clamp(sigmoid(src_overlap.view(-1)),min=0,max=1)
		src_saliency = torch.clamp(sigmoid(src_saliency.view(-1)),min=0,max=1)
		tgt_overlap = torch.clamp(sigmoid(tgt_overlap.view(-1)),min=0,max=1)
		tgt_saliency = torch.clamp(sigmoid(tgt_saliency.view(-1)),min=0,max=1)

		src_feats = F.normalize(src_feats, p=2, dim=1)
		tgt_feats = F.normalize(tgt_feats, p=2, dim=1)

		scores_overlap = torch.cat([src_overlap, tgt_overlap], dim=0)
		scores_saliency = torch.cat([src_saliency, tgt_saliency], dim=0)

		return src_feats,  tgt_feats, scores_overlap, scores_saliency



class ResUNetBN2(ResUNet2):
	NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
	NORM_TYPE = 'IN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 64, 128]
	BLOCK_NORM_TYPE = 'IN'

	# CHANNELS = [None, 64, 128, 256, 512]
	# TR_CHANNELS = [None, 64, 128, 128, 256]


class ResUNetBN2D(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 128, 128, 128, 256]
	TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'
