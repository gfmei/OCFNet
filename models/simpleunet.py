"""
FCGF-style plain UNets, ported from MinkowskiEngine to spconv 2.x.

Predator itself uses models/resunet.py; these backbones are kept for reference and
follow the same translation rules (see the module docstring of models/resunet.py):
stride-1 Minkowski convolutions become submanifold convolutions, stride-2 ones become
SparseConv3d(3, stride=2, padding=1), and every transposed convolution is keyed to the
encoder convolution whose coordinates it has to restore.

Note that SimpleNet3 downsamples four times, so feed it tensors built with
lib.spconv_utils.make_sparse_tensor(..., pad_multiple=16).
"""
import torch
import spconv.pytorch as spconv
from models.common import get_norm, sparse_relu, sparse_cat


class SimpleNet(spconv.SparseModule):
  NORM_TYPE = None
  CHANNELS = [None, 32, 64, 128]
  TR_CHANNELS = [None, 32, 32, 64]

  def __init__(self,
               in_channels=3,
               out_channels=32,
               bn_momentum=0.1,
               normalize_feature=None,
               conv1_kernel_size=None,
               D=3):
    super(SimpleNet, self).__init__()
    NORM_TYPE = self.NORM_TYPE
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = normalize_feature
    self.conv1 = spconv.SubMConv3d(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=conv1_kernel_size,
        bias=False,
        indice_key='subm_s1_stem')
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

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

    self.conv3_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[3],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        bias=False,
        indice_key='down_s4')
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        bias=False,
        indice_key='down_s2')
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = spconv.SubMConv3d(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=3,
        bias=False,
        indice_key='subm_s1')
    self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = spconv.SubMConv3d(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        bias=True,
        indice_key='subm_s1_final')

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out = sparse_relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out = sparse_relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out = sparse_relu(out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out_s2_tr = sparse_relu(out)

    out = sparse_cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out_s1_tr = sparse_relu(out)

    out = sparse_cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = self.norm1_tr(out)
    out = sparse_relu(out)

    out = self.final(out)

    if self.normalize_feature:
      return out.replace_feature(
          out.features / torch.norm(out.features, p=2, dim=1, keepdim=True))
    else:
      return out


class SimpleNetIN(SimpleNet):
  NORM_TYPE = 'IN'


class SimpleNetBN(SimpleNet):
  NORM_TYPE = 'BN'


class SimpleNetBNE(SimpleNetBN):
  CHANNELS = [None, 16, 32, 32]
  TR_CHANNELS = [None, 16, 16, 32]


class SimpleNetINE(SimpleNetBNE):
  NORM_TYPE = 'IN'


class SimpleNet2(spconv.SparseModule):
  NORM_TYPE = None
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 32, 64, 64]

  def __init__(self, in_channels=3, out_channels=32, bn_momentum=0.1, D=3, config=None):
    super(SimpleNet2, self).__init__()
    NORM_TYPE = self.NORM_TYPE
    bn_momentum = config.bn_momentum
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = config.normalize_feature
    self.conv1 = spconv.SubMConv3d(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=config.conv1_kernel_size,
        bias=False,
        indice_key='subm_s1_stem')
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

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

    self.conv4_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[4],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        bias=False,
        indice_key='down_s8')
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        bias=False,
        indice_key='down_s4')
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        bias=False,
        indice_key='down_s2')
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = spconv.SubMConv3d(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=3,
        bias=False,
        indice_key='subm_s1')
    self.norm1_tr = get_norm(NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

    self.final = spconv.SubMConv3d(
        in_channels=TR_CHANNELS[1],
        out_channels=out_channels,
        kernel_size=1,
        bias=True,
        indice_key='subm_s1_final')

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out = sparse_relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out = sparse_relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out = sparse_relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out = sparse_relu(out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out_s4_tr = sparse_relu(out)

    out = sparse_cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out_s2_tr = sparse_relu(out)

    out = sparse_cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out_s1_tr = sparse_relu(out)

    out = sparse_cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)
    out = self.norm1_tr(out)
    out = sparse_relu(out)

    out = self.final(out)

    if self.normalize_feature:
      return out.replace_feature(
          out.features / torch.norm(out.features, p=2, dim=1, keepdim=True))
    else:
      return out


class SimpleNetIN2(SimpleNet2):
  NORM_TYPE = 'IN'


class SimpleNetBN2(SimpleNet2):
  NORM_TYPE = 'BN'


class SimpleNetBN2B(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 64, 64, 64, 64]


class SimpleNetBN2C(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]


class SimpleNetBN2D(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256]
  TR_CHANNELS = [None, 32, 64, 64, 128]


class SimpleNetBN2E(SimpleNet2):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128]
  TR_CHANNELS = [None, 16, 32, 32, 64]


class SimpleNetIN2E(SimpleNetBN2E):
  NORM_TYPE = 'IN'


class SimpleNet3(spconv.SparseModule):
  NORM_TYPE = None
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 32, 64, 64, 128]

  def __init__(self, in_channels=3, out_channels=32, bn_momentum=0.1, D=3, config=None):
    super(SimpleNet3, self).__init__()
    NORM_TYPE = self.NORM_TYPE
    bn_momentum = config.bn_momentum
    CHANNELS = self.CHANNELS
    TR_CHANNELS = self.TR_CHANNELS
    self.normalize_feature = config.normalize_feature
    self.conv1 = spconv.SubMConv3d(
        in_channels=in_channels,
        out_channels=CHANNELS[1],
        kernel_size=config.conv1_kernel_size,
        bias=False,
        indice_key='subm_s1_stem')
    self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

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

    self.conv5 = spconv.SparseConv3d(
        in_channels=CHANNELS[4],
        out_channels=CHANNELS[5],
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        bias=False,
        indice_key='down_s16')
    self.norm5 = get_norm(NORM_TYPE, CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv5_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[5],
        out_channels=TR_CHANNELS[5],
        kernel_size=3,
        bias=False,
        indice_key='down_s16')
    self.norm5_tr = get_norm(NORM_TYPE, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

    self.conv4_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[4] + TR_CHANNELS[5],
        out_channels=TR_CHANNELS[4],
        kernel_size=3,
        bias=False,
        indice_key='down_s8')
    self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

    self.conv3_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[3] + TR_CHANNELS[4],
        out_channels=TR_CHANNELS[3],
        kernel_size=3,
        bias=False,
        indice_key='down_s4')
    self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

    self.conv2_tr = spconv.SparseInverseConv3d(
        in_channels=CHANNELS[2] + TR_CHANNELS[3],
        out_channels=TR_CHANNELS[2],
        kernel_size=3,
        bias=False,
        indice_key='down_s2')
    self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

    self.conv1_tr = spconv.SubMConv3d(
        in_channels=CHANNELS[1] + TR_CHANNELS[2],
        out_channels=TR_CHANNELS[1],
        kernel_size=1,
        bias=True,
        indice_key='subm_s1_1x1')

  def forward(self, x):
    out_s1 = self.conv1(x)
    out_s1 = self.norm1(out_s1)
    out = sparse_relu(out_s1)

    out_s2 = self.conv2(out)
    out_s2 = self.norm2(out_s2)
    out = sparse_relu(out_s2)

    out_s4 = self.conv3(out)
    out_s4 = self.norm3(out_s4)
    out = sparse_relu(out_s4)

    out_s8 = self.conv4(out)
    out_s8 = self.norm4(out_s8)
    out = sparse_relu(out_s8)

    out_s16 = self.conv5(out)
    out_s16 = self.norm5(out_s16)
    out = sparse_relu(out_s16)

    out = self.conv5_tr(out)
    out = self.norm5_tr(out)
    out_s8_tr = sparse_relu(out)

    out = sparse_cat(out_s8_tr, out_s8)

    out = self.conv4_tr(out)
    out = self.norm4_tr(out)
    out_s4_tr = sparse_relu(out)

    out = sparse_cat(out_s4_tr, out_s4)

    out = self.conv3_tr(out)
    out = self.norm3_tr(out)
    out_s2_tr = sparse_relu(out)

    out = sparse_cat(out_s2_tr, out_s2)

    out = self.conv2_tr(out)
    out = self.norm2_tr(out)
    out_s1_tr = sparse_relu(out)

    out = sparse_cat(out_s1_tr, out_s1)
    out = self.conv1_tr(out)

    if self.normalize_feature:
      return out.replace_feature(
          out.features / torch.norm(out.features, p=2, dim=1, keepdim=True))
    else:
      return out


class SimpleNetIN3(SimpleNet3):
  NORM_TYPE = 'IN'


class SimpleNetBN3(SimpleNet3):
  NORM_TYPE = 'BN'


class SimpleNetBN3B(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 64, 64, 64, 128]


class SimpleNetBN3C(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 32, 64, 128, 128]


class SimpleNetBN3D(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 32, 64, 128, 256, 512]
  TR_CHANNELS = [None, 32, 64, 64, 128, 128]


class SimpleNetBN3E(SimpleNet3):
  NORM_TYPE = 'BN'
  CHANNELS = [None, 16, 32, 64, 128, 256]
  TR_CHANNELS = [None, 16, 32, 32, 64, 128]


class SimpleNetIN3E(SimpleNetBN3E):
  NORM_TYPE = 'IN'
