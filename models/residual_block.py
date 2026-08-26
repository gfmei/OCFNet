import spconv.pytorch as spconv

from models.common import get_norm, sparse_relu


class BasicBlockBase(spconv.SparseModule):
  expansion = 1
  NORM_TYPE = 'BN'

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               bn_momentum=0.1,
               D=3,
               indice_key=None):
    super(BasicBlockBase, self).__init__()
    # ME.MinkowskiConvolution with stride 1 only writes to the input coordinates, which is
    # what a submanifold convolution does. Both convolutions of the block share `indice_key`
    # so the rulebook is built once per resolution and reused by src, tgt and the decoder.
    assert stride == 1 and dilation == 1, \
        'ResUNet2 only instantiates stride-1 / dilation-1 residual blocks; a strided or ' \
        'dilated block needs its own indice_key because its rulebook differs'

    self.conv1 = spconv.SubMConv3d(
        inplanes, planes, kernel_size=3, bias=False, indice_key=indice_key)
    self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.conv2 = spconv.SubMConv3d(
        planes, planes, kernel_size=3, bias=False, indice_key=indice_key)
    self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
    self.downsample = downsample

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = sparse_relu(out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out = out.replace_feature(out.features + residual.features)
    out = sparse_relu(out)

    return out


class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'


def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3,
              indice_key=None):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D,
                        indice_key)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D,
                        indice_key)
  else:
    raise ValueError(f'Type {norm_type}, not defined')
