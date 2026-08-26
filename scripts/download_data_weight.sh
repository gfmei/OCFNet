#!/bin/bash
# 3DMatch / 3DLoMatch fragments + ModelNet40 (data.zip, ~940 MB). Unpacks to ./data, which is
# what `root: data/indoor` in the configs expects. The old predator.zip link is gone, the
# files now live directly under ~gseg.
set -e
BASE=https://share.phys.ethz.ch/~gseg/Predator

wget --no-check-certificate --show-progress -c $BASE/data.zip
unzip -q -o data.zip
rm data.zip

# The released Predator weights are MinkowskiEngine checkpoints ($BASE/weights/sparseIndoor.pth
# for the sparse model). They do NOT load into this spconv port -- see the README, the two
# libraries generate different voxels for a stride-2 kernel-3 convolution. Uncomment if you
# want them for reference against the original implementation.
# wget --no-check-certificate --show-progress -c $BASE/weights.zip
# unzip -q -o weights.zip && rm weights.zip
