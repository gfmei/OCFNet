# Overlap-guided Coarse-to-fine Correspondence Prediction for Point Cloud Registration
*Guofeng Mei<sup>\*</sup>, Xiaoshui Huang, Jian Zhang, Qiang Wu*


This is the official PyTorch implementation of our paper Overlap-guided Coarse-to-fine Correspondence prediction for Point Cloud Registration that has been accepted to ICME 2022.

The code base is [OverlapPredator.Mink](https://github.com/overlappredator/OverlapPredator.Mink)
with the sparse backbone **ported from MinkowskiEngine to [spconv](https://github.com/traveller59/spconv) 2.x**,
plus the fixes needed to run it on a current python / torch / numpy / open3d stack.

## Why the port

MinkowskiEngine is unmaintained and does not build against recent CUDA/PyTorch. spconv ships
prebuilt wheels per CUDA version (`pip install spconv-cu126`) and needs no compilation.

### How MinkowskiEngine maps to spconv

| MinkowskiEngine | spconv | where |
|---|---|---|
| `ME.MinkowskiConvolution(k, stride=1)` | `spconv.SubMConv3d(k)` | a stride-1 Minkowski conv only writes to the input coordinates, i.e. it is submanifold |
| `ME.MinkowskiConvolution(3, stride=2)` | `spconv.SparseConv3d(3, stride=2, padding=1)` | encoder downsampling |
| `ME.MinkowskiConvolutionTranspose(3, stride=2)` | `spconv.SparseInverseConv3d(3, indice_key=<paired down conv>)` | keying it to the encoder conv restores exactly the encoder coordinates, in the same order |
| `ME.cat(a, b)` | `a.replace_feature(torch.cat([a.features, b.features], 1))` | skip connections, coordinates are identical by construction |
| `ME.SparseTensor(f, coordinate_map_key=...)` | `x.replace_feature(f)` | feature update after the overlap-attention module |
| `ME.MinkowskiBatchNorm` / `MinkowskiInstanceNorm` | `models/common.py: SparseBatchNorm` / `SparseInstanceNorm` | the instance norm is re-implemented per (sample, channel) over the active voxels |
| `ME.utils.sparse_quantize` / `sparse_collate` | `lib/spconv_utils.py` | data loading |
| `ME.SparseTensor(feats, coordinates=coords)` | `lib/spconv_utils.make_sparse_tensor` | shifts the cloud into the positive octant, spconv indexes a dense grid |

Convolutions that share an `indice_key` share their rulebook, so a key encodes
(resolution, kernel size). src and tgt go through the same modules because spconv caches
rulebooks on the tensor rather than on the layer.

### Things to be aware of

* **The released MinkowskiEngine weights do not load here.** For a stride-2 kernel-3
  convolution MinkowskiEngine emits one output voxel per `floor(coord / 2)`, while spconv
  emits every output voxel whose receptive field contains an active input -- a slightly
  larger set on odd coordinates. Layer types, parameter shapes and the parameter count
  (10.86 M for `ResUNetBN2C`) are unchanged, but the two libraries are not numerically
  identical, so the model has to be trained with this code.
* Coordinates are shifted to be non-negative (spconv indexes a grid, MinkowskiEngine hashed
  the coordinates). The backbone only feeds those coordinates to the kNN graph in
  `models/attention.py`, which is invariant to a global translation, so nothing downstream changes.
* Row order is preserved end to end: the backbone returns one row per input voxel, in the
  input order, which is what the correspondence indexing in the losses relies on.
  `scripts/test_spconv_forward.py` asserts this both on the coordinates themselves and by
  permuting the input voxels (max deviation 1e-4, i.e. float noise).
* Predator aggregated its self-attention over a kNN graph, which was the one order-sensitive
  piece: bottleneck voxels sit on a regular grid, so ~80% of them have a distance tie at k=10
  and `topk` breaks ties by storage position, which spconv does not fix. That is why a 'self'
  layer is plain masked attention here (see below).

### Other changes needed to run on a current stack

* `tensorboardX` -> `torch.utils.tensorboard`, `nibabel.quaternions` -> `scipy` rotations,
  `coloredlogs` / `GitPython` are optional now
* `iterator.next()` -> `next(iterator)`, `np.float`/`np.int`/`np.bool` -> builtins,
  `torch.load(..., weights_only=False)` for the pickled point clouds (torch >= 2.6)
* open3d's registration module moved under `o3d.pipelines`, and
  `RANSACConvergenceCriteria(max_iteration, confidence)` replaced the old `max_validation`
* correspondence search uses `scipy.spatial.cKDTree` instead of the open3d KDTree: it keeps
  the data loader independent of open3d (whose bindings are sensitive to the numpy version)
  and replaces a python loop over every source point with one vectorised radius query
* `scripts/demo.py` was still the KPConv version upstream; it now runs the sparse pipeline
  and falls back to writing `.ply` files when there is no display

## Batched training and the attention module

Upstream could only train one pair at a time: the overlap-attention module concatenated the
whole batch into a single cloud, so points of different pairs attended to each other. Two
changes fix that.

**Masks everywhere.** The bottleneck is laid out as a padded block `[B, N_max, C]` plus a
validity mask (`models/common.to_dense_batch`). The layout is point-major throughout: every
1x1 convolution over points is a per-point linear map, so the modules are written with
`nn.Linear` and nothing has to be transposed between the encoder, the attention, the overlap
head and the transport cost. Attention scores over padded keys are set to
-1e9 before the softmax, the cross-overlap softmax is masked to the block diagonal, and the
instance norms use `models/common.masked_instance_norm` so the statistics of a sample do not
depend on how long the other samples of the batch happen to be. The loss is computed pair by
pair and averaged (`MetricLoss.forward` takes `len_batch`), and the collate stacks one
rot/trans/scale per pair.

**Self-attention with RoPE-3D instead of the kNN graph block.** A 'self' layer is now masked
multi-head attention built on `torch.nn.functional.scaled_dot_product_attention`, with
positions injected as a 3D rotary embedding over the voxel coordinates, following
[Volt](https://github.com/YilmazKadir/Volt): the head dimension is split per axis (3/8, 3/8,
2/8) and query/key are rotated by an angle linear in the coordinate, so scores depend on the
*relative* position of two points and are invariant to the coordinate shift the sparse
tensor applies. Cross-attention gets no position embedding on purpose -- the two clouds are
in different frames until they are registered.

`rope: True` is the setting to use. Earlier configurations here disabled it to match
CoFiNet's `ape: False`, on the reasoning that CoFiNet reaches 0.83 patch inlier ratio with no
positional encoding at all. That does not transfer: measured at matched epochs against an
otherwise identical run, RoPE-3D is worth about **+0.085 coarse inlier ratio and +0.089 fine
inlier ratio**, in both the overlap and the no-overlap condition. CoFiNet's choice was made
for a KPConv backbone. The paper's own geometric encoding
(`geometric_pos_enc`) is a third option, off by default for an unrelated reason: it measures
each point's radius from its *own* cloud's centroid, and under partial overlap the two
centroids are different physical points, so corresponding super-points get different codes.

Besides being batchable, this is cheaper: the kNN aggregation materialised a tensor of shape
`batch x channels x points x points`,
which is why a single pair peaked at 6.5 GiB and 1.6 s; the same pair now runs in 0.72 s at
0.85 GiB. It also removed a subtle non-determinism -- voxels on a regular grid produce
massive distance ties (~80% of the points at k=10) that `topk` resolved by storage position,
and spconv gives no guarantee about the order in which it emits voxels.

```shell
python main.py configs/train/predator_indoor.yaml  # batch_size 12
python main.py configs/train/indoor.yaml          # upstream setting: batch_size 1, iter_size 4
```

`scripts/test_batch_equivalence.py` verifies the masking: a batch of identical pairs must
agree with the single-pair run (leakage would change every attention distribution), and a
batch of different pairs -- i.e. real padding -- must reproduce each pair's individual
output. Both hold to 3e-4.

## Requirements

```shell
pip install -r requirements.txt   # pick the spconv wheel matching your CUDA, e.g. spconv-cu126
```

On our cluster the ready-made environment is `conda activate reg3d`
(python 3.11, torch 2.7.1+cu126, spconv-cu126 2.3.8, cumm-cu126 0.7.11, open3d 0.19,
numpy 2.4.6), which is what the SLURM scripts activate; override with `CONDA_ENV=<name>`.

Two traps on this cluster, both of which cost a debugging round already:

* **Never mix CUDA variants of the same package.** `spconv-cu120` and `spconv-cu126` unpack
  into the same `spconv/` directory, so with both installed the import resolves to whichever
  won the last write and the forward pass dies with a `Floating point exception`. Same story
  for `open3d` vs `open3d-cpu`, which leaves an `open3d/` directory without `__init__.py`.
  Keep one wheel per package and `pip install --force-reinstall --no-deps` after removing the
  other, because the uninstall takes shared files with it.
* **open3d must come from conda-forge**: the PyPI wheels for >= 0.19 are `manylinux_2_31` and
  the login/compute nodes are glibc 2.28 (`OSError: GLIBC_2.29 not found`), while the newest
  pip-installable version, 0.18, segfaults with numpy >= 2. `conda install -c conda-forge
  open3d=0.19` gives a build that works with numpy 2.4.
  Training and testing do not need open3d at all -- it is resolved lazily and only the RANSAC
  pose estimation, the KITTI ICP refinement and the demo use it.

## Data

```shell
bash scripts/download_data_weight.sh   # 3DMatch/3DLoMatch fragments (~940 MB) + weights
```

On our cluster the data is already at `/leonardo_work/AIFPT_agrifood/data/predator/data`,
and `data` in this repo is a symlink to it, so `root: data/indoor` in the configs resolves.

## Train / test

```shell
python main.py configs/train/indoor.yaml     # train on 3DMatch
python main.py configs/test/indoor.yaml      # dump features/scores for the benchmark
python scripts/evaluate_predator.py --source_path snapshot/indoor/3DLoMatch --n_points 1000 --benchmark 3DLoMatch
```

On SLURM:

```shell
sbatch scripts/slurm_smoke_test.sh     # ~2 min: forward/backward + 3 real training iterations
sbatch scripts/slurm_train_indoor.sh   # full training run
```

Predator trains with `batch_size: 1` and `iter_size: 4` (gradient accumulation) on a single
GPU; the overlap-attention module treats the whole batch as one cloud, so larger batches
change the method rather than just the throughput.

## Results

3DMatch and 3DLoMatch, correspondence-based RANSAC over the non-consecutive pairs the
benchmark script scores (1279 and 1726 respectively). RR is
registration recall weighted by pair count, IR the inlier ratio and FMR the feature match
recall, both with the mutual-nearest-neighbour check; RRE is the mean median rotation error
in degrees and RTE the mean median translation error in metres. RR, IR and FMR are
percentages. This is the protocol and the `est_traj/{benchmark}/{samples}/result` format
CoFiNet reports, so the numbers are directly comparable to theirs.

Two models, both trained for 150 epochs and evaluated at the last checkpoint:

- **our OCFNet** -- overlap head trained with the coarse and fine overlap losses, uniform
  transport marginals, RoPE-3D at the coarse level.
- **our improved CoFiNet** -- the same network without the overlap head, RoPE-3D at the
  coarse level. It isolates what the overlap head is worth as an auxiliary task, since that
  is the only difference between the two.

### 3DMatch

<table>
<thead>
  <tr><th rowspan="2">samples</th><th colspan="5">our OCFNet</th><th colspan="5">our improved CoFiNet</th></tr>
  <tr><th>RR</th><th>IR</th><th>FMR</th><th>RRE</th><th>RTE</th><th>RR</th><th>IR</th><th>FMR</th><th>RRE</th><th>RTE</th></tr>
</thead>
<tbody>
  <tr><td align="center">5000</td><td align="right">90.1</td><td align="right">60.8</td><td align="right">97.3</td><td align="right">2.37</td><td align="right">0.078</td><td align="right">89.7</td><td align="right">61.5</td><td align="right">96.3</td><td align="right">2.25</td><td align="right">0.073</td></tr>
  <tr><td align="center">2500</td><td align="right">89.8</td><td align="right">60.8</td><td align="right">97.3</td><td align="right">2.29</td><td align="right">0.075</td><td align="right">89.2</td><td align="right">61.5</td><td align="right">96.3</td><td align="right">2.30</td><td align="right">0.070</td></tr>
  <tr><td align="center">1000</td><td align="right">89.8</td><td align="right">60.9</td><td align="right">97.3</td><td align="right">2.17</td><td align="right">0.071</td><td align="right">89.3</td><td align="right">61.5</td><td align="right">96.3</td><td align="right">2.29</td><td align="right">0.073</td></tr>
  <tr><td align="center">500</td><td align="right">90.8</td><td align="right">60.8</td><td align="right">97.3</td><td align="right">2.32</td><td align="right">0.076</td><td align="right">89.2</td><td align="right">61.5</td><td align="right">96.3</td><td align="right">2.38</td><td align="right">0.072</td></tr>
  <tr><td align="center">250</td><td align="right">89.4</td><td align="right">60.6</td><td align="right">97.4</td><td align="right">2.30</td><td align="right">0.077</td><td align="right">89.0</td><td align="right">61.3</td><td align="right">96.2</td><td align="right">2.34</td><td align="right">0.075</td></tr>
</tbody>
</table>

### 3DLoMatch

<table>
<thead>
  <tr><th rowspan="2">samples</th><th colspan="5">our OCFNet</th><th colspan="5">our improved CoFiNet</th></tr>
  <tr><th>RR</th><th>IR</th><th>FMR</th><th>RRE</th><th>RTE</th><th>RR</th><th>IR</th><th>FMR</th><th>RRE</th><th>RTE</th></tr>
</thead>
<tbody>
  <tr><td align="center">5000</td><td align="right">55.1</td><td align="right">27.5</td><td align="right">77.2</td><td align="right">3.60</td><td align="right">0.106</td><td align="right">53.3</td><td align="right">27.6</td><td align="right">76.1</td><td align="right">3.46</td><td align="right">0.102</td></tr>
  <tr><td align="center">2500</td><td align="right">55.5</td><td align="right">27.5</td><td align="right">77.2</td><td align="right">3.50</td><td align="right">0.102</td><td align="right">53.2</td><td align="right">27.6</td><td align="right">76.1</td><td align="right">3.62</td><td align="right">0.111</td></tr>
  <tr><td align="center">1000</td><td align="right">55.4</td><td align="right">27.5</td><td align="right">77.2</td><td align="right">3.70</td><td align="right">0.110</td><td align="right">54.0</td><td align="right">27.6</td><td align="right">76.1</td><td align="right">3.59</td><td align="right">0.106</td></tr>
  <tr><td align="center">500</td><td align="right">55.3</td><td align="right">27.5</td><td align="right">77.2</td><td align="right">3.72</td><td align="right">0.109</td><td align="right">53.1</td><td align="right">27.6</td><td align="right">76.1</td><td align="right">3.51</td><td align="right">0.107</td></tr>
  <tr><td align="center">250</td><td align="right">54.4</td><td align="right">27.4</td><td align="right">77.0</td><td align="right">3.58</td><td align="right">0.107</td><td align="right">53.1</td><td align="right">27.5</td><td align="right">76.6</td><td align="right">3.52</td><td align="right">0.104</td></tr>
</tbody>
</table>

### Against the published numbers

<table>
<thead>
  <tr><th rowspan="2">method</th><th colspan="5">3DMatch</th><th colspan="5">3DLoMatch</th></tr>
  <tr><th>RR</th><th>IR</th><th>FMR</th><th>RRE</th><th>RTE</th><th>RR</th><th>IR</th><th>FMR</th><th>RRE</th><th>RTE</th></tr>
</thead>
<tbody>
  <tr><td><b>our OCFNet</b></td><td align="right"><b>89.8</b></td><td align="right"><b>60.9</b></td><td align="right"><b>97.3</b></td><td align="right"><b>2.18</b></td><td align="right"><b>0.071</b></td><td align="right"><b>55.4</b></td><td align="right"><b>27.5</b></td><td align="right"><b>77.2</b></td><td align="right"><b>3.70</b></td><td align="right"><b>0.110</b></td></tr>
  <tr><td>our improved CoFiNet</td><td align="right">89.3</td><td align="right">61.5</td><td align="right">96.3</td><td align="right">2.29</td><td align="right">0.073</td><td align="right">54.0</td><td align="right">27.6</td><td align="right">76.1</td><td align="right">3.59</td><td align="right">0.106</td></tr>
  <tr><td><i>CoFiNet (published)</i></td><td align="right"><i>88.4</i></td><td align="right"><i>51.9</i></td><td align="right"><i>98.1</i></td><td align="right"><i>--</i></td><td align="right"><i>--</i></td><td align="right"><i>64.2</i></td><td align="right"><i>26.7</i></td><td align="right"><i>83.3</i></td><td align="right"><i>--</i></td><td align="right"><i>--</i></td></tr>
  <tr><td><i>OCFNet (published)</i></td><td align="right"><i>90.2</i></td><td align="right"><i>58.7</i></td><td align="right"><i>98.5</i></td><td align="right"><i>--</i></td><td align="right"><i>--</i></td><td align="right"><i>66.7</i></td><td align="right"><i>29.5</i></td><td align="right"><i>84.0</i></td><td align="right"><i>--</i></td><td align="right"><i>--</i></td></tr>
  <tr><td><i>Predator (published)</i></td><td align="right"><i>90.6</i></td><td align="right"><i>57.1</i></td><td align="right"><i>96.5</i></td><td align="right"><i>--</i></td><td align="right"><i>--</i></td><td align="right"><i>62.4</i></td><td align="right"><i>28.3</i></td><td align="right"><i>76.3</i></td><td align="right"><i>--</i></td><td align="right"><i>--</i></td></tr>
</tbody>
</table>

<sub>Published rows are the 1000-sample column of each paper's Table 1, so the measured rows
are quoted at 1000 samples too. Both measured models exceed the published CoFiNet inlier
ratio by about 9 points on 3DMatch and reach it on 3DLoMatch. 3DLoMatch registration recall
is 9 to 11 points below CoFiNet's despite the higher inlier ratio; that gap is not
understood -- see the note below.</sub>

### Reading the sampling columns

The columns are flat because the model does not produce enough correspondences for the
sample count to bind: the readout emits about 730 per pair on 3DMatch and 420 on 3DLoMatch,
so every column from 5000 down to 500 draws from essentially the same set and IR is
identical to three decimals. The spread in RR (89.4 to 90.8 on 3DMatch) is RANSAC seed
noise, not a sampling trend, and should not be read as one. A genuine sampling curve would
need a larger correspondence pool, which means relaxing the `mutual` and `deduplicate`
readout -- measured at 6228 correspondences per pair, that lowers 3DLoMatch RR from 55.7 to
54.8, so the pool is small by choice.

### The overlap score

The overlap head earns its place as an auxiliary loss, not as a transport prior. Using the
predicted score to drive the Sinkhorn marginals was measured in five configurations against
a control that trains the same head and both overlap losses but keeps uniform marginals:

| overlap marginals | dustbin | 3DMatch IR | 3DLoMatch IR |
| --- | --- | --- | --- |
| none (control) | on | 51.3 | 22.0 |
| coarse and fine | on | 54.8 | 24.1 |
| fine only | on at coarse | 53.5 | 23.2 |
| coarse and fine | off | 53.0 | 21.5 |
| coarse and fine, plus a cost-side bias | on | 55.2 | 24.0 |

Guiding both transports is worth about +3.5 IR, and only while the dustbin is kept: the
score says *which* points should match, the dustbin lets a point match *nothing*, and
removing it forces every point onto some target and manufactures outliers. The damage is
three times larger on 3DLoMatch, where more of each cloud genuinely has no counterpart.

That gain does not survive RoPE-3D. With RoPE at the coarse level, every model without
overlap marginals beats every model with them (60.9 against 58.5 IR on 3DMatch), so the two
appear to supply the same geometric information and RoPE supplies more of it. The models
above therefore keep the overlap head and its losses but leave the marginals uniform.

### The 3DLoMatch recall gap

Our 3DLoMatch RR is 55.4 against CoFiNet's published 64.2 while our inlier ratio is higher
(27.5 against 26.7). Three explanations were tested and all three are refuted:

- **Too few correspondences.** Raising `min_coarse_matches` from 128 to 1024 quadruples the
  pool and RR falls monotonically, 53.2 to 47.3.
- **Too aggressive a readout.** Turning off `mutual` and `deduplicate` gives 13 times more
  inliers, 116 to 1557 per pair, and RR falls from 55.7 to 54.8.
- **Too little RANSAC.** `RANSACConvergenceCriteria(50000, 0.999)` stops after
  `log(1-c)/log(1-w^n)` draws, 27 iterations at IR 0.61, where CoFiNet's older open3d read
  that argument as `max_validation` and ran 50000. Forcing the full search improves the pose
  (3DMatch RRE 2.37 to 2.02, 3DLoMatch 3.57 to 2.99) and *lowers* RR on both benchmarks
  (90.5 to 89.9 and 55.3 to 50.8) at 17 times the runtime. RANSAC maximises inliers under
  its own 5 cm threshold, which is not the recall criterion, so more search finds
  better-supported but less correct alignments. `0.999` is kept: it is both faster and better.

The remaining test is to run CoFiNet's released checkpoint through this evaluation code. If
it scores near 64 the difference is in the model; if it scores near 54 the difference is in
the harness.

## Reproducing

```shell
# both benchmarks, sweeping 5000/2500/1000/500/250 sampled correspondences
BENCH=3DMatch   sbatch scripts/slurm_eval_sweep.sh configs/test/ocfnet_uniform_ovlloss_rope.yaml
BENCH=3DLoMatch sbatch scripts/slurm_eval_sweep.sh configs/test/ocfnet_uniform_ovlloss_rope.yaml

python scripts/collect_results.py snapshot/ocfnet_uniform_ovlloss_rope_test --markdown
```

`sinkhorn_iters` in the test config must match the value the model was trained with, and
`batch_size` must be 1: the benchmark writes one transform per pair, so a batched loader
produces `ceil(pairs/B)` estimates and the write walks off the end of the array.

The evaluation script appends IR and FMR to `est_traj/<benchmark>/<samples>/result` rather
than printing them, so read that file rather than the job's stdout.

## Tests

```shell
python scripts/test_spconv_forward.py configs/train/indoor.yaml    # shapes, row order, gradients, batch of 2
python scripts/test_batch_equivalence.py configs/train/indoor.yaml --batch_size 3
python scripts/test_train_step.py configs/train/indoor.yaml --iters 3 --batch_size 4
```

All pass on an A100 (`reg3d`: torch 2.7.1+cu126 / spconv-cu126 2.3.8, and also with torch
2.8+cu126): 25k source voxels forward in 0.72 s cold at 0.85 GiB, ~0.6 s per training
iteration with batch_size 4 (3.9 GiB), batched output within 3e-4 of the per-pair output.
RANSAC pose estimation recovers a known rigid transform to 3e-8 with and without the mutual
check.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{mei2022overlap,
  title     = {Overlap-guided Coarse-to-fine Correspondence Prediction for Point Cloud Registration},
  author    = {Mei, Guofeng and Huang, Xiaoshui and Zhang, Jian and Wu, Qiang},
  booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
  year      = {2022}
}
```

Questions, issues and pull requests are welcome. If you build on the spconv port or the
batched overlap-attention module rather than the method itself, a citation of the original
[Predator](https://github.com/prs-eth/OverlapPredator) and
[CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences) papers is appropriate
too -- this repository owes a great deal to both.

## Acknowledgements

The code base is the official [Predator](https://github.com/prs-eth/OverlapPredator)
(sparse-convolution variant) implementation by Shengyu Huang, Zan Gojcic et al.
