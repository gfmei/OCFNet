"""Per-pair diagnostics: why a pair with good correspondences still fails to register.

    python scripts/diagnose_pairs.py --source_path snapshot/<exp>/3DLoMatch \
        --n_points 1000 --out diag.csv

Inlier ratio counts correspondences within 10 cm of the truth; it says nothing about *where*
they are. A pose is only well determined when its correspondences span the cloud: an inlier
set squeezed into one patch pins translation along one direction and leaves rotation almost
free, so a pair can hold a high inlier ratio and still produce a transform that fails the
0.2 m recall threshold. This dumps, per pair, the inlier ratio next to the spatial extent of
the inlier set, so the two can be correlated against the pose error.

Columns: n_corr, inlier ratio, inlier count, the three standard deviations of the inlier
source points along their own principal axes (sigma1 >= sigma2 >= sigma3, metres), the same
for the whole source cloud, the resulting anisotropy, and the pose error against the truth.
"""
import argparse, glob, os, sys

import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config

cwd = os.getcwd()
sys.path.append(cwd)
from lib.benchmark_utils import ransac_pose_estimation_correspondences
from lib.utils import natural_key, setup_seed

setup_seed(0)


def principal_sigmas(points):
    """Standard deviation along each principal axis, largest first."""
    if points.shape[0] < 3:
        return np.zeros(3)
    centred = points - points.mean(axis=0, keepdims=True)
    # eigenvalues of the covariance are the squared extents along the principal axes
    eigenvalues = np.linalg.eigvalsh(np.cov(centred.T) + 1e-12 * np.eye(3))
    return np.sqrt(np.clip(eigenvalues, 0, None))[::-1]


def diagnose(each_file, n_points, seed, inlier_distance_threshold=0.1):
    rng = np.random.default_rng(seed)
    data = torch.load(each_file, weights_only=False)
    src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
    matches, scores = data['correspondences'], data['scores']
    n_all = matches.shape[0]
    if n_all > n_points:
        probability = (scores / scores.sum()).numpy()
        keep = rng.choice(n_all, size=n_points, replace=False, p=probability)
        matches, scores = matches[keep], scores[keep]
    if matches.shape[0] == 0:
        return None

    rot, trans = data['rot'].numpy(), data['trans'].numpy().reshape(3)
    src_points = src_pcd[matches[:, 0]].numpy()
    tgt_points = tgt_pcd[matches[:, 1]].numpy()
    residual = np.linalg.norm(src_points @ rot.T + trans - tgt_points, axis=1)
    inlier = residual < inlier_distance_threshold
    ratio = float(inlier.mean())

    sig_in = principal_sigmas(src_points[inlier])
    sig_all = principal_sigmas(src_pcd.numpy())

    estimate = ransac_pose_estimation_correspondences(src_pcd, tgt_pcd, matches)
    rot_est, trans_est = estimate[:3, :3], estimate[:3, 3]
    # geodesic angle between the estimated and the true rotation
    cos = np.clip((np.trace(rot_est.T @ rot) - 1) / 2, -1.0, 1.0)
    rre = float(np.degrees(np.arccos(cos)))
    rte = float(np.linalg.norm(trans_est - trans))
    return dict(n_corr=n_all, ir=ratio, n_inlier=int(inlier.sum()),
                sig1=sig_in[0], sig2=sig_in[1], sig3=sig_in[2],
                cloud_sig1=sig_all[0], cloud_sig3=sig_all[2],
                # how much of the cloud the inliers actually span, and how flat that set is
                span=float(sig_in[0] / max(sig_all[0], 1e-9)),
                flatness=float(sig_in[2] / max(sig_in[0], 1e-9)),
                rre=rre, rte=rte)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', required=True, type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--out', default='diag.csv', type=str)
    args = parser.parse_args()

    files = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)
    workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    with parallel_config(backend='loky', inner_max_num_threads=1):
        rows = Parallel(n_jobs=workers)(
            delayed(diagnose)(f, args.n_points, i) for i, f in enumerate(tqdm(files)))
    rows = [r for r in rows if r is not None]

    keys = list(rows[0])
    with open(args.out, 'w') as handle:
        handle.write(','.join(keys) + '\n')
        for r in rows:
            handle.write(','.join(f'{r[k]:.6g}' for k in keys) + '\n')

    # a pair counts as registered on the usual 15 deg / 30 cm criterion
    ok = np.array([r['rre'] < 15 and r['rte'] < 0.3 for r in rows])
    ir = np.array([r['ir'] for r in rows])
    matchable = ir > 0.05                       # the feature-match-recall threshold
    print(f'\npairs {len(rows)}, registered {ok.mean():.3f}, '
          f'matchable (IR>0.05) {matchable.mean():.3f}, '
          f'registered among matchable {ok[matchable].mean():.3f}')
    print(f'{"":<26}{"registered":>12}{"failed":>12}')
    for k in ('ir', 'n_inlier', 'sig1', 'sig3', 'span', 'flatness'):
        v = np.array([r[k] for r in rows])
        print(f'{k:<26}{np.median(v[matchable & ok]):>12.4f}'
              f'{np.median(v[matchable & ~ok]):>12.4f}')
