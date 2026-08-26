"""Collect RR, FMR and IR from every est_traj/<benchmark>/<samples>/result file.

Registration recall is the pair-count weighted mean of the per-scene column, the way the
benchmark defines it; inlier ratio and feature match recall are appended to the same file by
the evaluation script and are already scene-averaged. `w_mutual` applies the mutual-nearest
-neighbour check to the correspondences before scoring, which is what the papers report.
"""
import argparse, datetime, pathlib, re, sys

def read(result):
    text = result.read_text()
    num = den = 0.0
    for line in text.splitlines():
        p = [x.strip() for x in line.split('¦')]
        if len(p) < 6 or not p[1] or p[1] == 'prec.':
            continue
        try:
            rec, n = float(p[2]), int(p[5])
        except ValueError:
            continue
        num += rec * n
        den += n
    grab = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, text))
    return {
        'RR': num / den if den else None,
        # evaluate_predator.py writes 'Inlier ratio w_mutual: x' (it reports both with and
        # without the mutual check); evaluate_ocfnet.py writes plain 'Inlier ratio: x'
        'IR': grab(r'Inlier ratio(?: w_mutual)?: ([\d.]+)'),
        'FMR': grab(r'Feature match recall(?: w_mutual)?: ([\d.]+)'),
        'RRE': grab(r'Mean median RRE: ([\d.]+)'),
        'RTE': grab(r'Mean median RTE: ([\d.]+)'),
    }

ap = argparse.ArgumentParser()
ap.add_argument('roots', nargs='+', help='snapshot/<exp>_test directories')
ap.add_argument('--label', nargs='*', default=None, help='row label per root, in order')
ap.add_argument('--since', default=None,
                help='ignore result files older than this (YYYY-MM-DDTHH:MM). An evaluation '
                     'of an earlier checkpoint leaves files behind, and a sweep that has not '
                     'reached every sample count would otherwise report them as current.')
ap.add_argument('--markdown', action='store_true',
                help='emit the README tables: one block per metric, methods as rows and '
                     'sample counts as columns, the layout of Table 1 of the paper')
args = ap.parse_args()
cutoff = datetime.datetime.fromisoformat(args.since) if args.since else None

def fresh(f):
    return cutoff is None or datetime.datetime.fromtimestamp(f.stat().st_mtime) >= cutoff

SAMPLES = [5000, 2500, 1000, 500, 250]

if args.markdown:
    labels = args.label or [pathlib.Path(r).name for r in args.roots]
    data = {}
    for root, label in zip(args.roots, labels):
        for bench in ('3DMatch', '3DLoMatch'):
            for n in SAMPLES:
                f = pathlib.Path(root) / 'est_traj' / bench / str(n) / 'result'
                if f.exists() and fresh(f):
                    data[(label, bench, n)] = read(f)
    for metric, scale in (('RR', 100), ('FMR', 100), ('IR', 100)):
        print(f'{metric} (%):\n')
        head = '| method | ' + ' | '.join(f'3DMatch {n}' if i == 0 else str(n)
                                          for i, n in enumerate(SAMPLES))
        head += ' | ' + ' | '.join(f'3DLoMatch {n}' if i == 0 else str(n)
                                   for i, n in enumerate(SAMPLES)) + ' |'
        print(head)
        print('| :-- |' + ' --: |' * (2 * len(SAMPLES)))
        for label in labels:
            cells = []
            for bench in ('3DMatch', '3DLoMatch'):
                for n in SAMPLES:
                    v = data.get((label, bench, n), {}).get(metric)
                    cells.append(f'{v * scale:.1f}' if v is not None else '')
            print(f'| {label} | ' + ' | '.join(cells) + ' |')
        print()
    sys.exit(0)

for root in args.roots:
    est = pathlib.Path(root) / 'est_traj'
    if not est.exists():
        print(f'{root}: no est_traj'); continue
    print(f'\n=== {root}')
    for bench in ('3DMatch', '3DLoMatch'):
        d = est / bench
        if not d.exists():
            continue
        rows = []
        for s in sorted(d.iterdir(), key=lambda p: -int(p.name) if p.name.isdigit() else 0):
            f = s / 'result'
            if f.exists() and fresh(f):
                rows.append((s.name, read(f)))
        if not rows:
            continue
        print(f'  {bench}')
        print(f'    {"samples":>8s} {"RR":>7s} {"FMR":>7s} {"IR":>7s} {"RRE":>7s} {"RTE":>7s}')
        for n, m in rows:
            fmt = lambda v: f'{v:7.3f}' if v is not None else f'{"-":>7s}'
            print(f'    {n:>8s} {fmt(m["RR"])}{fmt(m["FMR"])}{fmt(m["IR"])}{fmt(m["RRE"])}{fmt(m["RTE"])}')
