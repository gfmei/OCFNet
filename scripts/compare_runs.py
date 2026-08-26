"""Compare per-epoch benchmark lines across runs, matched by epoch.

Early-epoch ordering has reversed before in this project, so the comparison is only
meaningful at equal epochs and only once several agree in sign.
"""
import re, sys, pathlib

def read(exp):
    out = {}
    log = pathlib.Path('snapshot') / exp / 'log'
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        m = re.match(r'registration Epoch: (\d+).*registered: ([\d.]+).*coarse_IR: ([\d.]+).*fine_IR: ([\d.]+)', line)
        if m:
            out[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return out

runs = sys.argv[1:] or ['ocfnet_indoor', 'ocfnet_pairoverlap', 'ocfnet_gtoverlap']
data = {r: read(r) for r in runs}
# intersect over every run, including empty ones: a run with no epoch yet means
# there is nothing to compare, rather than a comparison among the others
common = sorted(set.intersection(*[set(d) for d in data.values()])) if data else []
if not common:
    print('no epoch is common to all runs yet:',
          {r: (max(d) if d else 0) for r, d in data.items()})
    sys.exit(0)
print(f'{"epoch":>6s} ' + ' '.join(f'{r[:22]:>24s}' for r in runs))
print(f'{"":>6s} ' + ' '.join(f'{"reg / coarse / fine":>24s}' for _ in runs))
for e in common[-12:]:
    cells = []
    for r in runs:
        reg, c, f = data[r][e]
        cells.append(f'{reg:.3f} / {c:.3f} / {f:.3f}'.rjust(24))
    print(f'{e:6d} ' + ' '.join(cells))
