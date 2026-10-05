#!/usr/bin/env python3
"""Join accepted horizon candidates to summaries by time AND surface geometry.

The summary's first column is the finder iteration, not the evolution cycle.
Run plotting in Slurm; only small preserved diagnostic tables are needed.
"""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


def accepted_measurements(consumers, summaries):
    times = [row[1] for row in summaries]
    result = []
    counts = dict(published=0, matched=0, unmatched=0, ambiguous=0)
    for candidate in consumers:
        flags = ('published_this_candidate', 'association_ok',
                 'quality_geometry_ok', 'quality_persist_ok')
        if any(candidate.get(k) != '1' for k in flags):
            continue
        counts['published'] += 1
        tm = float(candidate['time'])
        # FastFlow writes time with %g (six significant digits). The consumer
        # preserves full precision. The tolerance follows that printing contract.
        tolerance = .500001 * 10 ** (math.floor(math.log10(abs(tm))) - 5) if tm else 1e-12
        lo = bisect.bisect_left(times, tm - tolerance)
        hi = bisect.bisect_right(times, tm + tolerance)
        matches = []
        for row in summaries[lo:hi]:
            if len(row) != 21 or not all(math.isfinite(row[i]) for i in (2, 7, 11, 12, 13, 18, 19, 20)):
                continue
            if row[2] <= 0 or row[12] <= 0:
                continue
            geometry = ((7, 'area'), (11, 'rmin'), (18, 'center_x'),
                        (19, 'center_y'), (20, 'center_z'))
            if all(math.isclose(row[i], float(candidate[key]), rel_tol=1e-11,
                                abs_tol=1e-12) for i, key in geometry):
                matches.append(row)
        if len(matches) == 1:
            result.append((candidate, matches[0]))
            counts['matched'] += 1
        else:
            counts['ambiguous' if matches else 'unmatched'] += 1
    return result, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    summary = dict(join_contract='same run, rounded time, area, rmin and center; unique match; strict consumer publication/association/geometry/persistence',
                   summary_first_column='finder iteration, NOT evolution cycle',
                   units='M_ref=1 inherited source unit', objects={}, inputs={})
    for index in (0, 1, 2):
        combined = []
        counts = dict(published=0, matched=0, unmatched=0, ambiguous=0)
        for run in sorted(args.runs_root.glob('segment_*')):
            consumer = list(run.glob('*.horizon_consumer_' + str(index) + '.csv'))
            measurements = list(run.glob('*.horizon_summary_' + str(index) + '.txt'))
            if not consumer or not measurements:
                continue
            for path in consumer + measurements:
                summary['inputs'][str(path.relative_to(args.runs_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with consumer[0].open() as stream:
                candidates = list(csv.DictReader(stream))
            rows = sorted(([float(v) for v in line.split()] for line in measurements[0].read_text().splitlines()
                           if line.strip() and not line.startswith('#')), key=lambda row: row[1])
            matched, stats = accepted_measurements(candidates, rows)
            for key in counts:
                counts[key] += stats[key]
            combined.extend((run.name, c, row) for c, row in matched)
        combined.sort(key=lambda item: float(item[1]['time']))
        summary['objects'][str(index)] = counts
        if combined:
            with (args.output / ('accepted_horizon_' + str(index) + '.csv')).open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['run', 'cycle', 'time', 'summary_time', 'finder_iterations',
                                 'M_BH', 'M_irr', 'chi_BH', 'area', 'rmin', 'center_x', 'center_y', 'center_z'])
                for run, c, row in combined:
                    writer.writerow([run, c['cycle'], c['time'], row[1], row[0], row[2], row[12], row[13],
                                     row[7], row[11], row[18], row[19], row[20]])
            # Plot each run separately so restart/reacquisition gaps stay visible.
            for run in sorted({item[0] for item in combined}):
                part = [item for item in combined if item[0] == run]
                tm, mass, spin = [], [], []
                previous = None
                for _, c, row in part:
                    cycle = int(c['cycle'])
                    if previous is not None and cycle > previous + 1:
                        tm.append(float('nan')); mass.append(float('nan')); spin.append(float('nan'))
                    tm.append(float(c['time'])); mass.append(row[2]); spin.append(row[13])
                    previous = cycle
                axes[0].plot(tm, mass,
                             color=('tab:blue', 'tab:orange', 'tab:green')[index], label=str(index) if run == combined[0][0] else None)
                axes[1].plot(tm, spin,
                             color=('tab:blue', 'tab:orange', 'tab:green')[index])
            _, c, row = combined[-1]
            summary['objects'][str(index)]['last'] = dict(time=float(c['time']), M_BH=row[2], M_irr=row[12], chi_BH=row[13])
        if counts['unmatched'] or counts['ambiguous']:
            raise RuntimeError('Accepted candidate lacks a unique surface match: ' + str(index) + ': ' + str(counts))
    axes[0].set(xlabel='t/M_ref', ylabel='accepted M_BH/M_ref')
    axes[1].set(xlabel='t/M_ref', ylabel='accepted chi_BH (coordinate spin prescription)')
    axes[0].legend()
    fig.suptitle('Session 008: accepted individual horizons; restart segments separate')
    fig.tight_layout()
    fig.savefig(args.output / 'accepted_horizons.png', dpi=170)
    plt.close(fig)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['objects'], indent=2))


if __name__ == '__main__':
    main()
