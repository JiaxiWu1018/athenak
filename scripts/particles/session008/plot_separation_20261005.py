#!/usr/bin/env python3
"""Coordinate separation from saved centroids, trackers and accepted AH centers.

Only small diagnostic tables are read. Run through Slurm. No resampling across
diagnostic gaps or interpolation between left/right AH acceptance times.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(path):
    return np.loadtxt(path, delimiter=',', ndmin=2)


def pairs(left, right):
    def keyed(path):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        result = {}
        for row in rows:
            key = row['run'], int(row['cycle'])
            if key in result:
                raise ValueError('duplicate accepted horizon candidate ' + str(key))
            result[key] = row
        return result
    lrows, rrows = keyed(left), keyed(right)
    records = []
    for key in lrows.keys() | rrows.keys():
        lrow, rrow = lrows.get(key), rrows.get(key)
        tm = float((lrow or rrow)['time'])
        valid = bool(lrow and rrow)
        delta = np.full(3, np.nan)
        if valid:
            if abs(float(lrow['time']) - float(rrow['time'])) > 1e-12:
                raise ValueError('same-cycle horizon times differ')
            delta = np.array([float(rrow[k]) - float(lrow[k])
                              for k in ('center_x', 'center_y', 'center_z')])
            if not np.isfinite(delta).all():
                raise ValueError('nonfinite accepted horizon center')
        records.append(dict(run=key[0], cycle=key[1], time=tm, valid=valid,
                            separation=float(np.linalg.norm(delta)), delta=delta))
    records.sort(key=lambda row: row['time'])
    # Preserve one-sided acceptance as missing data; don't substitute stale centers.
    for i in range(1, len(records)):
        if records[i]['time'] <= records[i-1]['time']:
            raise ValueError('nonincreasing accepted horizon times')
    return records


def statistics(time, distance):
    good = np.isfinite(distance)
    time, distance = time[good], distance[good]
    return dict(samples=len(distance), first_time=float(time[0]), last_time=float(time[-1]),
                first_separation=float(distance[0]), last_separation=float(distance[-1]),
                minimum=float(distance.min()), maximum=float(distance.max()),
                endpoint_change_percent=float(100 * (distance[-1] / distance[0] - 1)),
                range_over_mean_percent=float(100 * np.ptp(distance) / distance.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--horizons', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [args.tables / 'particle_components.csv', args.tables / 'orbit.csv',
              args.horizons / 'accepted_horizon_0.csv', args.horizons / 'accepted_horizon_1.csv']
    ledger = table(inputs[0])
    left, right = ledger[ledger[:, 1] == 1], ledger[ledger[:, 1] == 2]
    if not np.array_equal(left[:, 0], right[:, 0]):
        raise ValueError('particle component snapshot times differ')
    if not np.all(left[:, 2:4] > 0) or not np.all(right[:, 2:4] > 0):
        raise ValueError('empty clump in centroid table')
    centroids = np.linalg.norm(right[:, 4:7] - left[:, 4:7], axis=1)
    if not np.isfinite(centroids).all():
        raise ValueError('nonfinite clump centroid')
    tracker = table(inputs[1])
    accepted = pairs(inputs[2], inputs[3])
    ah_time = np.array([row['time'] for row in accepted])
    ah_distance = np.array([row['separation'] for row in accepted])
    if not np.isfinite(ah_distance).any():
        raise ValueError('no simultaneous accepted individual horizons')
    # Break the line for failed acceptance, missing cycles or restart boundaries.
    draw_time, draw_distance = [], []
    for i, row in enumerate(accepted):
        if i and (row['run'] != accepted[i-1]['run'] or row['cycle'] > accepted[i-1]['cycle'] + 1):
            draw_time.append(np.nan); draw_distance.append(np.nan)
        draw_time.append(row['time']); draw_distance.append(row['separation'])
    ah_stats = statistics(ah_time, ah_distance)
    summary = dict(formula='sqrt((x_R-x_L)^2+(y_R-y_L)^2+(z_R-z_L)^2)',
                   units='coordinate distance / M_ref; time / M_ref; inherited M_ref=1 source unit, not measured ADM mass',
                   centroid_definition='rest-mass-weighted center of ALL surviving tagged component particles; full simulation counts used, no rendering subsampling',
                   horizon_definition='accepted AH finder coordinate centers (surface expansion origins); same run/cycle/time, unique matched strict acceptance tables',
                   tracker_definition='tagged-core-seeded local lapse minimum before BH tracking, motion-predicted local lapse minimum afterward; not full-clump centroid',
                   gap_rule='no stale or rejected horizons; no left/right time interpolation; one-sided acceptance and restart/missing-cycle gaps remain visible',
                   particle_centroids=statistics(left[:, 0], centroids), accepted_horizons=ah_stats,
                   unpaired_horizon_rows=sum(not row['valid'] for row in accepted),
                   tracker=statistics(tracker[:, 0], tracker[:, 1]),
                   inputs={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                   limitations='Coordinate distance, not proper distance or a gauge-invariant eccentricity. Small post-BH arc cannot establish circularity. Surviving-particle centroids change under lapse removal and need not equal BH centers.')
    np.savetxt(args.output / 'clump_centroid_separation.csv',
               np.c_[left[:, 0], centroids, left[:, 2], right[:, 2]], delimiter=',',
               header='time,coordinate_separation,left_surviving_particle_count,right_surviving_particle_count')
    with (args.output / 'accepted_ah_separation.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'cycle', 'time', 'both_accepted', 'coordinate_separation', 'delta_x', 'delta_y', 'delta_z'])
        for row in accepted:
            writer.writerow([row['run'], row['cycle'], row['time'], int(row['valid']), row['separation'], *row['delta']])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(left[:, 0], centroids, 'o-', ms=3, lw=1.5, color='tab:blue', label='Surviving-particle clump centers')
    axes[0].plot(tracker[:, 0], tracker[:, 1], lw=1, color='0.5', ls='--', label='Live tracker centers')
    axes[0].plot(draw_time, draw_distance, color='tab:orange', lw=1.5, label='Both accepted AH centers')
    axes[0].axhline(6, color='0.7', ls=':', label='Initial model separation = 6')
    axes[0].set(xlabel='t / M_ref', ylabel='Coordinate center separation / M_ref', title='Full available evolution', xlim=(0, 12))
    axes[0].legend(fontsize=8, loc='best')
    axes[1].plot(draw_time, draw_distance, color='tab:orange', lw=1.2)
    axes[1].set(xlabel='t / M_ref', ylabel='Accepted AH-center separation / M_ref',
                title='After both individual horizons are accepted',
                xlim=(ah_stats['first_time'], 12))
    axes[1].text(.04, .95, 'Observed range / mean: %.2f%%\nOnly ~6 degrees of post-formation motion' % ah_stats['range_over_mean_percent'],
                 transform=axes[1].transAxes, va='top', fontsize=9)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle('Session 008: separation versus time', fontsize=14)
    fig.text(.5, .015, 'Coordinate distance; AH gaps are preserved. Remaining-particle centers can shift as particles are removed.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    fig.savefig(args.output / 'separation_vs_time.png', dpi=180)
    fig.savefig(args.output / 'separation_vs_time.pdf')
    plt.close(fig)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('particle_centroids', 'accepted_horizons', 'unpaired_horizon_rows', 'tracker')}, indent=2))


if __name__ == '__main__':
    main()
