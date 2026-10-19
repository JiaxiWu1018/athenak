"""Bounded, logged status-file reads; all original memory/staleness guards remain."""
import json
import math
from pathlib import Path
import time


class StableSamplerReader:
    def __init__(self, attempts=3, retry_seconds=.25):
        self.attempts = attempts
        self.retry_seconds = retry_seconds
        self.paths = {}

    def __call__(self, telemetry, nodes, cfg):
        telemetry = Path(telemetry)
        key = str(telemetry.resolve())
        error = None
        for attempt in range(1, self.attempts + 1):
            paths = []
            try:
                # Discover once at startup; concurrent atomic replacements must
                # not make a later directory listing appear to lose a node.
                paths = self.paths.get(key) or sorted(telemetry.glob('*.json'))
                if len(paths) != nodes:
                    raise RuntimeError(f'expected {nodes} status files, found {len(paths)}')
                rows = [json.loads(p.read_text()) for p in paths]
                now = time.time()
                for path, row in zip(paths, rows):
                    if path.name != row['host'] + '.json':
                        raise ValueError('status filename/hostname mismatch')
                    age = now - float(row['epoch'])
                    if not math.isfinite(age) or age > cfg['telemetry_stale_seconds'] or age < -30:
                        raise RuntimeError(f'missing/stale status from {row["host"]}; age={age}')
                    # An explicit sampler error is returned immediately to the
                    # original safety logic, not hidden by retries.
                    if row.get('error'):
                        continue
                    if not (0 < row['host_total_bytes'] and 0 <= row['host_used_bytes'] <= row['host_total_bytes']):
                        raise ValueError('invalid host memory measurement')
                    if len(row['gpus']) != 1:
                        raise ValueError('expected one GPU measurement per node')
                    for g in row['gpus']:
                        if not all(math.isfinite(float(g[k])) for k in ('used_mib', 'total_mib')) or not 0 <= g['used_mib'] <= g['total_mib'] or g['total_mib'] <= 0:
                            raise ValueError('invalid GPU memory measurement')
                if len({r['host'] for r in rows}) != nodes:
                    raise ValueError('duplicate sampler host')
                self.paths[key] = tuple(paths)
                if attempt > 1:
                    self.log(telemetry, dict(event='read_recovered', attempt=attempt))
                return rows
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                error = exc
                self.log(telemetry, dict(event='read_error', attempt=attempt,
                         error_type=type(exc).__name__, message=str(exc),
                         paths_seen=[p.name for p in paths], expected_nodes=nodes))
                if attempt < self.attempts:
                    time.sleep(self.retry_seconds)
        raise RuntimeError(f'status-file read failed after {self.attempts} attempts: {error}') from error

    @staticmethod
    def log(telemetry, event):
        # A failure to retain the diagnostic is itself an I/O error; do not hide it.
        event['epoch'] = time.time()
        with (telemetry.parent / 'telemetry_read_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')

