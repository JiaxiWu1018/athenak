#!/usr/bin/env python3
"""Derive operational Session 011 input with an auditable parameter allow-list."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def parse(text):
    blocks = {}
    name = None
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        if re.fullmatch(r'<[^>]+>', line):
            name = line[1:-1]
            blocks.setdefault(name, {})
        elif '=' in line and name:
            key, value = line.split('=', 1)
            blocks[name][key.strip()] = value.strip()
    return blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline', type=Path)
    ap.add_argument('destination', type=Path)
    args = ap.parse_args()
    original = args.baseline.read_text()
    old = parse(original)
    new = {k: dict(v) for k, v in old.items() if not k.startswith('output') and k != 'comment'}
    changes = {
        ('job', 'basename'): 's11_amd',
        ('time', 'tlim'): '75.0',
        ('time', 'nlim'): '-1',
        ('time', 'stop_file'): 'REQUEST_STOP',
        ('time', 'stop_check_every'): '1',
        ('mesh_refinement', 'max_nmb_per_rank'): '256',
        ('z4c', 'nrad_wave_extraction'): '3',
        ('z4c', 'extraction_radius_1'): '36.0',
        ('z4c', 'extraction_radius_2'): '44.0',
        ('z4c', 'co_0_out_every'): '1',
        ('z4c', 'co_1_out_every'): '1',
        ('gi_diagnostics', 'enabled'): 'true',
    }
    for h in range(3):
        changes['fastflow', f'reuse_last_surface_shape_{h}'] = 'true'
        changes['fastflow', f'persist_surface_shape_{h}'] = 'true'
    for (block, key), value in changes.items():
        new.setdefault(block, {})[key] = value
    # Preserve every compact slice and history from the baseline. Replace full
    # particle/volume dumps with the fixed exact-ID cohort and milestone restarts.
    outputs = [v for k, v in old.items() if k.startswith('output') and
               (v.get('file_type') == 'hst' or any(k.startswith('slice_') for k in v)
                or (v.get('file_type')=='cbin' and v.get('coarsen_factor')=='32'))]
    outputs += [dict(file_type='trk', nparticles='90000', format_version='2',
                     cohort_per_component='30000', dt='0.25'),
                dict(file_type='rst', dt='1e30', last_time='-1')]
    for i, block in enumerate(outputs, 1):
        new[f'output{i}'] = block
    actual = {(b, k) for b in set(old) | set(new) if b != 'comment' and not b.startswith('output')
              for k in set(old.get(b, {})) | set(new.get(b, {}))
              if old.get(b, {}).get(k) != new.get(b, {}).get(k)}
    assert actual <= set(changes), actual-set(changes)
    text = '# Session 011: one trajectory, hard endpoint t=75; operational changes below.\n'
    text += '\n'.join(f'<{b}>\n'+'\n'.join(f'{k} = {v}' for k, v in vals.items())+'\n'
                      for b, vals in new.items())
    args.destination.write_text(text)
    receipt = dict(baseline=str(args.baseline),
                   baseline_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   changes=[dict(block=b, parameter=k, old=old.get(b, {}).get(k),
                                 new=new[b][k]) for b, k in sorted(actual)],
                   output_blocks=outputs)
    args.destination.with_suffix('.changes.json').write_text(json.dumps(receipt, indent=2)+'\n')

if __name__ == '__main__':
    main()
