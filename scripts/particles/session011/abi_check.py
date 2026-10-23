"""Inspect ARM machine code for the observed particle-layout mismatch; never run it."""
import json
import re
import subprocess
import sys

from prepare import sha


def inspect(exe):
    lines = subprocess.check_output(['nm', '-n', '-C', str(exe)], text=True).splitlines()
    symbols = {}
    for line in lines:
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        name = fields[2]
        if (name == 'particles::Particles::EnergyCalculation(Driver*, int)'
                or name.startswith('void particles::Particles::calc_prtcl_energy<')):
            symbols[name] = int(fields[0], 16)
    if len(symbols) != 4:
        raise RuntimeError('Missing expected particle energy symbols')
    offsets = {}
    snippets = {}
    for name, address in symbols.items():
        snippet = subprocess.check_output([
            'objdump', '-d', '-C', '--start-address='+str(address),
            '--stop-address='+str(address+96), str(exe)], text=True)
        match = re.search(r'\bldr\s+x\d+,\s*\[x0,\s*#(\d+)\]', snippet)
        if match is None:
            raise RuntimeError('Unrecognized pinned ARM function prologue: ' + name)
        offsets[name] = int(match[1])
        snippets[name] = snippet
    return dict(executable=str(exe), executable_sha256=sha(exe),
                particle_pack_offsets=offsets, consistent=len(set(offsets.values())) == 1,
                disassembly=snippets, simulation_tested=False)


if __name__ == '__main__':
    result = inspect(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['consistent'] else 1)
