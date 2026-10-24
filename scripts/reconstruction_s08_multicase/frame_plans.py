"""Initialize source-bound frame plans before parallel chunk workers start."""
import matplotlib.pyplot as plt
from common import *
from movie_chunks import prepare


def main():
    job()
    assembly = load(ROOT/'evidence/ASSEMBLY_OK.json')
    assert assembly['source_files_SHA256_verified']
    modes = [r['case'] for r in assembly['sources']]
    if set(modes) == {c['id'] for c in ALL_CASES}:
        modes.append('comparison')
    plans = []
    for mode in modes:
        name, fig, states, times, directory, fingerprint, expected = prepare(mode)
        plt.close(fig)
        plans.append(dict(mode=mode, name=name, frames=len(times), frames_directory=str(directory), fingerprint=fingerprint))
    atomic(ROOT/'evidence/FRAME_PLANS_READY.json', dict(utc=now(), job=job(), source_assembly=receipt(ROOT/'evidence/ASSEMBLY_OK.json'), plans=plans,
        scope='Serial initialization prevents concurrent creation of a shared plan. This is frame planning, not rendering, decoding or actual visual acceptance.'))
    print(json.dumps(plans, indent=2))


if __name__ == '__main__':
    main()
