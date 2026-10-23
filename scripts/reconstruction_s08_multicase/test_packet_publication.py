"""Tiny metadata regressions; no solver, GPU work or real raw-data validation."""
from contextlib import ExitStack
import tempfile
from unittest.mock import patch
from pathlib import Path
import trajectory
import review_packet
from common import atomic,case_config,load

def main():
    c=case_config('yc0500_u0010')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);work=root/'runs/yc0500';work.mkdir(parents=True)
        (root/'inputs').mkdir();(root/'inputs/yc0500.athinput').write_text('fixture')
        atomic(root/'evidence/samples/yc0500.json',{})
        atomic(root/'evidence/CASE_NUMERICAL_ACCEPTANCE.json',dict(checkpoint=dict(sha256='fixture'),numerical_checks_passed=True))
        atomic(work/'state.json',dict(complete=True,latest=dict(path='fixture.rst',sha256='fixture')))
        atomic(work/'FINAL_FIELDS_OK.json',dict(checkpoint=dict(sha256='fixture'),no_evolution=True))
        atomic(root/'reduced/yc0500/summary.json',dict(complete=True))
        partial=root/'review_packet/numerical_acceptance.json';partial.parent.mkdir()
        partial.write_bytes(b'interrupted old copy')
        with ExitStack() as stack:
            stack.enter_context(patch.object(review_packet,'ROOT',root))
            stack.enter_context(patch.object(review_packet,'job',return_value='metadata_test'))
            stack.enter_context(patch.object(review_packet,'storage',return_value=dict(own=0,limit=10*1024**3)))
            review_packet.export(c)
            retained=list((root/'interrupted').glob('review_packet_*/numerical_acceptance.json'))
            assert len(retained)==1 and retained[0].read_bytes()==b'interrupted old copy'
            marker=root/'review_packet/MANIFEST.json';original=marker.read_bytes()
            review_packet.export(c);assert marker.read_bytes()==original
        # A fully published successor returns before setup or any solver call,
        # preserving the packet's exact bytes and its original acceptance.
        with ExitStack() as stack:
            stack.enter_context(patch.object(trajectory,'ROOT',root))
            stack.enter_context(patch.object(trajectory,'CASE',c))
            stack.enter_context(patch.object(trajectory,'job',return_value='metadata_test'))
            stack.enter_context(patch.object(trajectory,'admission',return_value=None))
            stack.enter_context(patch.object(trajectory,'validate',return_value=dict(sha256='fixture',time=200,dt=.025,ah_onset=-1)))
            stack.enter_context(patch.object(trajectory,'setup',side_effect=AssertionError('Published endpoint must skip setup/evolution')))
            for _ in range(2):trajectory.main()
            assert marker.read_bytes()==original
            assert load(root/'evidence/jobs/metadata_test/CASE_RESULT.json')['no_native_evolution_or_republication']
            # Source-file corruption is detected; no new publication masks it.
            (root/'review_packet/exact.athinput').write_text('corrupt')
            try:trajectory.main()
            except AssertionError:pass
            else:raise AssertionError('Corrupted published packet was accepted')
            assert marker.read_bytes()==original
    print('PASS: interrupted copies retained; published retries immutable; no native setup/evolution; corruption rejected. Metadata fixtures only; numerical payload validation mocked.')

if __name__=='__main__':main()
