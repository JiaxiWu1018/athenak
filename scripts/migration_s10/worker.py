"""Explicit allocation roles; scientific failures never release production."""
import socket
import traceback
from common import *

def main():
    job()
    out = EXT/'evidence/allocations'/job()
    role = os.environ['S10_ROLE']
    atomic(out/'STARTED.json', dict(utc=now(), job=job(), role=role, case=CASE,
        host=socket.gethostname(), source_commit=subprocess.check_output(
        ['git', '-C', str(EXT/'src/athenak'), 'rev-parse', 'HEAD'], text=True).strip()))
    try:
        if role == 'reference':
            from check_configuration import check
            check()
            from prepare import main as prepare
            prepare()
            from constraint_reference import main as reference
            reference()
        elif role == 'production':
            assert load(EXT/'evidence/PREPARATION_OK.json')['native_source_unchanged']
            assert load(EXT/'evidence/INDEPENDENT_CONSTRAINT_REFERENCE_OK.json')['results']
            from trajectory import main as trajectory
            trajectory()
        elif role == 'analysis':
            from analysis import main as analysis
            analysis()
        elif role == 'products':
            from products import main as products
            products()
        elif role == 'qa':
            from qa_workflow import main as qa
            qa()
        else:
            raise ValueError(role)
        atomic(out/'WORKER_OK.json', dict(utc=now(), job=job(), role=role, case=CASE))
    except BaseException:
        atomic(out/'FAILURE.json', dict(utc=now(), job=job(), role=role,
            traceback=traceback.format_exc()))
        raise

if __name__ == '__main__':
    main()
