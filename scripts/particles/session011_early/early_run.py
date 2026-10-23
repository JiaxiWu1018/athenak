#!/usr/bin/env python3
"""Reuse the unchanged production runner with the explicit user-approved launch gate."""
import sys
from early_gate import LEGACY, manifest, readiness

sys.path.insert(0, str(LEGACY))
import launch


def main():
    if sys.argv[1:] != ['production']:
        raise RuntimeError('This entrypoint supports production only')
    # Dependency injection is local to this new process. The original files and
    # validation process are unchanged. Every native startup/runtime/output/final
    # guard remains in launch.production(); only its launch policy is replaced.
    launch.workflow_ready = readiness
    launch.manifest = manifest
    print('USER-AUTHORIZED UNVALIDATED PRODUCTION: GPU validation and pre-launch push are waived.', flush=True)
    launch.main()


if __name__ == '__main__':
    main()
