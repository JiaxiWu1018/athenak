#!/usr/bin/env python3
"""Explicit conditional push for the next chat; never invoked by preparation."""
import subprocess
from prepare import CODE,SESSION,sha,atomic
from workflow import manifest, validated, git, BRANCH

def main():
 current=manifest();receipt=validated(current)
 if git('branch','--show-current')!=BRANCH:raise RuntimeError('Wrong campaign branch')
 if git('status','--porcelain'):raise RuntimeError('Commit and review all work before conditional push')
 head=git('rev-parse','HEAD');branch='refs/heads/project/GI-in-cluster'
 remote=git('ls-remote','origin',branch).split()[0]
 subprocess.run(['git','merge-base','--is-ancestor',remote,head],cwd=CODE,check=True)
 subprocess.run(['git','push','origin','HEAD:'+branch],cwd=CODE,check=True)
 if git('ls-remote','origin',branch).split()[0]!=head:raise RuntimeError('Remote push verification failed')
 atomic(SESSION/'evidence/PUSH_VERIFIED.json',dict(passed=True,commit=head,branch=branch,
  native_source_sha256=current['native_source_sha256'],manifest=current,
  validation_receipt_sha256=sha(SESSION/'evidence/VALIDATION_PASS.json'),validation_job=receipt['validation_job']))
 print('Verified tested fast-forward push:',head)

if __name__=='__main__':main()
