#!/usr/bin/env python3
"""Explicit conditional push for the next chat; never invoked by preparation."""
import json
from pathlib import Path
import subprocess
from prepare import CODE,SESSION,HERE,sha,source_hash,atomic
from launch import builds

def main():
 h=builds();receipt=json.loads((SESSION/'evidence/VALIDATION_PASS.json').read_text())
 if receipt.get('passed') is not True or receipt['native_source_sha256']!=h or receipt['production_input_sha256']!=sha(HERE/'inputs/production.athinput'):
  raise RuntimeError('Real matching GPU validation must pass before any push')
 def git(*args):return subprocess.check_output(['git',*args],cwd=CODE,text=True).strip()
 if git('status','--porcelain'):raise RuntimeError('Commit and review all work before conditional push')
 head=git('rev-parse','HEAD');branch='refs/heads/project/GI-in-cluster'
 remote=git('ls-remote','origin',branch).split()[0]
 subprocess.run(['git','merge-base','--is-ancestor',remote,head],cwd=CODE,check=True)
 subprocess.run(['git','push','origin','HEAD:'+branch],cwd=CODE,check=True)
 if git('ls-remote','origin',branch).split()[0]!=head:raise RuntimeError('Remote push verification failed')
 atomic(SESSION/'evidence/PUSH_VERIFIED.json',dict(passed=True,commit=head,branch=branch,
  native_source_sha256=h,validation_job=receipt['validation_job']))
 print('Verified tested fast-forward push:',head)

if __name__=='__main__':main()
