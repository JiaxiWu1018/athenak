#!/usr/bin/env python3
"""Non-science preparation audit; never executes Athena or scheduler actions."""
import io
import json
from pathlib import Path
import subprocess
import unittest
from prepare import HERE,CODE,SESSION,source_hash,sha,atomic

def main():
 for path in HERE.rglob('*.py'):compile(path.read_text(),str(path),'exec')
 for path in [HERE/'build.sh',HERE/'validate.sbatch',HERE/'production.sbatch']:
  subprocess.run(['bash','-n',str(path)],check=True)
 stream=io.StringIO()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE/'tests')))
 atomic(SESSION/'evidence/SOURCE_TESTS.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),output=stream.getvalue(),science_runs_performed=False))
 if not result.wasSuccessful():raise RuntimeError(stream.getvalue())
 h=source_hash();verified={}
 for kind in ['gi','os','spin']:
  p=SESSION/f'evidence/build/{kind}.json';r=json.loads(p.read_text())
  if r['native_source_sha256']!=h or sha(r['executable'])!=r['executable_sha256']:
   raise RuntimeError('Stale or inconsistent native build: '+kind)
  verified[kind]=r
 old=Path('/scratch/10222/jiaxiwu1018/jeans_s10_transfer.HpgyzK/session010_vista_handoff/runtime_s10/build/source/athenak')
 unchanged=[]
 for rel in ['src/z4c/fastflow.cpp','src/z4c/fastflow.hpp','src/particles/particles.cpp',
             'src/particles/particles.hpp','src/particles/particles_pushers.cpp','src/driver/driver.cpp',
             'src/z4c/z4c_wave_extr.cpp','src/CMakeLists.txt']:
  blob=subprocess.check_output(['git','show','386870f88f60e25e1a8fac76fe04a071b87a01cf:'+rel],cwd=CODE)
  import hashlib
  if sha(old/rel)!=hashlib.sha256(blob).hexdigest():raise RuntimeError('Session10 source changed: '+rel)
  unchanged.append(rel)
 d=dict(preparation_code_and_compile_passed=True,native_source_sha256=h,builds=verified,
  source_tests=result.testsRun,session10_source_unchanged=unchanged,
  production_input_sha256=sha(HERE/'inputs/production.athinput'),
  science_runs_performed=False,slurm_jobs_submitted=False,production_ready=False,
  pushed=False,remaining=['Native OS collapse/reference/restart and spin/angular tests',
                        'Inspect actual diagnostic outputs','Conditional fast-forward GitHub push',
                        'User-submitted single production job after readiness',
                        'Post-run scientific analysis and paper visual verification'])
 atomic(SESSION/'evidence/PREPARATION_FINAL.json',d)
 print(f'PASS: three native builds match final source; {result.testsRun} non-science tests pass. NO SCIENCE RUNS OR PUSH.')

if __name__=='__main__':main()
