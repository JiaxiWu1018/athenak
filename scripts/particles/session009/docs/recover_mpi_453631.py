#!/usr/bin/env python3
"""Reviewed pre-initialization MPI recovery; preserve all science and cumulative caps."""
import json,pathlib,shutil,subprocess,time,hashlib,sys
site=sys.argv[1];root=pathlib.Path('/work1/eliasmost/jiaxiwu/gi_s009_amd_20261005' if site=='amd' else '/data3/jiaxiwu/NRPIC/GI_in_cluster/session_009_orbit_gw_production_20261005')
h=root/'history/mpi_failure_453631_20261007';h.mkdir(parents=True)
if site=='amd':
 s=json.loads((root/'control/state.json').read_text());c=json.loads((root/'control/config.json').read_text())
 assert s['status']=='configuration_failure' and s['time']==0
 assert not any((root/'control'/n).exists() for n in ('USER_STOP','REQUEST_STOP','RESOURCE_STOP.json'))
 assert not {str(j['id']) for j in s['jobs']}.intersection(subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i'],text=True).split())
 for folder in ('control','scripts','inputs'):shutil.copytree(root/folder,h/folder)
 (h/'evidence').mkdir();(h/'logs').mkdir()
 for name in ('gate_provenance.txt','mpi_48rank_gate.txt'):
  shutil.copy2(root/'evidence'/name,h/'evidence'/name)
 for name in ('453631_gate.log','453632_inspect0.log'):shutil.copy2(root/'logs'/name,h/'logs'/name)
 for name in ('README.md','REPORT_AGENT.md','REPORT_Jeans9.md'):shutil.copy2(root/name,h/name)
 (h/'REVIEW.md').write_text('453631 failed after6seconds at MPI_Init; k003-010 choseob1 while k002-006 choseucx. No particles or geometry initialized. Same node fault seenSession008. Exclude k003-010 in both fullstartup andfutureproduction, retain pinnedstack/automaticPML/48ranks. Original source/input/executables andcumulativebudget/deadline unchanged. No numerical retry.\n')
else:
 ledger=json.loads((root/'evidence/active_archive_jobs.json').read_text());assert ledger['pending'] is None
 assert not {str(j['id']) for j in ledger['jobs']}.intersection(subprocess.check_output(['squeue','-h','-u','jiaxiwu','-o','%i'],text=True).split())
 shutil.copytree(root/'evidence',h/'evidence')
 for name in ('README.md','REPORT_AGENT.md','REPORT_Jeans9.md'):shutil.copy2(root/name,h/name)
 for name in ('ARCHIVE_ANALYSIS_COMPLETE.json','updates.json','INITIAL_SCIENCE_REPORT.json'):
  p=root/'evidence'/name
  if p.exists():p.rename(h/'evidence'/('active_'+name))
 p=root/'analysis/latest.json'
 if p.exists():p.rename(h/'analysis_latest.json')
 (h/'REVIEW.md').write_text('Anta2356 completed failure-status reports; not a completed physical run. Preserve analysis/update_final_2356 and cumulative4/96archivejobs, reactivate metadata trigger after auditedAMDstate reset.\n')
print(json.dumps(dict(preserved=site,history=str(h))))
