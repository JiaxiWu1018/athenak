#!/usr/bin/env python3
"""Use the complete twelve-node group witnessed by Session009's passed48-rank MPI test."""
import argparse,collections,json,os,re,subprocess,time
from pathlib import Path
from runtime import ROOT,atomic,digest
NODE_SPEC='k002-005,k003-[003-007],k005-[002-006,009]'
WITNESS=ROOT/'history/r50_before_reset_20261006/evidence/mpi_48rank_gate.txt'
def witness_hosts(text):
 rows=re.findall(r'rank=(\d+)/(\d+) host=(\S+) alltoall_errors=(\d+)',text)
 if len(rows)!=48 or {int(r[0]) for r in rows}!=set(range(48)) or any(int(r[1])!=48 or int(r[3])!=0 for r in rows):raise ValueError('Witness must contain all48 distinct successful ranks')
 counts=collections.Counter(r[2].split('.')[0] for r in rows)
 if len(counts)!=12 or set(counts.values())!={4}:raise ValueError('Witness must contain12nodes with4ranks each')
 return sorted(counts)
def verify_hosts(hosts,text):
 expected=witness_hosts(text)
 if len(hosts)!=12 or len(set(hosts))!=12 or set(hosts)!=set(expected):raise ValueError('Allocation must match the complete previously passed twelve-node group')
 return expected
def main():
 p=argparse.ArgumentParser();p.add_argument('--witness-only',action='store_true');a=p.parse_args()
 config=json.loads((ROOT/'control/config.json').read_text());text=WITNESS.read_text();expected=witness_hosts(text)
 if digest(WITNESS)!=config['mpi_witness_sha256'] or expected!=config['validated_mpi_nodes']:raise ValueError('Pinned MPI witness changed')
 receipt=dict(utc=time.time(),job=os.environ.get('SLURM_JOB_ID'),witness=str(WITNESS),witness_sha256=digest(WITNESS),expected=expected)
 if a.witness_only:receipt['witness_only']=True
 else:
  if int(os.environ['SLURM_NNODES'])!=12 or int(os.environ['SLURM_NTASKS'])!=48:raise ValueError('Allocation size changed')
  hosts=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
  verify_hosts(hosts,text);receipt.update(actual=hosts,passed=True)
  atomic(ROOT/'evidence'/('mpi_node_selection_'+os.environ['SLURM_JOB_ID']+'.json'),receipt)
 print(json.dumps(receipt))
if __name__=='__main__':main()
