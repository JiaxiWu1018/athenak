"""Portable Session010 reporting; resource limits are explicit reviewed inputs."""
import json,subprocess,time
from pathlib import Path
from runtime import atomic

def write_reports(root,out,metrics,milestone):
 receipts=[]
 for p in root.rglob("result.json"):
  try:r=json.loads(p.read_text())
  except (ValueError,OSError):continue
  c=r.get("native_checkpoint")
  if r.get("checkpoint_complete") and c:receipts.append(c)
 latest=max(receipts,key=lambda c:c["cycle"]) if receipts else None
 statefile=root/"control/state.json"
 state=json.loads(statefile.read_text()) if statefile.exists() else {}
 rows=[]
 ids=[str(j["job_id"]) for j in state.get("jobs",[])]
 if ids:
  raw=subprocess.check_output(["sacct","-X","-n","-P","-j",",".join(ids),"--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocNodes"],text=True)
  for line in raw.splitlines():
   r=line.split("|")
   if len(r)>=5 and r[0] in ids and r[3].isdigit() and r[4].isdigit():rows.append(dict(job_id=r[0],state=r[1],exit_code=r[2],elapsed_seconds=int(r[3]),nodes=int(r[4]),node_hours=int(r[3])*int(r[4])/3600))
 limitsfile=root/"control/reviewed_limits.json"
 limits=json.loads(limitsfile.read_text()) if limitsfile.exists() else None
 record=dict(session="010",milestone=milestone,generated_utc=time.time(),latest_verified_checkpoint=latest,tracker_time=metrics.get("tracker_final_time"),actual_node_hours=sum(r["node_hours"] for r in rows),accounting=rows,accounting_complete=set(ids)=={r["job_id"] for r in rows},reviewed_limits=limits,future_cost_forecast=None,monetary_cost=None)
 atomic(out/"resources.json",record)
 text="# Jeans/GI Session010 — "+str(milestone)+"\n\nVerified checkpoint: "+str(latest)+"\n\nTracker coverage: "+str(record["tracker_time"])+". These are separate time coverages.\n\nDomain±512, root256³/dx4, initial4936blocks, central ceiling1/256, fine wave0.25 throughR46. Exactly5M initialparticles, localboost0.133215, inheritedapproximateKij0 initialdata and safeguards.\n\nActual allocated node-hours: "+str(record["actual_node_hours"])+". No stale-time extrapolation or invented monetary tariff. Reviewed limits: "+str(limits)+".\n\nAvailable scientific measurements:\n\n```json\n"+json.dumps(metrics,indent=2,allow_nan=False)+"\n```\n\nRawR40 multipoles are primary; coordinate motion/finite-radius strain are not convergence or radiation-driven inspiral proof. Boundary/gauge contamination amplitude is unmeasured.\n"
 (out/"REPORT_Jeans10.md").write_text(text)
 (out/"REPORT_AGENT.md").write_text(text+"\nResources and coverage receipt: resources.json. Finalization uses native checkpoint layout/hash; terminal resource/user stops remain terminal.\n")
 atomic(root/"analysis/latest.json",dict(path=str(out),milestone=milestone))
