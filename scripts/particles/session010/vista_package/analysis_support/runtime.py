"""Portable atomic JSON only; no remote roots, retention or submissions."""
import json,os
from pathlib import Path
def atomic(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp."+str(os.getpid()))
 with tmp.open("w") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
