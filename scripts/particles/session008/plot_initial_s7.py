#!/usr/bin/env python3
"""Small deterministic Session 007 initialization check plot (not a science movie)."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def block(data: bytes, marker: bytes, count: int):
    start=data.find(marker)
    if start < 0: raise ValueError(f"missing {marker!r}")
    start=data.find(b"\n",start+len(marker))+1
    return np.frombuffer(data,dtype=">f4",count=count,offset=start)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("vtk",type=Path)
    ap.add_argument("output",type=Path); args=ap.parse_args()
    data=args.vtk.read_bytes(); points=re.search(rb"POINTS\s+(\d+)\s+float",data)
    if points is None: raise ValueError("missing particle count")
    n=int(points.group(1)); xyz=block(data,points.group(0),3*n).reshape(n,3)
    mom=block(data,b"VECTORS prtcl_vel float",3*n).reshape(n,3)
    tag=block(data,b"SCALARS ptag float\nLOOKUP_TABLE default",n).astype(np.int64)
    groups=(("envelope",tag<3_000_000,"0.5"),
            ("left",(tag>=3_000_000)&(tag<4_000_000),"tab:blue"),
            ("right",tag>=4_000_000,"tab:orange"))
    fig,ax=plt.subplots(figsize=(8,8),constrained_layout=True)
    for label,mask,color in groups:
        idx=np.flatnonzero(mask)[::max(1,int(np.count_nonzero(mask))//25000)]
        ax.scatter(xyz[idx,0],xyz[idx,1],s=0.25,c=color,alpha=.35,label=label,
                   rasterized=True)
    for x,sign,color in ((-3,-1,"tab:blue"),(3,1,"tab:orange")):
        ax.arrow(x,0,0,sign*1.2,width=.035,head_width=.25,head_length=.25,
                 length_includes_head=True,color=color,zorder=5)
    ax.set(xlabel="isotropic x / M",ylabel="isotropic y / M",xlim=(-32,32),ylim=(-32,32),
           aspect="equal",title="Session 007 initial particles; arrows show clump bulk motion")
    ax.legend(markerscale=10,loc="upper right")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=190)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)

if __name__=="__main__": main()
