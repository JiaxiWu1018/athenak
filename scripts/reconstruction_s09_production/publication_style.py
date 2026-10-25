"""Publication files from fully audited Session 8 reductions; no synthetic rows."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import AutoLocator,StrMethodFormatter
from common import *
from reduce import csv_rows,write_csv

def style():
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'dejavuserif','font.size':10,
      'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,'axes.linewidth':.7,
      'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
      'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})

def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(ROOT/'plots'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)

def array(rows,key):return np.array([r[key] for r in rows])

def time_axis(ax,end):
    """Keep the actual fractional endpoint explicit without crowded edge ticks."""
    ax.set_xlim(0,end);ax.xaxis.set_major_locator(AutoLocator())
    ticks=ax.get_xticks();step=float(np.median(np.diff(ticks))) if len(ticks)>1 else end
    inside=ticks[(ticks>0)&(ticks<end-.75*step)]
    ax.set_xticks(np.r_[0,inside,end]);ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))

def shell_limits(*datasets):
    values=np.array([row[f'r{q}'] for rows in datasets for row in rows
                     for q in range(10,100,10)])
    assert values.size and np.isfinite(values).all() and (values>0).all()
    # Keep the historical reference range, expanding it to include every curve.
    return (min(.05,float(values.min())/1.08),max(30.,float(values.max())*1.08))
