"""Single CPU support lane: packet verification, plots, bounded movies and reports."""
import sys
from common import *
from extension_resources import admission

def reference():
    from products import verify
    packet=EXT/'imported/yc0500_u0005';m=verify(packet)
    a=load(packet/'numerical_acceptance.json');s=load(packet/'reduced/summary.json')
    assert m['case']['key']=='yc0500_u0005' and m['case']['U']==-.005 and m['case']['yc']==.5
    assert a['numerical_checks_passed'] and s['complete'] and abs(s['endpoint']-93)<1e-7 and abs(s['ah_first']-73)<1e-7
    assert m['checkpoint']['sha256']==a['checkpoint']['sha256']
    atomic(EXT/'evidence/REFERENCE_PACKET_OK.json',dict(utc=now(),job=job(),manifest=receipt(packet/'MANIFEST.json'),original_source_root=m['source_root'],source_machine=m['machine'],complete=True,endpoint_M=s['endpoint'],accepted_AH_M=s['ah_first'],no_new_evolution=True,source_acceptance=receipt(packet/'numerical_acceptance.json')))
def main():
    job();admission();mode=sys.argv[1]
    if mode=='reference':reference()
    elif mode=='plots':
        from products import assemble,figures
        reference();figures(assemble(sys.argv[2:] or None))
        from pdf_previews import main as pdfs
        pdfs()
    elif mode=='movie':
        import movies
        sys.argv=['movies',sys.argv[2]];movies.main()
    elif mode=='render':
        from movie_chunks import render
        render(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
    elif mode=='encode':
        from movie_chunks import encode
        encode(sys.argv[2])
    elif mode=='reports':
        from reports import main as reports
        reports()
    else:raise ValueError(mode)
if __name__=='__main__':main()
