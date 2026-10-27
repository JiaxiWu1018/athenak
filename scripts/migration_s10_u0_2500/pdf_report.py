"""Allocated MD-to-PDF report using the accepted Session10 page layout."""
import shutil
import textwrap
from common import *


def render_pdf(lines, path, figures):
    job()
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    wrapped = '\n'.join('\n'.join(textwrap.wrap(line, 94)) if line else ''
                        for line in lines).splitlines()
    pages = []
    with PdfPages(path) as pdf:
        for start in range(0, len(wrapped), 62):
            page = wrapped[start:start+62]
            assert all(len(line) <= 94 for line in page)
            fig = plt.figure(figsize=(8.3, 11.7))
            fig.text(.055, .95, '\n'.join(page), va='top', fontsize=8, family='monospace')
            pdf.savefig(fig)
            plt.close(fig)
            pages.append(dict(page=len(pages)+1, kind='narrative', lines=len(page)))
        for source in figures:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.imshow(plt.imread(source))
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            pages.append(dict(page=len(pages)+1, kind='figure', source=receipt(source)))
        assert pdf.get_pagecount() == len(pages)
    previews = []
    rasterizer = shutil.which('pdftoppm')
    if rasterizer:
        prefix = path.parent/'report_page'
        cmd([rasterizer, '-r', '90', '-png', str(path), str(prefix)], timeout=90)
        previews = [receipt(p) for p in sorted(path.parent.glob('report_page-*.png'))]
        assert len(previews) == len(pages)
    record = dict(utc=now(), job=job(), renderer=receipt(Path(__file__)),
        pdf=receipt(path), pages=pages, maximum_text_columns=94,
        maximum_text_lines_per_page=62, actual_PDF_raster_review_pending=True,
        actual_pdf_previews=previews, rasterizer=rasterizer,
        note='Generated PDF pages still require actual visual inspection before acceptance.')
    atomic(path.parent/'REPORT_LAYOUT.json', record)
    return record
