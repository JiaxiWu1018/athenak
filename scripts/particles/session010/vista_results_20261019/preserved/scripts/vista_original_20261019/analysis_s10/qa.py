#!/usr/bin/env python3
"""Generate review sheets from actual exported PDFs and decoded MP4 frames."""
from pathlib import Path
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import imageio_ffmpeg

from run import OUT, atomic, check_time, digest


def sheet(items, path, columns=2, width=760):
    thumbs = []
    for title, image in items:
        picture = image.copy().convert("RGB")
        picture.thumbnail((width, 500))
        canvas = Image.new("RGB", (width, picture.height + 28), "white")
        canvas.paste(picture, ((width - picture.width) // 2, 28))
        ImageDraw.Draw(canvas).text((8, 6), title, fill="black")
        thumbs.append(canvas)
    height = max(im.height for im in thumbs)
    result = Image.new("RGB", (columns * width, height * ((len(thumbs) + columns - 1) // columns)), "#eeeeee")
    for i, im in enumerate(thumbs):
        result.paste(im, ((i % columns) * width, (i // columns) * height))
    result.save(path)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != "gg":
        raise RuntimeError("Review rendering requires allocated compute")
    qa = OUT / "qa"; qa.mkdir(exist_ok=True)
    previews = sorted((qa / "pdf_previews").glob("*.png"))
    if len(previews) != 7:
        raise RuntimeError("Run the verification stage before review sheets")
    items = [(p.stem, Image.open(p)) for p in previews]
    image = sheet(items, qa / "all_pdf_figures.png")
    ImageOps.grayscale(image).save(qa / "all_pdf_figures_grayscale.png")
    items = []; entries = []
    for movie in sorted(OUT.glob("*.mp4")):
        check_time()
        reader = imageio_ffmpeg.read_frames(str(movie), input_params=["-threads", "2"], output_params=["-threads", "2"])
        meta = next(reader)
        expected = json.loads((OUT / "MOVIES.json").read_text())[movie.name]["encoded_frames"]
        picks = {0, expected // 2, expected - 1}
        for k, raw in enumerate(reader):
            if k not in picks:
                continue
            picture = Image.frombytes("RGB", tuple(meta["size"]), raw)
            target = qa / f"{movie.stem}_{k:04d}_decoded.png"
            picture.save(target)
            items.append((f"{movie.name}: decoded frame {k}", picture))
            entries.append(dict(path=str(target.relative_to(OUT)), movie_sha256=digest(movie),
                                preview_sha256=digest(target), movie=movie.name, frame=k))
    sheet(items, qa / "all_movies_decoded.png", columns=3, width=620)
    atomic(qa / "REVIEW_SOURCES.json", dict(pdf_previews=[dict(path=str(p.relative_to(OUT)),
           preview_sha256=digest(p), pdf=p.name.replace('.png', '.pdf'),
           pdf_sha256=digest(OUT / p.name.replace('.png', '.pdf'))) for p in previews], decoded_frames=entries))


if __name__ == "__main__":
    main()
