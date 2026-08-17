"""Artifact readers and assertions shared by particle MPI regressions."""

import csv
import glob
import os
import re
import shutil
import struct

import numpy as np
import pytest

import test_suite.testutils as testutils


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def require_problem(name):
    """Skip unless the test build uses the requested user problem generator."""
    cache = os.path.join(_REPO, "tst", "build", "CMakeCache.txt")
    problem = None
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as stream:
            for line in stream:
                match = re.match(r"PROBLEM:\w+=(.*?)\s*$", line)
                if match:
                    problem = match.group(1)
    if problem != name:
        pytest.skip(
            f"needs a -DPROBLEM={name} build (cache has PROBLEM={problem})",
            allow_module_level=True,
        )


def rank_counts(requested):
    """Do not oversubscribe small development or CI machines."""
    available = min(4, os.cpu_count() or 1)
    return [count for count in requested if count <= available]


def run_case(input_file, run_dir, ranks, extra_args=None):
    if extra_args is None:
        extra_args = []
    command = [
        "mpirun",
        "-np",
        str(ranks),
        "./athena",
        "-i",
        input_file,
        "-d",
        run_dir,
    ] + list(extra_args)
    if not testutils.run_command(command):
        raise RuntimeError(f"particle regression failed with {ranks} MPI rank(s)")


def remove_dirs(*paths):
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def read_particle_vtk(path):
    """Return time, position, velocity, and tag arrays from a binary particle VTK."""
    with open(path, "rb") as stream:
        data = stream.read()
    time = float(re.search(rb"time=\s*([-+0-9.eE]+)", data).group(1))
    points = re.search(rb"POINTS\s+(\d+)\s+float", data)
    count = int(points.group(1))

    def block(marker, size):
        start = data.find(marker)
        if start < 0:
            pytest.fail(f"{path}: missing {marker.decode()!r}")
        start = data.find(b"\n", start + len(marker)) + 1
        return np.array(struct.unpack(f">{size}f", data[start:start + 4 * size]))

    positions = block(points.group(0), 3 * count).reshape(count, 3)
    velocities = block(b"VECTORS prtcl_vel float", 3 * count).reshape(count, 3)
    tags = block(
        b"SCALARS ptag float\nLOOKUP_TABLE default", count
    ).astype(int)
    return time, positions, velocities, tags


def particle_dumps(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "pvtk", "*.part.vtk")))
    if not paths:
        pytest.fail(f"no particle VTK output in {run_dir}")
    return paths


def assert_analytic_periodic_drift(run_dir, tolerance=5.0e-7):
    dumps = particle_dumps(run_dir)
    assert len(dumps) >= 2, "analytic drift check needs at least two outputs"
    first, last = dumps[0], dumps[-1]
    t0, x0, v0, tag0 = read_particle_vtk(first)
    t1, x1, _, tag1 = read_particle_vtk(last)
    order0, order1 = np.argsort(tag0), np.argsort(tag1)
    assert np.array_equal(tag0[order0], tag1[order1]), "particle tag set changed"
    expected = (x0[order0] + v0[order0] * (t1 - t0) + 0.5) % 1.0 - 0.5
    error = np.abs(x1[order1] - expected)
    error = np.minimum(error, 1.0 - error)
    assert error.max() <= tolerance, f"analytic drift error {error.max():.3e}"


def assert_final_positions_bitwise(first_dir, second_dir):
    first = particle_dumps(first_dir)[-1]
    second = particle_dumps(second_dir)[-1]
    t0, x0, _, tag0 = read_particle_vtk(first)
    t1, x1, _, tag1 = read_particle_vtk(second)
    order0, order1 = np.argsort(tag0), np.argsort(tag1)
    assert t0 == t1, "final output times differ"
    assert np.array_equal(tag0[order0], tag1[order1]), "final tag sets differ"
    assert np.array_equal(
        x0[order0], x1[order1]
    ), "final positions are not bitwise equal"


def read_death_csv(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].startswith("#"):
                continue
            rows.append(
                {
                    "cycle": int(row[0]),
                    "time": float(row[1]),
                    "tag": int(row[2]),
                    "reason": row[3],
                    "x": float(row[4]),
                    "y": float(row[5]),
                    "z": float(row[6]),
                    "vx": float(row[7]),
                    "vy": float(row[8]),
                    "vz": float(row[9]),
                    "crit": float(row[12]),
                }
            )
    return rows


def death_key(row):
    """Compare physical death data while excluding rank and gather order."""
    return tuple(
        row[key]
        for key in (
            "cycle",
            "time",
            "reason",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "crit",
        )
    )
