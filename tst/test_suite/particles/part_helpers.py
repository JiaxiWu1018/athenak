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


ABORT_TIMEOUT = 600.0   # a run expected to abort must not be able to hang CI


def run_args(args, threads=1, expect_failure=False):
    """Run Athena with arbitrary CLI arguments and return this command's log output.

    ``expect_failure`` inverts the exit-status assertion, for regressions whose point is
    that the code must refuse to continue. The log is returned either way so the caller
    can assert on the message: a nonzero exit alone would also match an unrelated crash.
    """
    command = ["mpirun", "-np", str(threads), "./athena"] + list(args)
    start = (
        os.path.getsize(testutils.LOG_FILE_PATH)
        if os.path.exists(testutils.LOG_FILE_PATH)
        else 0
    )
    ok = testutils.run_command(
        command, timeout=ABORT_TIMEOUT if expect_failure else None
    )
    if ok and expect_failure:
        raise RuntimeError(
            f"run completed with {threads} MPI rank(s) but was expected to abort"
        )
    if not ok and not expect_failure:
        raise RuntimeError(f"particle regression failed with {threads} MPI rank(s)")
    with open(testutils.LOG_FILE_PATH, "rb") as stream:
        stream.seek(start)
        return stream.read().decode(errors="replace")


def run_case(input_file, run_dir, ranks, extra_args=None, expect_failure=False):
    if extra_args is None:
        extra_args = []
    return run_args(
        [
            "-i",
            input_file,
            "-d",
            run_dir,
        ]
        + list(extra_args),
        threads=ranks,
        expect_failure=expect_failure,
    )


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


def assert_final_particle_state_bitwise(first_dir, second_dir):
    """Compare final particle positions and velocities by persistent particle tag."""
    first = particle_dumps(first_dir)[-1]
    second = particle_dumps(second_dir)[-1]
    t0, x0, v0, tag0 = read_particle_vtk(first)
    t1, x1, v1, tag1 = read_particle_vtk(second)
    order0, order1 = np.argsort(tag0), np.argsort(tag1)
    assert t0 == t1, "final output times differ"
    assert np.array_equal(tag0[order0], tag1[order1]), "final tag sets differ"
    assert np.array_equal(x0[order0], x1[order1]), "final positions differ"
    assert np.array_equal(v0[order0], v1[order1]), "final velocities differ"


def _binary_header(stream):
    """Read an Athena binary-v1.1 header, leaving the stream at its first block."""
    stream.seek(0)
    magic = stream.readline().split()
    assert magic and magic[0] == b"Athena" and magic[-1].endswith(b"1.1")
    header = {}
    for _ in range(int(stream.readline().split(b"=")[-1]) - 1):
        key, value = stream.readline().decode().split("=", 1)
        header[key.strip()] = value.strip()
    variable_count = int(stream.readline().split(b"=")[-1])
    stream.readline()
    stream.seek(int(stream.readline().split(b"=")[-1]), 1)
    return (
        float(header["time"]),
        int(header["size of location"]),
        int(header["size of variable"]),
        variable_count,
    )


def read_binary_blocks(path):
    """Return dump time, leaf levels, and field bytes keyed by logical location."""
    levels = []
    blocks = {}
    with open(path, "rb") as stream:
        stream.seek(0, 2)
        file_size = stream.tell()
        time, location_bytes, variable_bytes, variable_count = _binary_header(stream)
        while stream.tell() < file_size:
            index = struct.unpack("=6i", stream.read(24))
            logical = struct.unpack("=4i", stream.read(16))
            levels.append(logical[3])
            stream.seek(6 * location_bytes, 1)
            cells = (
                (index[1] - index[0] + 1)
                * (index[3] - index[2] + 1)
                * (index[5] - index[4] + 1)
            )
            blocks[logical] = stream.read(cells * variable_count * variable_bytes)
        assert stream.tell() == file_size, f"binary block walk misaligned in {path}"
    return time, levels, blocks


def tmunu_dumps(run_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "bin", "*.tmunu.*.bin")))
    if not paths:
        pytest.fail(f"no Tmunu binary output in {run_dir}")
    return paths


def assert_final_tmunu_bitwise(first_dir, second_dir):
    """Compare final Tmunu payloads independent of rank-dependent block write order."""
    first = tmunu_dumps(first_dir)[-1]
    second = tmunu_dumps(second_dir)[-1]
    t0, _, blocks0 = read_binary_blocks(first)
    t1, _, blocks1 = read_binary_blocks(second)
    assert t0 == t1, "final Tmunu output times differ"
    assert blocks0.keys() == blocks1.keys(), "final Tmunu meshes differ"
    for logical in blocks0:
        assert blocks0[logical] == blocks1[logical], (
            f"final Tmunu differs at logical block {logical}"
        )


def assert_mixed_level_output(run_dir):
    """Require a dump with both coarse and fine leaves, hence a real AMR seam."""
    seen = []
    for path in tmunu_dumps(run_dir):
        _, levels, _ = read_binary_blocks(path)
        unique = sorted(set(levels))
        seen.append((os.path.basename(path), unique))
        if len(unique) > 1:
            return
    pytest.fail(f"no mixed-level Tmunu dump in {run_dir}: {seen}")


def assert_cross_level_deposition(log):
    counts = [int(value) for value in re.findall(r"cross_level=(\d+)", log)]
    assert counts, "run log contains no cross-level diagnostics"
    assert max(counts) > 0, f"cross-level deposition stayed zero: {counts}"


def assert_regridded(log):
    changes = [
        (int(created), int(deleted))
        for created, deleted in re.findall(
            r"(\d+) MeshBlocks created,\s*(\d+) deleted by AMR", log
        )
    ]
    assert changes, "run log contains no AMR topology summary"
    assert any(created or deleted for created, deleted in changes), (
        f"AMR made no topology change: {changes}"
    )


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
