"""
Particle tests for the experimental transported-frame Boris pusher.

The Schwarzschild regression is self-contained: it generates a one-particle
HDF5 table in the build directory using h5import, then runs the KS circular
orbit that exposed the secular drift in the original geo_boris pusher.
"""

import math
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _flat_boris_push(u, e, b, qom, dt):
    u_minus = tuple(u[i] + 0.5 * qom * dt * e[i] for i in range(3))
    gamma = math.sqrt(1.0 + sum(ui * ui for ui in u_minus))
    t = tuple(0.5 * qom * dt * bi / gamma for bi in b)
    t2 = sum(ti * ti for ti in t)
    s = tuple(2.0 * ti / (1.0 + t2) for ti in t)
    u_prime_cross_t = _cross(u_minus, t)
    u_prime = tuple(u_minus[i] + u_prime_cross_t[i] for i in range(3))
    u_plus_cross_s = _cross(u_prime, s)
    return tuple(u_minus[i] + u_plus_cross_s[i] + 0.5 * qom * dt * e[i]
                 for i in range(3))


def _four_dot(gcov, a, b):
    return sum(gcov[mu][nu] * a[mu] * b[nu]
               for mu in range(4) for nu in range(4))


def _orthonormality_error(e, gcov):
    eta = [-1.0, 1.0, 1.0, 1.0]
    err = 0.0
    for a in range(4):
        for b in range(4):
            target = eta[a] if a == b else 0.0
            err = max(err, abs(_four_dot(gcov, e[a], e[b]) - target))
    return err


def test_flat_frame_boris_math_zero_fields_and_rotation():
    """Fast algebra check for the identity geodesic map and tetrad rotation."""
    u = (0.23, -0.11, 0.31)
    pushed = _flat_boris_push(u, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 0.8)
    assert pushed == pytest.approx(u, abs=1.0e-15)

    theta = 0.41
    c = math.cos(theta)
    s = math.sin(theta)
    tetrad = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, c, s, 0.0),
        (0.0, -s, c, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    minkowski = (
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert _orthonormality_error(tetrad, minkowski) < 1.0e-12


def _configured_for_schwz_ptcl():
    cache = REPO_ROOT / "tst" / "build" / "CMakeCache.txt"
    if not cache.exists():
        return False
    for line in cache.read_text().splitlines():
        if line.startswith("PROBLEM:STRING="):
            return line.split("=", 1)[1] == "Schwz_ptcl"
    return False


def _write_particle_hdf5(path):
    h5import = shutil.which("h5import")
    if h5import is None:
        pytest.skip("h5import is required to generate the Schwarzschild particle IC")

    radius = 10.0
    mass = 1.0
    ut = 1.0 / math.sqrt(1.0 - 3.0 * mass / radius)
    uphi = math.sqrt(mass / radius**3) / math.sqrt(1.0 - 3.0 * mass / radius)
    values = {
        "x": radius,
        "y": 0.0,
        "z": 0.0,
        "ux": 2.0 * mass * ut / radius,
        "uy": radius * uphi,
        "uz": 0.0,
    }

    path.unlink(missing_ok=True)
    command = [h5import]
    for name, value in values.items():
        data_path = path.with_name(f"{path.stem}_{name}.txt")
        cfg_path = path.with_name(f"{path.stem}_{name}.cfg")
        data_path.write_text(f"{value:.17g}\n")
        cfg_path.write_text(
            "\n".join((
                f"PATH {name}",
                "INPUT-CLASS TEXTFP",
                "INPUT-SIZE 64",
                "INPUT-BYTE-ORDER LE",
                "RANK 1",
                "DIMENSION-SIZES 1",
                "OUTPUT-CLASS FP",
                "OUTPUT-SIZE 64",
                "OUTPUT-ARCHITECTURE IEEE",
                "OUTPUT-BYTE-ORDER LE",
                "",
            ))
        )
        command.extend((str(data_path), "-c", str(cfg_path)))
    command.extend(("-o", str(path)))
    subprocess.run(command, check=True)


def _write_athinput(path, particle_path):
    path.write_text(f"""
<job>
basename = schwz_ks_geo

<mesh>
nghost    = 4
nx1       = 30
x1min     = -15.0
x1max     = 15.0
ix1_bc    = outflow
ox1_bc    = outflow

nx2       = 30
x2min     = -15.0
x2max     = 15.0
ix2_bc    = outflow
ox2_bc    = outflow

nx3       = 30
x3min     = -15.0
x3max     = 15.0
ix3_bc    = outflow
ox3_bc    = outflow

<meshblock>
nx1       = 30
nx2       = 30
nx3       = 30

<time>
evolution  = dynamic
integrator = rk2
cfl_number = 0.3
nlim       = 100000
tlim       = 400.0
ndiag      = 200

<coord>
general_rel = true
a           = 0.0
excise      = false

<adm>
dynamic = false

<particles>
init              = file
prtcl_init_file   = {particle_path}
particle_type     = cosmic_ray
pusher            = geo_boris
charge_mass_ratio = 0.0

<problem>
metric_type = ks

<output1>
file_type = pvtk
variable  = prtcl_all
dt        = 1.0
""")


def _read_vtk_scalar(data, header, count):
    idx = data.find(header)
    assert idx >= 0
    start = idx + len(header)
    if data[start:start + 1] == b"\n":
        start += 1
    if data[start:start + 12] == b"LOOKUP_TABLE":
        start = data.find(b"\n", start) + 1
    return struct.unpack(f">{count}f", data[start:start + 4 * count])


def _read_particle_history(basename):
    rows = []
    files = sorted(Path("pvtk").glob(f"{basename}.*.part.vtk"))
    assert len(files) == 401
    for path in files:
        data = path.read_bytes()
        time = float(re.search(rb"time=\s*([0-9eE+\-.]+)", data).group(1))
        x = _read_vtk_scalar(data, b"POINTS 1 float\n", 3)
        energy = _read_vtk_scalar(data, b"SCALARS energy float\n", 1)[0]
        rows.append((time, x, energy))
    return rows


def _run_case(input_path, pusher, basename):
    shutil.rmtree("pvtk", ignore_errors=True)
    command = [
        "./athena",
        "-i", str(input_path),
        f"job/basename={basename}",
        f"particles/pusher={pusher}",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr, _read_particle_history(basename)


def test_ks_geo_boris_fw_boris_removes_secular_drift_cpu():
    if not Path("athena").exists():
        pytest.skip("run from tst/build/src after building AthenaK")
    if not _configured_for_schwz_ptcl():
        pytest.skip("requires AthenaK configured with -DPROBLEM=Schwz_ptcl")

    particle_path = Path.cwd() / "schwz_ptcl_particles_ks_test.h5"
    input_path = Path.cwd() / "schwz_ptcl_ks_geo_boris_test.athinput"
    _write_particle_hdf5(particle_path)
    _write_athinput(input_path, particle_path)

    try:
        _, baseline_rows = _run_case(input_path, "geo_boris", "schwz_ks_geo_base")
        base_e0 = baseline_rows[0][2]
        base_rel_end = (baseline_rows[-1][2] - base_e0) / base_e0
        assert abs(base_rel_end) > 1.0e-4

        log, fw_rows = _run_case(input_path, "geo_boris_fw_boris", "schwz_ks_fw_boris")
        energies = [row[2] for row in fw_rows]
        radii = [math.sqrt(sum(xi * xi for xi in row[1])) for row in fw_rows]
        rel_end = (energies[-1] - energies[0]) / energies[0]

        assert abs(rel_end) < 2.0e-6
        assert min(radii) > 9.98
        assert max(radii) < 10.02
        assert "geo_boris_fw_boris max tetrad orthonormality error" not in log
        assert "Root finding of geo_boris_fw_boris" not in log
    finally:
        shutil.rmtree("pvtk", ignore_errors=True)
