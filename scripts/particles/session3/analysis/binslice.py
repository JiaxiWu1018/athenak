#!/usr/bin/env python3
"""Standalone numpy-only reader for AthenaK `bin` slice dumps.

Lifted verbatim from evidence/2026-08-17_bssn_parabolic_H_damping/tests/run_battery.py
(validated bit-for-bit against vis/python/bin_convert.py in the 2026-08-10 campaign).
Used instead of bin_convert.py because the latter's get_from_header does
line.split("=") with no maxsplit and dies on these campaigns' headers, whose
<comment>/problem value itself contains "=".
"""
import numpy as np


def read_bin_slice(filename):
    fd = {}
    with open(filename, "rb") as fp:
        fp.seek(0, 2)
        filesize = fp.tell()
        fp.seek(0, 0)

        code_header = fp.readline().split()
        if not code_header or code_header[0] != b"Athena":
            raise TypeError("not an Athena binary dump: %s" % filename)
        version = code_header[-1].split(b"=")[-1]
        if version != b"1.1":
            raise TypeError("unsupported bin version %r" % version)

        pheader_count = int(fp.readline().split(b"=")[-1])
        pheader = {}
        for _ in range(pheader_count - 1):
            key, val = [x.strip() for x in
                        fp.readline().decode("utf-8").split("=", 1)]
            pheader[key] = val
        time = float(pheader["time"])
        cycle = int(pheader["cycle"])
        locsizebytes = int(pheader["size of location"])
        varsizebytes = int(pheader["size of variable"])

        nvars = int(fp.readline().split(b"=")[-1])
        var_list = [v.decode("utf-8") for v in fp.readline().split()[1:]]
        header_size = int(fp.readline().split(b"=")[-1])
        header = [line.decode("utf-8").split("#")[0].strip()
                  for line in fp.read(header_size).split(b"\n")]
        header = [line for line in header if line]

        if locsizebytes not in (4, 8) or varsizebytes not in (4, 8):
            raise ValueError("unsupported location/variable size")
        locdtype = np.float64 if locsizebytes == 8 else np.float32
        vardtype = np.float64 if varsizebytes == 8 else np.float32

        def hget(block, key):
            cur = "<none>"
            for line in header:
                if line.startswith("<"):
                    cur = line
                    continue
                k, v = line.split("=", 1)
                if cur == block and k.strip() == key:
                    return v.strip()
            raise KeyError("%s/%s" % (block, key))

        nghost = int(hget("<mesh>", "nghost"))

        mb_index, mb_logical, mb_geometry = [], [], []
        mb_data = {v: [] for v in var_list}
        while fp.tell() < filesize:
            idx = np.frombuffer(fp.read(24),
                                dtype=np.int32).astype(np.int64) - nghost
            mb_index.append(idx)
            nx1_out = (idx[1] - idx[0]) + 1
            nx2_out = (idx[3] - idx[2]) + 1
            nx3_out = (idx[5] - idx[4]) + 1
            mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
            mb_geometry.append(np.frombuffer(fp.read(6 * locsizebytes),
                                            dtype=locdtype))
            data = np.fromfile(fp, dtype=vardtype,
                               count=nx1_out * nx2_out * nx3_out * nvars)
            data = data.reshape(nvars, nx3_out, nx2_out, nx1_out)
            for vi, var in enumerate(var_list):
                mb_data[var].append(data[vi])

    fd["header"] = header
    fd["time"] = time
    fd["cycle"] = cycle
    fd["var_names"] = var_list
    fd["nvars"] = nvars
    fd["n_mbs"] = len(mb_index)
    fd["mb_index"] = np.array(mb_index)
    fd["mb_logical"] = np.array(mb_logical)
    fd["mb_geometry"] = np.array(mb_geometry)
    fd["mb_data"] = mb_data
    return fd
