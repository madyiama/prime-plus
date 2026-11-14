"""Strict disk cache for unfiltered LGS focal-angular geometry SF.

Cache ONLY the pupil-INDEPENDENT focal structure-function result:
no rotated pupil mask, NCPA/OPD, static OTF, HO statistics, TT,
or non-scalar spatial Hfilter. Store one unweighted per-layer SF.
The caller applies r0 and Cn2 weights on EVERY reconstruction.

Key includes the exact geometry, wavelength-independent telescope
diameter, DM sampling, pupil-axis convention, angular field, beacon
height and directions, atmospheric altitudes and L0.
Validated schema and checksum reject corrupt/stale entries.
Writes are atomic; no pickle, mutable globals or network requests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile
from time import perf_counter

import numpy as np

SCHEMA = "p3.focal_unfiltered_sf.v1"
VERSION = "flicker_exact_symmetry_2026_11_v1"
PREFIX = "focal_symmetric"
# Bounded, approximate on-disk LRU. Only this cache's own entries
# can be evicted; no optical products or user files are touched.
MAX_CACHE_ENTRIES = 64
MAX_CACHE_BYTES = 256*1024*1024


def _prune_cache(folder, *, protected):
    """Evict oldest cached geometry entries, never a newly written entry."""
    files=[]
    for path in Path(folder).glob(PREFIX+"_*.npz"):
        try:
            st=path.stat()
        except OSError:
            continue
        files.append((st.st_mtime_ns,st.st_size,path))
    files.sort(key=lambda row:(row[0],row[2].name))
    size=sum(x[1] for x in files)
    count=len(files)
    for _, nbytes, path in files:
        if count<=MAX_CACHE_ENTRIES and size<=MAX_CACHE_BYTES:
            break
        if path==protected:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        count-=1
        size-=nbytes




def _scalar(value, label, *, allow_positive_inf=False):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a real scalar") from exc
    if a.size != 1:
        raise ValueError(f"{label} must have exactly one value")
    v = float(a.reshape(-1)[0])
    if not np.isfinite(v) and not (allow_positive_inf and np.isposinf(v)):
        raise ValueError(f"{label} must be finite")
    return "positive_inf" if np.isposinf(v) else v.hex()


def _array(value, label):
    v = np.asarray(value,dtype=np.float64)
    if v.size < 1 or not np.isfinite(v).all():
        raise ValueError(f"{label} must contain finite values")
    return {
        "shape":list(v.shape),
        "values":[float(x).hex() for x in v.ravel(order="C")],
    }


def key_for_focal(
    tel, atm, src, gs, nActu, nOtf, samp,
    *, pupil_grid_mode, output_grid, algorithm="symmetry",
):
    """Exact float64 cache fingerprint: never inspect the rotated pupil.

    Callers must bypass the cache if Hfilter is non-scalar.
    No pupil, pupilAngle, opdMap, otfNCPA, turbulence weights
    or r0 should appear in this key.
    """
    n_actu=np.asarray(nActu)
    if n_actu.size!=1 or not np.isfinite(n_actu).all():
        raise ValueError("Focal cache requires a single actuator count")
    n=int(float(n_actu.reshape(-1)[0]))
    if n<1 or float(n)!=float(n_actu.reshape(-1)[0]):
        raise ValueError("Focal cache requires a positive integer actuator count")
    if algorithm not in ("symmetry", "quadrant"):
        raise ValueError("Unsupported focal geometry cache algorithm")
    params={
        "schema":SCHEMA,"version":VERSION,"algorithm":algorithm,
        "telescope_diameter_m":_scalar(tel.D,"tel.D"),
        "n_actu":n,
        "n_otf":int(nOtf),
        "sampling":_scalar(samp,"samp"),
        "pupil_grid_mode":str(pupil_grid_mode),
        "output_grid":str(output_grid),
        "n_src":int(src.nSrc),
        "n_layers":int(atm.nL),
        "atm_heights_m":_array(atm.heights,"atm.heights"),
        "outer_scale_m":_scalar(atm.L0,"atm.L0",allow_positive_inf=True),
        "source_direction_rad":_array(src.direction,"src.direction"),
        "laser_direction_rad":_array(gs.direction,"gs.direction"),
        "laser_height_m":_scalar(gs.height,"gs.height",allow_positive_inf=True),
    }
    blob=json.dumps(params,sort_keys=True,separators=(",",":"),
                    ensure_ascii=True,allow_nan=False).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def _checksum(image):
    a=np.ascontiguousarray(image,dtype=np.float64)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def _validate_array(array,shape):
    return (isinstance(array,np.ndarray)
            and array.dtype==np.dtype("float64")
            and array.shape==tuple(shape)
            and bool(np.isfinite(array).all()))


def read_focal_cache(folder, key, shape, *, diagnostics=None):
    """Return an independent float64 copy or None, never stale bytes."""
    start=perf_counter()
    target=Path(folder)/f"{PREFIX}_{key}.npz"
    status="miss"
    result=None
    if target.is_file():
        try:
            with np.load(target,allow_pickle=False) as entry:
                if set(entry.files)!={"schema","key","checksum","layers"}:
                    raise ValueError("unknown cache fields")
                a=np.asarray(entry["layers"])
                if (str(entry["schema"].item())!=SCHEMA
                        or str(entry["key"].item())!=key
                        or not _validate_array(a,shape)
                        or _checksum(a)!=str(entry["checksum"].item())):
                    raise ValueError("invalid cached SF")
                result=np.array(a,dtype=np.float64,copy=True)
                status="hit"
            try:
                target.touch()  # approximate disk LRU, not science metadata
            except OSError:
                pass
        except (OSError,ValueError,KeyError,TypeError,EOFError,BadZipFile):
            status="invalidated"
    if diagnostics is not None:
        diagnostics["cache_status"]=status
        diagnostics["cache_read_ms"]=1000*(perf_counter()-start)
        diagnostics["cache_key"]=key
    return result


def write_focal_cache(folder, key, image, *, diagnostics=None):
    """Atomically store validated unfiltered SF with data checksum."""
    start=perf_counter()
    a=np.asarray(image,dtype=np.float64)
    if a.ndim!=4 or not np.isfinite(a).all():
        raise ValueError("Cannot cache an invalid focal-angular SF")
    path=Path(folder)
    path.mkdir(parents=True,exist_ok=True)
    target=path/f"{PREFIX}_{key}.npz"
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",dir=path,prefix=f".{PREFIX}_{key}_",
            suffix=".npz",delete=False
        ) as stream:
            tmp=Path(stream.name)
            np.savez(
                stream,schema=np.asarray(SCHEMA),key=np.asarray(key),
                checksum=np.asarray(_checksum(a)),layers=a,
            )
        os.replace(tmp,target)
        tmp=None
        if diagnostics is not None:
            diagnostics["cache_write_status"]="stored"
        try:
            _prune_cache(path,protected=target)
        except OSError:
            # Cache eviction is optional; a successful scientific result
            # must never be discarded because pruning failed.
            if diagnostics is not None:
                diagnostics["cache_prune_status"]="failed_nonfatal"
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        if diagnostics is not None:
            diagnostics["cache_write_ms"]=1000*(perf_counter()-start)
    return target
