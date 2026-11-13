# Keck PSF-R V1 — SIGKILL during psfR initialization

## Observation (2026-11-13)

The Keck PSF Lab job `b095fc22-01f3-4635-9af6-fbf8c9cd854b`
completed Keck telemetry ingestion, system diagnosis, atmospheric fitting
(`r0_tel=0.1192871951 m`, `L0=13.12936977 m`),
HO covariance **preflight with zero negative fraction**, and then emitted
`[psfR] START` with no further traceback.

The upstream `keck_psfr_v1_e2e.py` acceptance report recorded
`process_returncode=-9`, i.e. the **nested P3 process received SIGKILL**.
The outer V1 wrapper reported exit=1. **No Python exception was recorded.**

**SIGKILL is not proof of OOM.** Other possible causes include an
external kill or container resource policy. Confirm Linux cgroup memory
events and the actual memory cap before assigning root cause.

## Identified memory hot spot

`p3/psfr/psfrUtils.py:modes2Otf` previously formed the full
eigenmode matrix and a *simultaneously padded 3-D cube*:

```python
M_all = modes @ eigenvectors
M_all = M_all.reshape(n_pup, n_pup, nModes)
M_all = FourierUtils.enlargeSupport(M_all, samp) * pup_pad[..., None]
for k in range(nModes):
    Mk = M_all[:, :, k]
    # Vii Fourier evaluation
```

The final cube alone costs approximately

`8 * (samp * n_pup)**2 * nModes` bytes for float64.

**Illustrative size, not a measured Keck allocation:** at
`n_pup=272`, `samp=2`, `nModes=300`, this is ~677 MiB
(not counting original modes, eigenvectors, padded temporary,
complex FFT arrays, spatial-frequency grids or earlier PSFR fields).

This PR computes the *same* eigenvector-projected modal pupil one mode at
a time, then performs the same Vii Fourier terms in the same order.
The full padded 3-D cube and the full pre-pad eigenmode matrix are
no longer materialised. Scientific policy remains unchanged:

- strictly reject negative modal covariance eigenvalues beyond
  numerical eigensolver tolerance;
- never clip physically significant HO variances;
- retain the same pupil-mask multiplication and FFT terms;
- do not alter TT noise, atmospheric fitting or focal anisoplanatism.

The modal linear algebra can differ at floating-point roundoff because
a matrix-vector BLAS product replaces one large matrix-matrix product.
An independent *legacy full-cube reference test* checks both OTF and
structure function with tolerance `rtol=atol=1e-11`, across 3, 8
and 16 correlated modes, and guards against future 3-D padding.

## Extra diagnosis for future SIGKILL

`psfR.__init__` emits flushed Linux live-RSS checkpoints at:
AO-system initialization; spatial frequency grid; DM-fitting
structure function; aliasing; TT jitter; HO Vii covariance/FFT; and
the final error budget.

For each stage the log prints `[psfR memory] stage: RSS=... MiB`
if `/proc/self/status` is readable, otherwise explicitly
`RSS unavailable`. These prints do **not** change the model arrays.

On the user's running Docker container:

```bash
docker compose exec -T app sh -lc '
echo "=== cgroup memory ==="
for f in /sys/fs/cgroup/memory.max \
         /sys/fs/cgroup/memory.peak \
         /sys/fs/cgroup/memory.events; do
  echo "--- $f"; cat "$f" 2>/dev/null || true
done
'
```

`oom_kill > 0` in `memory.events` is compelling evidence of a
cgroup OOM kill of a process. `docker inspect ... State.OOMKilled=false`
is **not sufficient** to rule out a killed nested Python process.
Do not claim OOM solely because the P3 exit code is -9.

## Validation on Keck Mac / Docker

```bash
cd /Users/obm/Projects/prime-plus/prime-plus
git fetch origin
git switch fix/psfr-ho-vii-memory
uv run pytest -q tests/test_psfr_ho_functional.py
uv run pytest -q tests/test_psfr_spatial_physics_contract.py
```

The Keck PSF Lab Compose service mounts the local
`prime-plus/prime-plus` checkout read-only as `/app/prime-plus`.
Recreate the app container only if needed to reload the mounted
directory; ensure the mounted branch really contains this commit.
No app database migration or scientific config override is required.

Once cgroup state has been recorded and the tests pass, rerun ONE n0013
PSF-R case. Check new `[psfR memory]` stage lines. If the last line is
**before** HO Vii, locate the earlier memory hot spot rather than
assuming this patch resolves it. If it progresses beyond HO,
verify the V1 acceptance report and all WFE/TT/anisoplanatism
numerical invariants.

Do not merge into `refactor/psfr-core` without the above validations.
