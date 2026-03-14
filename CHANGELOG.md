# Changelog
All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-14
### Added
- Initial working version based on the original [pHierCC](https://github.com/zheminzhou/pHierCC) by Zhou et al.
- Modified SciPy's `hierarchy.py` and `_hierarchy.pyx` to accept `np.int16` distance matrices, reducing RAM usage from `float64` (~8x saving) during hierarchical clustering.
- Replaced multiprocessing Pool + SharedArray distance computation with Numba `prange` thread parallelism and TBB work-stealing scheduler.
- Incremental distance matrix expansion: reuse previous run's `dist0.npy`, `dist1.npy`, and `ordering.npy` to avoid full recalculation when new STs are appended.
- Support for mixed numeric and text-based ST identifiers (e.g. public + `local_` profiles), with local STs always sorted to the bottom of the distance matrix.
- `--clean` flag to force full recalculation even when previous run artefacts exist.
- Dockerized build with custom SciPy compilation for `int16` clustering support.
- Weekly clustering wrapper script (`plepiseq_bin/run_clustering.sh`) for Salmonella, Escherichia, and Campylobacter.
