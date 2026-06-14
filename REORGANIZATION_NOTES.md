# Reorganized Project Notes

This folder was rebuilt from a flattened `mmrotate-CDM-main_no_pth_no_txt` directory.
The original source mixed images, XML annotations, JSON outputs, Python modules, tests, docs, and cached bytecode in one level.
This reorganization follows the top-level style of `Sparse-R-CNN-OBB-main` as closely as possible without guessing original nested package paths.

## Main Directories

- `configs`: model configs and metafiles.
- `data`: images, XML annotations, JSON records, and logs.
- `docs`: markdown and rst documentation, plus docs build helpers.
- `mmrotate_flat`: Python source files that could not be safely mapped back to their exact original package path.
- `projects`: experiment configs and custom project scripts.
- `tests`: test files.
- `tools`: runnable utility scripts such as train, benchmark, and analysis scripts.
- `artifacts`: archives and bytecode caches.

## File Counts

- `artifacts`: 259 files
- `configs`: 572 files
- `data`: 24498 files
- `docs`: 127 files
- `mmrotate_flat`: 184 files
- `notebooks`: 2 files
- `projects`: 63 files
- `tests`: 51 files
- `tools`: 24 files

## Important Limitation

Many files in the original folder had naming collisions and were already flattened into names like `README_12.md`, `__init___18.py`, and `config_31.py`.
So this organized version is structured and browsable, but it is not guaranteed to be directly runnable as a faithful source checkout.
