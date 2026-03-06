## 1. Create New Directory Structure

- [ ] 1.1 Create `data` directory at the root.
- [ ] 1.2 Create `docs` directory at the root.
- [ ] 1.3 Create `notebooks` directory at the root.
- [ ] 1.4 Move the existing `src` directory contents to a new temporary location.

## 2. Organize Source Files

- [ ] 2.1 Recreate a clean `src` directory.
- [ ] 2.2 Create `__init__.py` inside `src`.
- [ ] 2.3 Move `main.py` into `src`.
- [ ] 2.4 Create `src/article_classification` directory.
- [ ] 2.5 Create `__init__.py` inside `src/article_classification`.
- [ ] 2.6 Move `classifier.py` and `data_transformer.py` into `src/article_classification`.

## 3. Relocate Other Assets

- [ ] 3.1 Move data files from `src/data` to the root `data` directory.
- [ ] 3.2 Move notebooks from `src/notebooks` to the root `notebooks` directory.
- [ ] 3.3 Move PDF from `artigo` to the root `docs` directory.

## 4. Update Code and Configuration

- [ ] 4.1 Review and update all import statements in `.py` files to reflect the new structure.
- [ ] 4.2 Update `.gitignore` to correctly ignore temporary files and directories, and to include the new `data`, `docs`, and `notebooks` directories if they were previously ignored.

## 5. Clean Up

- [ ] 5.1 Remove the now-empty `artigo` directory.
- [ ] 5.2 Remove the now-empty `src/data`, `src/notebooks`, and other leftover directories from the old structure.
- [ ] 5.3 Delete any temporary directories used during the migration.
