## Why

The current project folder structure is disorganized, making it difficult to navigate, understand the project, and locate files efficiently. A well-defined and logical folder structure will significantly improve the project's maintainability, scalability, and ease of onboarding for new team members.

## What Changes

- Reorganize the source code into a more logical and standard structure for Python projects.
- Move all source code to a dedicated `src` directory.
- Relocate data files to a `data` directory at the root level.
- Move Jupyter notebooks to a `notebooks` directory.
- Move article files to a `docs` directory.
- Adjust import paths in the code to reflect the new structure.

## Capabilities

### New Capabilities

- None

### Modified Capabilities

- None

## Impact

- **Code**: All Python files in the `src` directory will be moved and may require their import statements to be updated.
- **Data**: The `articles.json` and `Artigos IEE Xplore.csv` files will be moved.
- **Notebooks**: The `SLR_ChatGPT.ipynb` notebook will be moved.
- **Documentation**: The `artigo/SE4AI_MLOps_Mapeamento.pdf` file will be moved.
- **Configuration**: The `.gitignore` file may need to be updated to reflect the new structure.
