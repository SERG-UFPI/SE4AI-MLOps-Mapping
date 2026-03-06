## Context

The current project structure lacks organization, making it difficult to maintain and scale. The `proposal.md` outlines the need for a more logical and standardized folder layout to improve developer experience and project clarity. This design document details the specifics of the new structure and the plan for migrating to it.

## Goals / Non-Goals

**Goals:**

-   Define a new, clean, and intuitive folder structure for the project.
-   Provide a clear, step-by-step migration plan to transition from the old structure to the new one.
-   Ensure all code remains functional after the refactoring by updating import paths.
-   Improve project navigability and maintainability.

**Non-Goals:**

-   This redesign will not introduce any new features or functionalities.
-   The core logic of the application and its components will not be altered.
-   No changes will be made to the CI/CD pipeline, although it may need to be updated separately.

## Decisions

The new folder structure is designed for clarity and separation of concerns, following common Python project conventions.

### Proposed Folder Structure

```
/
├── data/
│   ├── articles.json
│   └── Artigos IEE Xplore.csv
├── docs/
│   └── SE4AI_MLOps_Mapeamento.pdf
├── notebooks/
│   └── SLR_ChatGPT.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── article_classification/
│       ├── __init__.py
│       ├── classifier.py
│       └── data_transformer.py
├── .gitignore
├── README.md
└── venv/
```

### Key Changes:

1.  **`data/`**: A top-level directory to store all raw data files, such as `articles.json` and `Artigos IEE Xplore.csv`.
2.  **`docs/`**: A new directory for all project documentation, including the `SE4AI_MLOps_Mapeamento.pdf` article.
3.  **`notebooks/`**: All Jupyter notebooks, such as `SLR_ChatGPT.ipynb`, will be moved here.
4.  **`src/`**: This directory will house all the Python source code.
    -   `src/main.py`: The main entry point of the application.
    -   `src/article_classification/`: The sub-package for the classification module.
    -   `__init__.py`: An empty `__init__.py` file will be added to `src/` and `src/article_classification/` to mark them as Python packages, which is crucial for correct import resolution.

## Risks / Trade-offs

-   **Risk**: Broken imports after moving files.
    -   **Mitigation**: All Python files will be scanned, and their import statements will be carefully updated to reflect the new `src`-based structure. For example, an import like `from article_classification.classifier import ...` in `main.py` will remain the same as long as the project root is in the `PYTHONPATH`, but might need to be adjusted depending on how the application is run. A `try-except` block could be used to handle `ImportError` and provide more informative error messages during the transition.
-   **Risk**: Scripts or tools that rely on the old file paths might fail.
    -   **Mitigation**: A project-wide search for any hardcoded paths will be performed. The `README.md` will be updated with the new instructions on how to run the project.
-   **Trade-off**: The one-time effort of refactoring will add a small overhead initially but will pay off in long-term project health and developer productivity.
