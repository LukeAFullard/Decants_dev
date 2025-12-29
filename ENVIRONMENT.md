# Decants: Environment Setup Guide

This guide provides detailed instructions for setting up the development environment for the `Decants` project. This project requires both Python and R, as well as several system dependencies. Following these steps in order is crucial for a successful installation.

## Step 1a: System Dependencies (Linux & macOS)

Before installing any Python or R packages, you must install R itself and the necessary system-level libraries. The `rpy2` Python package and some R packages depend on these.

-   **On Debian/Ubuntu:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y r-base libcurl4-openssl-dev libtirpc-dev
    ```
-   **On Fedora/CentOS/RHEL:**
    ```bash
    sudo dnf install R R-devel libcurl-devel libtirpc-devel
    ```
-   **On macOS (using Homebrew):**
    ```bash
    brew install r
    ```
    *(Homebrew typically handles system dependencies automatically.)*

## Step 1b: System Dependencies (Windows)

Setting up the environment on Windows requires a few manual steps.

1.  **Install R and Rtools:**
    -   Download and install the latest version of R from [CRAN](https://cran.r-project.org/bin/windows/base/).
    -   Download and install **Rtools** from the [Rtools page](https://cran.r-project.org/bin/windows/Rtools/). It is crucial to install the version that corresponds to your R version.
    -   During the Rtools installation, ensure you check the box that says **"Add rtools to system PATH"**.

2.  **Verify PATH:** After installation, open a new Command Prompt or PowerShell and verify that both R and Rtools are in your system's PATH.
    ```powershell
    # Check for R
    R --version
    # Check for Rtools (specifically make)
    make --version
    ```
    If these commands do not work, you will need to add the `bin` directories of your R and Rtools installations to the system PATH environment variable manually.

## Step 2: R Packages

Once R is installed, you need to install the R packages required for validation and comparison tests.

### Required Packages

The following R packages are required:
*   `plyr`
*   `tidyr`
*   `viridis`
*   `NADA`
*   `lubridate`
*   `gam`
*   `ggplot2`
*   `ggpubr`
*   `Icens` (from Bioconductor)

### Manual Installation

You can install the packages manually in an R session.

1.  Start an R interactive session.
    ```bash
    # On Linux/macOS
    sudo R
    # On Windows
    R
    ```

2.  Inside the R session, install the CRAN packages:
    ```R
    install.packages(c('plyr', 'tidyr', 'viridis', 'NADA', 'lubridate', 'gam', 'ggplot2', 'ggpubr'))
    ```

3.  Install `Icens` from Bioconductor:
    ```R
    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    BiocManager::install("Icens")
    ```

4.  Exit the R session by typing `q()` and pressing Enter.

## Step 3: Python Environment

With the system and R dependencies in place, you can now set up the Python environment.

1.  **Create a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install Python dependencies and the `Decants` package:**
    The project's dependencies are defined in `pyproject.toml`. Installing with the `[dev]` optional dependency will also install `rpy2` and testing libraries. This installs the package in "editable" mode, which means you can modify the source code and run it without reinstalling.
    ```bash
    pip install -e .[dev]
    ```

## Step 4: Running Tests

Once the environment is fully set up, you can run the test suite. Because the package was installed in editable mode, you no longer need to manually set the `PYTHONPATH`.

```bash
# Run the test suite
python3 -m pytest tests/
```

## Step 5: Troubleshooting

A common point of failure is the `rpy2` package being unable to locate your R installation.

-   **`rpy2` Installation Fails:** If you see errors related to `R_HOME` or `No such file or directory` during the `pip install` step, it likely means `rpy2` cannot find R.
    -   **Solution 1 (Recommended):** Ensure that the directory containing the R executable is in your system's `PATH`.
    -   **Solution 2 (Manual Override):** You can explicitly set the `R_HOME` environment variable before running `pip install`.
        ```bash
        # Example for Linux
        export R_HOME="/usr/lib/R"
        pip install -e .[dev]
        ```

-   **R Package Timeouts (e.g., `ggpubr`):** Large packages like `ggpubr` may time out during installation. If this happens, try installing its heavy dependencies individually first: `install.packages(c('car', 'rstatix'))`.
