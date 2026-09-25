<h1 align="center">Consulting Project on Substance Abuse Analysis</h1>

<!-- Emojis: 🗺️📝📊🧑‍⚕️👩‍⚕️👨‍⚕️🧪🧠🤝🎯👥🧑‍🏫🔗🍺🍻🍾🍷🍸🍹🚬🍃🌿🧪💉💊 -->
<p align="center" size="1">🍺🍻🍾🍷🍸🍹🚬🍃🌿🧪💉💊</p>

<br><hr><details open><summary><h2 style="display:inline-block">🤝 Project partner</h2></summary>

Our project partner is Dr. Sally Olderbak (<olderbak@ift.de>), head of the research group for epidemiology and
diagnostics at the [Institut für Therapieforschung (IFT)](https://ift.de/), which publishes the results of the
German Epidemiological Survey of Substance Abuse (ESA).

</details>

<br><hr><details open><summary><h2 style="display:inline-block">📝 Data</h2></summary>

Our data comes from the Epidemiological Survey of Substance Abuse (ESA) of the years 2018 and 2021. It is
anonymized and cannot be shared by us. The Stata dataset path has to be specified in the relevant notebooks.

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🎯 Research goal</h2></summary>

We want to answer the question:

> Can robust clustering techniques be used to reliably identify *consumer types* among users of substances with
> varying prevalence?

</details>

<br><hr><details open><summary><h2 style="display:inline-block">💻 How to use</h2></summary>

These instructions use Docker. If the results should be reproducible on different system architectures the ARM
docker image should be used. The Compose configuration builds an ARM container. ARM emulation is substantially
slower than native execution on an x86 computer.

1. **Download the repository and add the data**

    The project is available at <https://github.com/LuuisK/esa_consulting.git>.

    ```bash
    git clone https://github.com/LuuisK/esa_consulting.git
    cd esa_consulting
    ```

    The Stata dataset should be put in the `data/raw` directory (otherwise the path in the notebooks has to be
    changed).

2. **Build the Docker environment**

    Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine with the Compose
    plugin), then build the image once. On macOS or Windows:

    ```bash
    docker compose build
    ```

    On Linux, pass the host user and group IDs so generated files have the correct ownership:

    ```bash
    HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose build
    ```

3. **Choose how you want to work**

    - **Run notebooks interactively in VS Code:** Install and enable the [Dev Containers extension][dev-containers]
      in your local VS Code. In the top command bar, type `> Dev Containers: Open Folder in Container...` and select
      this repository. Depending on how the folder was opened, VS Code may instead offer `> Dev Containers: Reopen
      in Container`. It then opens the repository inside the analysis container. Make sure the kernel selected in
      the top-right corner of each notebook uses `/opt/venv/bin/python`.

    - **Run notebooks as scripts from the command line:** ipynb files can be run from the command line with
      `nbconvert`, use the following options to save its cell outputs into the notebook and deactivate timeouts
      of single cells:

      ```bash
      docker compose run --rm analysis \
        jupyter nbconvert --execute --to notebook --inplace \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.record_timing=False workflow/preprocessing.ipynb
      ```

      Replace the final path for a different ipynb file.

      R files can be run from the command line as follows:

      ```bash
      docker compose run --rm analysis Rscript workflow/imputation.R
      ```

      Replace the final path for a different R file.

    - **Run notebooks interactively in the browser (JupyterLab):** Start the server:

      ```bash
      docker compose up
      ```

      Open the `http://127.0.0.1:8888/lab?token=...` URL printed in the terminal. To stop it type `Ctrl+C` in the
      terminal, followed by `docker compose down`. In JupyterLab's Notebook settings, leave **Recording timing**
      disabled to avoid storing execution timestamps in the notebook.

4. **Run the analysis in order**

    Some paths in the notebooks and R scripts may need to be modified depending on the input-file and output
    locations used on your system.

    Approximate runtimes are shown below and depend on the available hardware.

    1. [`workflow/preprocessing.ipynb`](workflow/preprocessing.ipynb) — nearly instant
    2. [`workflow/imputation.R`](workflow/imputation.R) — about 8 min
    3. [`workflow/dimensionality_reduction.R`](workflow/dimensionality_reduction.R) — a few seconds
    4. One or more clustering notebooks — about 100 min each:
       - [`workflow/gmm_clustering.ipynb`](workflow/gmm_clustering.ipynb)
       - [`workflow/kmeans_clustering.ipynb`](workflow/kmeans_clustering.ipynb)
    5. Optional figures and diagnostics in [`create_plots/`](create_plots)

5. **Explore the results**

    Intermediate data is written to `data/preprocessed/`, `data/imputed/`, and `data/dim_red/`. Clustering
    results and figures are written to `outputs/`.

</details>

<br><hr><details open><summary><h2 style="display:inline-block">👥 Project team</h2></summary>

| Role | Name | Contact |
| - | - | - |
| Project partner | Dr. Sally Olderbak | <olderbak@ift.de> |
| Supervisor | Dr. Sabine Hoffmann | <shoffmann@ibe.med.uni-muenchen.de> |
| Consultant | Agnes Baumbach | <a.baumbach@campus.lmu.de> |
| Consultant | Milena Redlingshöfer | <m.redlingshoefer@campus.lmu.de> |
| Consultant | Luis Karrlein | <luis.karrlein@campus.lmu.de> |

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🔗 Links</h2></summary>

| General | |
| - | - |
| Institut für Therapieforschung | <ift@ift.de>, <https://ift.de/> |
| Research group for epidemiology and diagnostics | <Epidemiologie-Diagnostik@ift.de> |
| Epidemiological Survey of Substance Abuse | <https://www.esa-survey.de/en/> |
| Survey designs | <https://www.esa-survey.de/en/study/instruments/> |
| Project link | <https://github.com/LuuisK/esa_consulting> |

| ESA 2018 | |
| - | - |
| Methodology | <https://di.aerzteblatt.de/int/archive/article/209404> |
| Short reports (Kurzberichte) | <https://www.esa-survey.de/ergebnisse/kurzberichte-1/#c5565> |

| ESA 2021 | |
| - | - |
| Methodology | <https://di.aerzteblatt.de/int/archive/article/226333> |
| Short reports (Kurzberichte) | <https://www.esa-survey.de/ergebnisse/kurzberichte-1/#c5575> |

</details>

<br><hr>
[▲ Back to the top](#consulting-project-on-substance-abuse-analysis)

[dev-containers]: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
