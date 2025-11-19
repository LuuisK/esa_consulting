<h1 align="center">Consulting Project on Substance Abuse Analysis</h1>

<!-- Emojis: 🗺️📝📊🧑‍⚕️👩‍⚕️👨‍⚕️🧪🧠🤝🎯👥🧑‍🏫🔗🍺🍻🍾🍷🍸🍹🚬🍃🌿🧪💉💊 -->
<p align="center", size="1">🍺🍻🍾🍷🍸🍹🚬🍃🌿🧪💉💊</p>

<br><hr><details open><summary><h2 style="display:inline-block">🤝 Project partner</h2></summary>

Our project partner is Dr. Sally Olderbak (<olderbak@ift.de>), head of the research group for epidemiology and diagnostics at the Institut für Therapieforschung (IFT), which publishes the results of the German Epidemiological Survey of Substance Abuse (ESA).

</details>

<br><hr><details open><summary><h2 style="display:inline-block">📝 Data</h2></summary>

Our data comes from the Epidemiological Survey of Substance Abuse (ESA) of the years 2018 and 2021. It is anonymized and cannot be shared by us.

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🎯 Research goal</h2></summary>

We want to answer the question:
> Can robust clustering techniques be used to reliably identify *consumer types* among users of substances with varying prevalence?

</details>

<br><hr><details open><summary><h2 style="display:inline-block">💻 How to use</h2></summary>

To get started:

1. **Download the repository**

    The project is available at <https://github.com/LuuisK/esa_consulting.git>. It can be cloned, e.g. from the terminal using

    ```bash
    git clone https://github.com/LuuisK/esa_consulting.git
    ```

    Next you need to put your working directory into the project folder, e.g. from the terminal using

    ```bash
    cd esa_consulting
    ```

2. **Install Python dependencies**

    The right Python version and package versions are specified in `uv.lock` and `pyproject.toml`. They need to be installed, e.g. using the tool `uv`. The tool `uv` automatically creates a virtual environment with the necessary packages installed when a script or notebook is run. To make uv install the dependencies without running a file use the following terminal command in the directory `esa_consulting`:

    ```bash
    uv sync
    ```

    Alternatively use `uv run` to run scripts, which automatically installs dependencies:

    ```bash
    uv run <script>
    ```

    <details><summary>Adding or removing dependencies with uv</summary><ul>

    When using a new dependency (a new package), it needs to be added with `uv add <package>`. It is then installed automatically when needed to run a file, or explicitly using `uv sync`.

    Similarly, a dependency can be removed using `uv remove <package>`.

    </ul></details><br>

3. **Install R dependencies**

    The right R version needs to be installed. It can be read out from the `"R"["Version"]` register in `renv.lock`. (If the tool `rig` is installed it automatically does this and activates the right installation.)

    Then the required R packages can be installed from `renv.lock` using the R package `renv` (installed from the R console using `install.packages("renv")`). In order to download the dependencies, run the following command in the R console:

    ```bash
    renv::restore()
    ```

    <details><summary>A short explanation of renv</summary><ul>

    *Taken from* <https://www.rdocumentation.org/packages/renv/versions/1.1.4>*:*

    **Workflow**

    Use `renv::init()` to initialize `renv` in a new or existing project. This will set up a project library, containing all the packages you’re currently using. The packages (and all the metadata needed to reinstall them) are recorded into a lockfile, `renv.lock`, and a `.Rprofile` ensures that the library is used every time you open that project.

    As you continue to work on your project, you will install and upgrade packages, either using `install.packages()` and `update.packages()` or `renv::install()` and `renv::update()`. After you’ve confirmed your code works as expected, use `renv::snapshot()` to record the packages and their sources in the lockfile.

    Later, if you need to share your code with someone else or run your code on a new machine, your collaborator (or you) can call `renv::restore()` to reinstall the specific package versions recorded in the lockfile.

    </ul></details><br>

4. **Run the analysis**

    Follow the instructions in the relevant scripts or notebooks.

    If necessary, export the data from `Stata` as a `dta` file, and save it in the folder named `data/raw`, otherwise specify the path to the data as is detailed in the script you want to run. For confidentiality reasons, this GitHub project does not include the data used for training (ESA 2018 and 2021).

    The order of the scripts is as follows:

    ```bash
    preprocessing.R -> (plot_corr_assoc.R ->) dimensionality_reduction.R -> clustering.R
    ```

    or:

    ```bash
    preprocessing.py -> (plot_corr_assoc.py ->) dimensionality_reduction.py -> clustering.py
    ```

5. **Explore the results**

    Check the generated output files in the output directory (if left unspecified the default is `results`).

</details>

<br><hr><details open><summary><h2 style="display:inline-block">👥 Consultants</h2></summary>

| Names and emails        |                                  |
|-------------------------|----------------------------------|
| • Agnes Baumbach:       | <a.baumbach@campus.lmu.de>       |
| • Milena Redlingshöfer: | <m.redlingshoefer@campus.lmu.de> |
| • Luis Karrlein:        | <luis.karrlein@campus.lmu.de>    |

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🧑‍🏫 Supervisor</h2></summary>

- Dr. Sabine Hoffmann

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🔗 Links</h2></summary>

| General ||
|-|-|
| Institut für Therapieforschung | <ift@ift.de>, <https://ift.de/> |
| Research group for epidemiology and diagnostics | <Epidemiologie-Diagnostik@ift.de> |
| Epidemiological Survey of Substance Abuse | <https://www.esa-survey.de/en/> |
| Survey designs | <https://www.esa-survey.de/en/study/instruments/> |
| Project link: | <https://github.com/LuuisK/esa_consulting> |

| ESA 2018 ||
|-|-|
| Methodology | <https://di.aerzteblatt.de/int/archive/article/209404> |
| Short reports (Kurzberichte) | <https://www.esa-survey.de/ergebnisse/kurzberichte-1/#c5565> |

| ESA 2021 ||
|-|-|
| Methodology | <https://di.aerzteblatt.de/int/archive/article/226333> |
| Short reports (Kurzberichte) | <https://www.esa-survey.de/ergebnisse/kurzberichte-1/#c5575> |

</details>

<br><hr>
[▲ Back to the top](#consulting-project-on-substance-abuse-analysis)
