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

    The right Python version and package versions are specified in `uv.lock` and `pyproject.toml`. They need to be installed, e.g. using the tool `uv`. The tool `uv` automatically creates a virtual environment (named `.venv`) with the necessary packages installed when a script or notebook is run. To make uv install the dependencies use the following terminal command in the directory `esa_consulting`:

    ```bash
    uv sync
    ```

    To explicitly switch into the virtual environment, use the terminal command:

    ```bash
    source .venv/bin/activate
    ```

    <details><summary>Adding or removing dependencies with uv</summary><ul>

    When using a new dependency (a new package), it needs to be added with `uv add <package>`. It is then installed automatically when needed to run a file, or explicitly using `uv sync`.

    Similarly, a dependency can be removed using `uv remove <package>`.

    </ul></details><br>

3. **Run the analysis**

    Follow the instructions in the relevant notebooks.

    If necessary, export the data from `Stata` as a `dta` file, and save it in the folder named `data/raw`, otherwise specify the path to the data as is detailed in the notebook you want to run. For confidentiality reasons, this GitHub project does not include the data used for training (ESA 2018 and 2021 surveys).

    The order of the notebooks is as follows:

    ```bash
    preprocessing.ipynb -> (plot_corr_assoc.ipynb ->) dimensionality_reduction.ipynb -> clustering.ipynb
    ```

4. **Explore the results**

    Check the generated output files in the output directory (the default is `results`).

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
