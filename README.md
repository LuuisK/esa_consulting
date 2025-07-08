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

2. **Install dependencies**

    The Python installation and the used package versions are saved in the `pylock.toml` file. Installation ...

    The right R version needs to be installed. It can be read out from the `"R"["Version"]` register in `renv.lock`. (If the tool `rig` is installed it automatically does this and activates the right installation.)

    Then the required R packages can be installed from `renv.lock` using the R package `renv` (installed from the R console using `install.packages("renv")`). In order to download the dependencies, in the R console run:

    ```bash
    renv::restore()
    ```

    <details><summary>A short explanation of `renv`</summary><ul>

    *Taken from* <https://www.rdocumentation.org/packages/renv/versions/1.1.4>*:*

    **Workflow**

    Use `renv::init()` to initialize `renv` in a new or existing project. This will set up a project library, containing all the packages you’re currently using. The packages (and all the metadata needed to reinstall them) are recorded into a lockfile, `renv.lock`, and a `.Rprofile` ensures that the library is used every time you open that project.

    As you continue to work on your project, you will install and upgrade packages, either using `install.packages()` and `update.packages()` or `renv::install()` and `renv::update()`. After you’ve confirmed your code works as expected, use `renv::snapshot()` to record the packages and their sources in the lockfile.

    Later, if you need to share your code with someone else or run your code on a new machine, your collaborator (or you) can call `renv::restore()` to reinstall the specific package versions recorded in the lockfile.

    </ul></details><br>

3. **Run the analysis**

    Follow the instructions in the relevant scripts or notebooks.

    If necessary, export the data from `Stata` as a `dta` file, and save it in the folder named `data`, otherwise specify the path to the data as is detailed in the script you want to run. For confidentiality reasons, this GitHub project does not include the data used for training (ESA 2018 and 2021).

4. **Explore the results**

    Check the generated output files in the output directory (if left unspecified the default is `results`).

</details>

<br><hr><details open><summary><h2 style="display:inline-block">👥 Consultants</h2></summary>

|                         |                                  |
|-------------------------|----------------------------------|
| • Agnes Baumbach:       | <a.baumbach@campus.lmu.de>       |
| • Milena Redlingshöfer: | <m.redlingshoefer@campus.lmu.de> |
| • Luis Karrlein:        | <luis.karrlein@campus.lmu.de>    |

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🧑‍🏫 Supervisor</h2></summary>

- Dr. Sabine Hoffmann

</details>

<br><hr><details open><summary><h2 style="display:inline-block">🔗 Links</h2></summary>

- Institut für Therapieforschung: <ift@ift.de>, <https://ift.de/>
- Research group for epidemiology and diagnostics: <Epidemiologie-Diagnostik@ift.de>
- Epidemiological Survey of Substance Abuse: <https://www.esa-survey.de/en/>
- Survey designs: <https://www.esa-survey.de/en/study/instruments/>
- ESA 2018: ...
- ESA 2021: ...
<!-- ESA 2021 short report?: https://di.aerzteblatt.de/int/archive/article/226333 -->
- Project link: <https://github.com/LuuisK/esa_consulting>

</details>

<br><hr>
[▲ Back to the top](#consulting-project-on-substance-abuse-analysis)
