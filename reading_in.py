import os
import re

import pandas as pd

from utils import check_save_path

# Hardcoded paths, change to apply to new data
data_folder = "data/raw"
data_path = os.path.join(data_folder, "ESA1821_neu_SO.dta")
logfile_path = os.path.join(data_folder, "codebook_output.txt")
save_folder = "data/readin"
outputs_folder = "outputs"

allow_overwrite = True


def process_logfile(logfile_path: str) -> pd.DataFrame:
    """
    Process the Stata logfile, as it is output by R.
    """
    # Read the logfile
    with open(logfile_path, "r", encoding="utf-8") as f:
        stata_logfile = f.read()

    # Extract information about each variable using regex
    regex_header = r"\{txt\}\{hline\}\n\{res\}([^\{]*?)\{right:([^\}]*?)\}\n\{txt\}\{hline\}"
    regex_footer = r"\n\n(?:\{txt\}\{hline\}|\{com\})"
    var_matches: list[tuple[str, str, str]] = re.findall(
        rf"{regex_header}(.*?)(?={regex_footer})", stata_logfile, flags=re.DOTALL
    )

    # Store information about variables in a dataframe
    esa_vars_info = pd.DataFrame(columns=["var", "desc", "info"])
    for var_name, var_desc, var_body in var_matches:
        var_info = f"== {var_name} =={{col 26}}" + var_desc + var_body

        # Replace placeholders, e.g. {col 26}, with indentation
        indented_lines: list[str] = []
        for line in var_info.split("\n"):
            current_line = ""
            parts = re.split(r"(\{col \d+\})", line)
            for part in parts:
                indent_match = re.match(r"\{col (\d+)\}", part)
                if indent_match is not None:
                    current_line += " " * (int(indent_match.group(1)) - 1 - len(current_line))
                else:
                    current_line += re.sub(r"\{([^\}]+?)(:([^\}]+))?\}", r"\3", part)
            indented_lines.append(current_line)
        var_info = "\n".join(indented_lines)

        # Append information about variable as a row to the dataframe
        esa_vars_info.loc[len(esa_vars_info)] = [var_name, var_desc, var_info]

    # Return the dataframe
    esa_vars_info = esa_vars_info.set_index("var")
    return esa_vars_info


if __name__ == "__main__":
    # Read in data and information about variables
    esa_2018_2021 = pd.read_stata(data_path)
    esa_2018_2021 = esa_2018_2021.set_index("fall")
    esa_2018 = esa_2018_2021.query("zeit == 2018")
    esa_2021 = esa_2018_2021.query("zeit == 2021")
    esa_vars_info = process_logfile(logfile_path)

    # Save processed data and information about variables
    save_path = os.path.join(save_folder, "esa_2018_2021.csv")
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    esa_2018_2021.to_csv(save_path)
    print(f"Saved full dataset as CSV to {save_path}.")

    save_path = os.path.join(save_folder, "esa_2018.csv")
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    esa_2018.to_csv(save_path)
    print(f"Saved 2018 dataset as CSV to {save_path}.")

    save_path = os.path.join(save_folder, "esa_2021.csv")
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    esa_2021.to_csv(save_path)
    print(f"Saved 2021 dataset as CSV to {save_path}.")

    save_path = os.path.join(save_folder, "esa_vars_info.csv")
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    esa_vars_info.to_csv(save_path)
    print(f"Saved information about variables as CSV to {save_path}.")

    save_path = os.path.join(outputs_folder, "esa_vars_info.txt")
    check_save_path(save_path, allow_overwrite=allow_overwrite)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n\n\n\n".join(esa_vars_info["info"]))
    print(f"Saved information about variables as a readable TXT file to {save_path}.")
