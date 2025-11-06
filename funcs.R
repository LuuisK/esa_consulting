#' Script defining functions.

# Load required packages
library(readr)
library(checkmate)
library(tools)
library(dplyr)
library(ggplot2)
library(tidyr)
library(utils)
library(haven)


#' "Not In" operator.
`%not_in%` = Negate(`%in%`)


#' Empty a folder.
#'
#' Empties the given folder by deleting every file and folder contained in it.
empty_folder <- function(folder) {
  assert_string(folder)

  # Get contents and delete them
  files <- list.files(
    folder,
    include.dirs = TRUE,
    full.names = TRUE,
    recursive = TRUE
  )
  file.remove(files)

  # Return silently
  return(invisible(NULL))
}


#' Save a data frame as an rds-file.
#'
#' Saves the passed data with a given file name and prints it with the label.
save_data <- function(data, save_path) {
  assert_data_frame(data)
  assert_string(save_path)

  # Save as an rds-file and print message
  ext <- file_ext(save_path)
  assert_string(ext, pattern = "rds", fixed = NULL)
  write_rds(data, file = save_path)
  print(sprintf(
    "Saved %s as an %s file to %s.",
    deparse(substitute(v1)),
    ext,
    save_path
  ))
}


# TODO: maybe check_save_path
# TODO: maybe second save function for other objects

#' Creates a dataframe containing information about variables.
create_vars_info <- function(data) {
  assert_data_frame(data)

  # Extract variable and value labels
  var_labels <- vapply(
    data,
    function(x) attr(x, "label"),
    FUN.VALUE = character(1)
  )
  value_labels <- lapply(data, function(x) {
    label_map <- attr(x, "labels")
    # Invert named vector
    setNames(names(label_map), label_map)
  })

  # Create data frame with variable descriptions and value labels
  vars_info <- tibble(
    name = names(var_labels),
    description = unname(var_labels),
    labels = unname(value_labels)
  )

  return(vars_info)
}

#' Sort variables into groups.
#'
#' Returns the input data frame of information about variables with an added
#' grouping of the variables (miscellaneous, general, socioeconomic, alcohol,
#' nicotine, other drugs, medication).
sort_groups_vars_info <- function(vars_info, groups_by_name) {
  assert_data_frame(vars_info)
  assert_list(groups_by_name, names = "named")

  # Return a modified copy of the input
  vars_info = data.frame(vars_info)

  # Check that there are no multiply assigned variables
  all_grouped_vars <- unlist(groups_by_name, use.names = FALSE)
  duplicates <- (table(all_grouped_vars) != 1) %>% which %>% names
  if (length(duplicates) != 0) {
    sprintf(
      "Some variables have multiple group membership: [%s].",
      toString(duplicates)
    ) %>%
      stop
  }

  # Display variables that are not assigned a group
  not_assigned <- setdiff(vars_info %>% pull(name), all_grouped_vars)
  if (length(not_assigned) != 0) {
    sprintf(
      "Some variables are not assigned to a group: [%s].",
      toString(not_assigned)
    ) %>%
      print
  }

  # Assign groups to variables
  for (group_name in names(groups_by_name)) {
    group_vars <- groups_by_name[[group_name]]
    mask <- vars_info$name %in% group_vars
    vars_info[mask, "group"] <- group_name
  }

  return(vars_info)
}

#' Sort variables by type.
#'
#' Returns the input data frame of information about variables with an added
#' sorting by variable type (all_na, categorical, continuous, or no type (NA)).
#'
#' The type of a variable is one of: all_na, categorical, continuous, or NA in case of no type assignment.
sort_types_vars_info <- function(vars_info, types_by_name) {
  assert_data_frame(vars_info)
  assert_list(types_by_name, names = "named")

  # Return a modified copy of the input
  vars_info = data.frame(vars_info)

  # Check that there are no multiply assigned variables
  all_typed_vars <- unlist(types_by_name, use.names = FALSE)
  duplicates <- (table(all_typed_vars) != 1) %>% which %>% names
  if (length(duplicates) != 0) {
    sprintf(
      "Some variables have several types: [%s].",
      toString(duplicates)
    ) %>%
      stop
  }

  # Display variables that are not assigned a type
  not_assigned <- setdiff(vars_info %>% pull(name), all_typed_vars)
  if (length(not_assigned) != 0) {
    sprintf(
      "Some variables are not assigned to a type: [%s].",
      toString(not_assigned)
    ) %>%
      print
  }

  # Assign types to variables
  for (type_name in names(types_by_name)) {
    type_vars <- types_by_name[[type_name]]
    mask <- vars_info$name %in% type_vars
    vars_info[mask, "type"] <- type_name
  }

  return(vars_info)
}

#' Convert a numeric column to a factor
convert_col_to_factor <- function(x, ordered = FALSE) {
  assert_logical(ordered, len = 1)

  # Convert with different function calls, based on if it is labelled or not
  if (is.labelled(x)) {
    as_factor(x, levels = "values", ordered = ordered)
  } else {
    if (ordered) as.ordered(x) else as.factor(x)
  }
}

#' Preprocesses the data.
#'
#' - Converts factorial data to have categorical data type and saves the original levels in vars_info.
#' - Removes unimportant features.
preprocess_data <- function(
  data,
  vars_info,
  remove_all_na_vars = TRUE,
  drop_vars = character(0)
) {
  assert_data_frame(data)
  assert_data_frame(vars_info)
  assert_logical(remove_all_na_vars, len = 1)
  assert_character(drop_vars)

  # Return a modified copy of the input
  data <- data.frame(data)
  vars_info <- data.frame(vars_info)

  # Convert categorical variables to factors
  data <- data %>%
    mutate(
      across(
        vars_info %>% filter(type %in% c("binary", "nominal")) %>% pull(name),
        \(x) convert_col_to_factor(x, ordered = FALSE)
      ),
      across(
        vars_info %>% filter(type == "ordinal") %>% pull(name),
        \(x) convert_col_to_factor(x, ordered = TRUE)
      )
    )

  # # Recode binary to 0-1 in data and in vars_info["labels"]
  # binary_vars: pd.Series[pd.StringDtype] = vars_info.loc[vars_info["type"] == "binary", "variable"]
  # not_0_1 = [var for var in binary_vars if set(data[var].cat.categories) != set((0, 1))]
  # for var in not_0_1:
  #     old_cats = data[var].cat.categories
  #     new_cats = [0, 1]
  #     old_value_dict = vars_info.loc[vars_info["variable"] == var, "labels"].item()
  #     new_value_dict = {x: old_value_dict.get(old_cats[x], old_cats[x]) for x in new_cats}
  #     data[var] = data[var].cat.rename_categories(new_cats)
  #     vars_info.loc[vars_info["variable"] == var, "labels"] = [new_value_dict]

  if (remove_all_na_vars) {
    # Remove features that are NA everywhere from the dataset and vars_info
    all_na_vars <- vars_info %>% filter(type == "all_na") %>% pull(name)
    data <- data %>% select(-all_of(all_na_vars))
    sprintf(
      "Dropped the following variables from the dataset because they are missing everywhere: [%s]",
      toString(all_na_vars)
    ) %>%
      print
    vars_info <- vars_info %>% filter(name %not_in% all_na_vars)
    # TODO:? vars_info["type"] = vars_info["type"].cat.remove_categories("all_na")
  }

  # Remove specified unimportant variables
  data <- data %>% select(-all_of(drop_vars))
  vars_info <- vars_info %>% filter(name %not_in% drop_vars)

  # Return data and vars_info
  output <- list(
    data = data,
    vars_info = vars_info
  )

  return(output)
}
