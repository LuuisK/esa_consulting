##########################################################################################
################## data preparation ######################################################
##########################################################################################

#load necessary packages
library(haven)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(tidyr)
library(tibble)

# load dataset
data <- read_dta("/Users/aginatore/Desktop/consulting/swisstransfer_f4cb6707-d70b-43a7-a248-4c3cfcd7a7ec/ESA1821_neu_SO.dta")

# create a dataframe with variable names and labels
labels <- sapply(data, function(x) attr(x, "label"))
result <- tibble(
  column_name = names(data),
  label = labels
)

# save this dataframe as a csv
write.csv(result, "all.csv", row.names = FALSE)

##########################################################################################

# group variables

idk_vars         <- names(data)[c(4, 5, 47, 108:109, 117, 156)]
general_vars     <- names(data)[c(1:3, 6)]
socioecon_vars   <- names(data)[c(7:19, 174:175)]
alcohol_vars     <- names(data)[c(20:46, 48:60, 176)]
nicotine_vars    <- names(data)[c(61:73, 177:179)]
other_drugs_vars <- names(data)[c(74:107, 111:116, 118:122)]
medication_vars  <- names(data)[c(123:155, 157:173)]

# Combine groups into a named list
var_groups <- list(
  IDK = idk_vars,
  General = general_vars,
  Socioeconomic = socioecon_vars,
  Alcohol = alcohol_vars,
  Nicotine = nicotine_vars,
  OtherDrugs = other_drugs_vars,
  Medication = medication_vars
)

# Group by group, extract variable names and labels and combine them into a data frame.
# The dataframe contains group membership, variable names and Labels.
var_labels_df <- do.call(rbind, lapply(names(var_groups), function(group) {
  vars <- var_groups[[group]]
  df <- data.frame(
    Group = group,
    Variable = vars,
    Label = sapply(vars, function(var) attr(data[[var]], "label")),
    stringsAsFactors = FALSE
  )
  return(df)
}))

# Remove the rownames
rownames(var_labels_df) <- NULL

# View result
print(var_labels_df)

##########################################################################################

# delete the redundant variables "fall", "alk12mon", "", "", ""
data <- dplyr::select(data, ! c("fall", "alk12mon", ))

