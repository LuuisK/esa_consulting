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
library(corrplot)

# load dataset
data <- read_dta("/Users/aginatore/Desktop/consulting/swisstransfer_f4cb6707-d70b-43a7-a248-4c3cfcd7a7ec/ESA1821_neu_SO.dta")

# create a dataframe with variable names and labels
labels <- sapply(data, function(x) attr(x, "label"))
result <- tibble(
  column_name = names(data),
  label = labels
)

# save this dataframe as a csv
write.csv(result, "all.csv")

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

write.csv(var_labels_df, "groups.csv")


##########################################################################################
################## Variable Selection ####################################################
##########################################################################################


# Correlation Matrices: Pearson or Spearman?
# The Pearsons correlation coefficient captures linear relationships, but we have survey data.
# So, our variables are ordinal and the relationship between them is not linear.
# Using Spearman's rank correlation coefficient makes sense because it measures monotonic relationships.

# Since columns with too many missing values yield a question mark in the matrix, delete those variables.
# identify columns where all values are NA
na_only_cols <- sapply(data, function(col) all(is.na(col)))

# show names of those columns
names(na_only_cols[na_only_cols])

# "alkkat"    "alk12"     "alk12m"    "audit_alt" "alk12mon"  "alk12tag"  "alpo30"   
# "alpo_tag"  "alpo30gr"  "alpo12j"   "alpo_tax"  "alpo12gr"  "alpogr"    "alpoli"   
# "alpokons"  "Shi"       "spice_12"  "amp_30"    "fr12_amp1" "ls_alt"    "sds"      
# "fr12_spi1" "am_12" 

# which(names(data) == "")

# delete the redundant variables
data <- dplyr::select(data, ! c("fall", "alkkat", "alk12", "alk12m", "audit_alt", "alk12mon", 
                                "alk12tag", "alpo30", "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax", 
                                "alpo12gr", "alpogr", "alpoli", "alpokons", "Shi", "spice_12", 
                                "amp_30", "fr12_amp1", "ls_alt", "sds", "fr12_spi1", "am_12"))

# new groups without deleted Variables

alcohol_vars     <- names(data)[c(20:32, 34:42, 45, 46, 57, 60, 176)]
nicotine_vars    <- names(data)[c(61, 62, 64:73, 177:179)]
other_drugs_vars <- names(data)[c(74:78, 89:92, 94:106, 111:116, 119:122)]
medication_vars  <- names(data)[c(123:155, 157:173)]


# Create Correlation Matrices with the new groups
alcohol_data <- data[, alcohol_vars]
cor_matrix <- cor(alcohol_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6)
title(main = "Correlation Matrix of  Alcohol Variables", line = 2, cex.main = 1.5)
         

nicotine_data <- data[, nicotine_vars]
cor_matrix <- cor(nicotine_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6)
title(main = "Correlation Matrix of  Nicotine Variables", line = 2, cex.main = 1.5)
         

other_drugs_data <- data[, other_drugs_vars]
cor_matrix <- cor(other_drugs_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6)
title(main = "Correlation Matrix of  Drug Variables", line = 2, cex.main = 1.5)


medication_data <- data[, medication_vars]
cor_matrix <- cor(medication_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.6) 
title(main = "Correlation Matrix of Medication Variables", line = 2, cex.main = 1.5)


# If two variables are very strongly correlated (correlation > 0.8 or < -0.8),
# then they basically measure very similar information. This leads to redundancy in the data set and
# possible distortions in the analysis (e.g. clusters are more strongly influenced by duplicate information)
# We have to decide on one of the two variables in each case
# How to select variable?
# We could check for number of data points, use Variables with more observations

##########################################################################################

# does it make sense to summarize e.g. alcohol use in one variable?
# data$alcohol <- wolf_data$var1 + wolf_data$var2 + wolf_data$var3
