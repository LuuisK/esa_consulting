##########################################################################################
################## 1. Data Preparation ###################################################
##########################################################################################

#load necessary packages
library(haven)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(tidyr)
library(tibble)
library(corrplot)
library(mice)
library(psych)

# load the dataset
data_file = "data/raw/ESA1821_neu_SO.dta"
data <- read_dta(data_file)

# create a dataframe with variable names and labels
# labels <- sapply(data, function(x) attr(x, "label"))
# var_labels <- tibble(
#  column_name = names(data),
#  label = labels
#)
# save this dataframe as a csv
# write.csv(var_labels, "outputs/var_labels.csv")


# identify the columns where all values are NA
na_only_cols <- sapply(data, function(col) all(is.na(col)))

# show the names of those columns
names(na_only_cols[na_only_cols])

# and remove them from the dataset
data <- dplyr::select(data, ! c("fall", "alkkat", "alk12", "alk12m", "audit_alt", "alk12mon", 
                                "alk12tag", "alpo30", "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax", 
                                "alpo12gr", "alpogr", "alpoli", "alpokons", "Shi", "spice_12", 
                                "amp_30", "fr12_amp1", "ls_alt", "sds", "fr12_spi1", "am_12"))

# build groups
idk_vars         <- names(data)[c(47, 108:109, 117, 156)]
general_vars     <- names(data)[c(1:5)]
socioecon_vars   <- names(data)[c(6:18, 150, 151)]
alcohol_vars     <- names(data)[c(19:32, 34:44, 152)]
nicotine_vars    <- names(data)[c(45:56,153:155)]
other_drugs_vars <- names(data)[c(57:99)]
medication_vars  <- names(data)[c(100:149)]

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
grouped_vars <- do.call(rbind, lapply(names(var_groups), function(group) {
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
rownames(grouped_vars) <- NULL
write.csv(grouped_vars, "outputs/grouped_vars.csv")


##########################################################################################
################## 2. Correlations #######################################################
##########################################################################################

# Correlation Matrices:
# Pearsons correlation coefficient captures linear relationships, but we have survey data.
# So, our variables are ordinal and the relationship between them is not linear.
# Using Spearman's rank correlation coefficient makes sense because it measures monotonic relationships.

# Create Correlation Matrices for groups of substances:
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

# Get indices where the absolute correlation is > 0.8 and not on the diagonal
strong_corr_indices <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)

# Create a dataframe of those pairs
strong_correlations <- data.frame(
  var1 = rownames(cor_matrix)[strong_corr_indices[, 1]],
  var2 = colnames(cor_matrix)[strong_corr_indices[, 2]],
  correlation = cor_matrix[strong_corr_indices]
)

# Remove duplicate pairs (since correlation matrix is symmetric)
strong_correlations <- strong_correlations[strong_correlations$var1 < strong_correlations$var2, ]

# View result
print(strong_correlations)

# If two variables are very strongly correlated (correlation > 0.8 or < -0.8),
# then they basically measure very similar information. This leads to redundancy in the data set and
# possible distortions in the analysis (e.g. clusters are more strongly influenced by duplicate information)
# We have to decide on one of the two variables in each case
# How to select variable?
# We could check for number of data points, use Variables with more observations


##########################################################################################
################## 3. Factor Analysis ####################################################
##########################################################################################


alcohol_data <- data[, alcohol_vars]
# convert into numeric form, so they can be used in factor analysis
alcohol_data_num <- alcohol_data %>%
  mutate(across(everything(), ~ as.numeric(.)))

# Imputation
imputed <- mice(alcohol_data_num, m = 5, method = "pmm", seed = 102)
# m = 5 means 5 imputed datasets are created, because imputing missing values involves some uncertainty.
# If you only create 1 dataset, you treat the imputed values as if they were the truth, but they’re really just plausible guesses.
# pmm (predictive mean matching) nutzt alle anderen Variablen im Datensatz, um fehlende Werte vorherzusagen.
# Die Vorhersage basiert implizit auf der linearen Beziehung (also Korrelation) zwischen Variablen.
# Je stärker zwei Variablen korrelieren, desto stärker beeinflusst eine die Imputation der anderen.

# Einen vollständigen Datensatz extrahieren
complete_data <- complete(imputed, action = 1)

summary(complete_data)

# Faktorenanalyse
# Bestimmen der Anzahl der Faktoren (z.B. Scree-Test)
fa.parallel(complete_data, fa="fa", n.iter=100)
# Die ersten 3 Faktoren liegen deutlich über der roten Linie.
# Danach fallen die blauen Dreiecke unter die rote Linie 
# -> diese Faktoren erklären nicht mehr mehr Varianz als Zufall.
# Es ist sinnvoll, die ersten 3 Faktoren zu behalten.
# Diese Faktoren fassen die wichtigste gemeinsame Varianz deiner Alkoholvariablen zusammen.

# Scree-Test schlägt 3 Faktoren vor
# using oblimin rotation, which is an oblique rotation method
# it allows factors to be correlated
fa_result <- fa(complete_data, nfactors = 3, rotate = "oblimin") 
print(fa_result)

# Faktorwerte extrahieren (diese können fürs Clustering verwendet werden)
# $scores contains the factor scores for each observation (i.e., how strongly each individual respondent loads on each factor).
factor_scores <- fa_result$scores
head(factor_scores)



general_data <- data[, general_vars]
general_data_num <- general_data %>%
  mutate(across(everything(), ~ as.numeric(.)))
imputed <- mice(general_data_num, m = 5, method = "pmm", seed = 102)
complete_data <- complete(imputed, action = 1)
summary(complete_data)
fa.parallel(complete_data, fa="fa", n.iter=100)
fa_result <- fa(complete_data, nfactors = 3, rotate = "oblimin") 
print(fa_result)
factor_scores <- fa_result$scores
head(factor_scores)


################## Funktion: Imputation → Faktorenanalyse → Poolen → Scores fürs Clustering

# var_groups is a named list of all groups (see 1. Data Preparation)

run_factor_pipeline <- function(data, var_groups, nfactors_per_group, m = 5, seed = 102) {
  
  set.seed(seed)
  
  # 1. Imputation
  # Wir imputieren EINMAL über ALLE Variablen (damit Gruppen konsistent sind)
  imputed <- mice(data[, unlist(var_groups)], m = m, method = "pmm", seed = seed)
  
  # 2. Für jede Gruppe Faktorenanalyse durchführen
  results <- lapply(names(var_groups), function(group) {
    vars <- var_groups[[group]]
    
    # Liste der Scores aus allen imputierten Datensätzen
    group_scores_list <- list()
    
    for (i in 1:m) {
      complete_data <- complete(imputed, action = i)
      subset_data <- complete_data[, vars]
      subset_data <- subset_data %>% mutate(across(everything(), as.numeric))
      
      fa_res <- fa(subset_data, nfactors = nfactors_per_group[[group]], rotate = "oblimin")
      
      group_scores_list[[i]] <- fa_res$scores
    }
    
    # 3. Manuelles Poolen der Scores
    pooled_scores <- Reduce("+", group_scores_list) / m
    
    # Dataframe mit Gruppennamen in den Spalten
    colnames(pooled_scores) <- paste0(group, "_F", 1:ncol(pooled_scores))
    
    return(pooled_scores)
  })
  
  # 4. Alles zusammenführen
  final_scores <- do.call(cbind, results)
  return(final_scores)
}



run_factor_pipeline(data, var_groups, nfactors_per_group)

