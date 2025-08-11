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
data_file = "data/raw/ESA1821_neu_SO.dta"
data <- read_dta(data_file)

# create a dataframe with variable names and labels
labels <- sapply(data, function(x) attr(x, "label"))
var_labels <- tibble(
  column_name = names(data),
  label = labels
)

# save this dataframe as a csv
write.csv(var_labels, "outputs/var_labels.csv")

names(data)[110]

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

# # delete the redundant variables
# data <- dplyr::select(data, ! c("fall", "alkkat", "alk12", "alk12m", "audit_alt", "alk12mon", 
#                                 "alk12tag", "alpo30", "alpo_tag", "alpo30gr", "alpo12j", "alpo_tax", 
#                                 "alpo12gr", "alpogr", "alpoli", "alpokons", "Shi", "spice_12", 
#                                 "amp_30", "fr12_amp1", "ls_alt", "sds", "fr12_spi1", "am_12"))

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
### Visualisierung der Verteilungen von:

# Frequenz-Menge-Index Alkoholkonsum 
var_labels_de <- c(
  wein30gr = "Frequenz-Menge-Index letzter Monat: Wein/Sekt",
  mish30gr = "Frequenz-Menge-Index letzter Monat: Mischgetränke",
  bier30gr = "Frequenz-Menge-Index letzter Monat: Bier",
  spir30gr = "Frequenz-Menge-Index letzter Monat: Spirituosen",
  alk30gr  = "Frequenz-Menge-Index letzter Monat: Alkohol (gesamt)"
)

data %>%
  select(names(var_labels_de)) %>%
  pivot_longer(cols = everything(), names_to = "Substance", values_to = "Value") %>%
  filter(is.finite(Value)) %>%  
  mutate(
    Substance = factor(Substance, levels = names(var_labels_de), labels = var_labels_de)
  ) %>%
  ggplot(aes(x = Value)) +
  geom_histogram(bins = 30, fill = "coral", color = "black") +
  facet_wrap(~ Substance, scales = "free_x") +
  labs(
    title = "Verteilung der Frequenz-Menge-Index-Variablen",
    x = "Gesamtkonsumintensität",
    y = "Anzahl der Teilnehmenden mit diesem Wert"
  ) +
  theme_minimal()



# 30-Tage Prävalenz Nikotinkonsum
var_labels <- c(
  typ       = "30-Tage Prävalenz Konsum Zigarette/Zigarre/Zigarillo/Pfeife",
  typ_ezig  = "30-Tage Prävalenz E-Zigarette",
  hnb30     = "30-Tage Prävalenz Tabakerhitzer",
  typ_hnb   = "30-Tage Prävalenz Heat-Not-Burn",
  typ_shish = "30-Tage Prävalenz Shisha"
)

data %>%
  select(names(var_labels)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  mutate(
    Value = factor(Value, levels = c(0, 1), labels = c("Nein", "Ja")),
    Variable = factor(Variable, levels = names(var_labels), labels = var_labels)
  ) %>%
  ggplot(aes(x = Value)) +
  geom_bar(fill = "coral", color = "black") +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(
    x = NULL,
    y = "Anzahl"
  ) +
  theme_minimal()



# 12-Monats-Frequenz Drogenkonsum
var_labels <- c(
  fr12_ha1   = "12-Monats-Frequenz Cannabis Gesamt-SP",
  fr12_ec1   = "12-Monats-Frequenz Ecstasy Gesamt-SP",
  fr12_ls1   = "12-Monats-Frequenz LSD Gesamt-SP",
  fr12_ao1   = "12-Monats-Frequenz andere Opiate Gesamt-SP",
  fr12_ko1   = "12-Monats-Frequenz Kokain Gesamt-SP",
  fr12_cr1   = "12-Monats-Frequenz Crack Gesamt-SP",
  fr12_sc1   = "12-Monats-Frequenz Schnüffelstoffe Gesamt-SP",
  fr12_pi1   = "12-Monats-Frequenz Pilze Gesamt-SP",
  fr12_nps1  = "12-Monats-Frequenz Neue Psychoaktive Substanzen Gesamt-SP"
)

data %>%
  select(names(var_labels)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  filter(!is.na(Value)) %>% 
  mutate(
    Value = factor(Value,
                   levels = 1:5,
                   labels = c("Gar nicht", "<1×/Monat", "≥1×/Monat", "≥1×/Woche", "(Fast) täglich")),
    Variable = factor(Variable, levels = names(var_labels), labels = var_labels)
  ) %>%
  ggplot(aes(x = Value)) +
  geom_bar(fill = "coral", color = "black") +
  facet_wrap(~ Variable, scales = "free_x") +
  labs(
    x = NULL,
    y = "Anzahl"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
