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

# load the dataset
data_file = "data/raw/ESA1821_neu_SO.dta"
data <- read_dta(data_file)


# calculate proportion of missing values per variable
missing_prop <- colMeans(is.na(data))

# select variables with missingness >= 30%
high_missingness <- missing_prop[missing_prop >= 0.3]

# create data frame with counts and proportions
missing_df <- data.frame(
  Variable = names(high_missingness),
  NAs = colSums(is.na(data))[names(high_missingness)],
  Proportion = high_missingness
)

# sort by proportion descending
missing_df <- missing_df[order(-missing_df$Proportion), ]



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
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 1)
title(main = "Correlation Matrix of  Alcohol Variables", line = 2, cex.main = 1.5)

nicotine_data <- data[, nicotine_vars]
cor_matrix <- cor(nicotine_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 1)
title(main = "Correlation Matrix of  Nicotine Variables", line = 2, cex.main = 1.5)

other_drugs_data <- data[, other_drugs_vars]
cor_matrix <- cor(other_drugs_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)
title(main = "Correlation Matrix of  Drug Variables", line = 2, cex.main = 1.5)

medication_data <- data[, medication_vars]
cor_matrix <- cor(medication_data, method = "spearman", use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8) 
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
library(mice)

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


##########################################################################################
################## 3.1 Factor Analysis Mixed Data ########################################
##########################################################################################
library(FactoMineR)
library(missMDA)  # for imputation
library(factoextra)
library(plotly)

# Numeric variables
numeric_vars <- c(
  "alk30", "spir30", "mish30", "wein30gr", "mish30gr", "bier30gr", 
  "spir30gr", "alk30gr", "alk12j", "alk12gr", "alkgr", "alkjahr"
)

# Ordinal / categorical variables
ordinal_vars <- c(
  "alk30kat", "alkkons", "binge", "bierkons", "spirkons", "mishkons", 
  "hazar", "harmf", "alkrisk", "dsm_al", "dsm_al_m", "alk_kat", 
  "Miss_Abh", "vx210_2"
)

# Convert to appropriate type
alcohol_data_famd <- alcohol_data %>%
  mutate(across(all_of(numeric_vars), as.numeric)) %>%
  mutate(across(all_of(ordinal_vars), ~ factor(.x, ordered = TRUE)))

# Impute missing data for mixed variables
alcohol_data_imputed <- imputeFAMD(alcohol_data_famd, ncp = 10)  # ncp = number of components to retain

# Run FAMD
famd_res <- FAMD(alcohol_data_imputed$completeObs, ncp = 10, graph = TRUE)
# Each arrow represents one numeric variable
# The direction of the arrow shows how the variable is related to the dimensions (principal components).
# The length of the arrow shows how well that variable is represented by the first two dimensions (Dim 1 and Dim 2). 
# Longer = better representation.
# Interpretation:
# Variables pointing in the same direction are positively correlated.
# Variables pointing in opposite directions are negatively correlated.
# Variables that are nearly orthogonal (90° apart) are uncorrelated.


famd_res$eig[1:3, "eigenvalue"]
# Variance explained for the first three dimensions
famd_res$eig[1:3, "percentage of variance"]
famd_res$eig[1:3, "cumulative percentage of variance"]
eig.val <- get_eigenvalue(famd_res)
head(eig.val)


# Factor loadings of the numeric variables
# loadings for the first three dimensions
round(famd_res$var$coord[, 1:3], 3)
# Factor loadings of the categorial variables
# loadings for the first three dimensions
round(famd_res$quali$coord[, 1:3], 3)

fviz_screeplot(famd_res)

# Wir nehmen die ersten Dimensionen (z. B. 1:5 oder 1:10, je nach Screeplot)
X <- factor_scores[, 1:3]

# K-means Clustering für verschiedene Clusterzahlen berechnen
set.seed(102)  # Reproduzierbarkeit
fviz_nbclust(X, kmeans, method = "wss") + 
  ggtitle("Knee Plot - Bestimmung der Clusteranzahl (WSS)")


plot(famd_res, choix = "quali")

# Factor scores for clustering
factor_scores <- famd_res$ind$coord
head(factor_scores)

# visualize variables contribution
fviz_famd_var(famd_res, repel = TRUE)
# Punkte im Plot:
# Jeder Punkt entspricht einer Variablen
# Rote Pfeile zeigen die Richtung, in der Variablen auf die Achsen wirken.
#
# Nähe zwischen Variablen:
# Variablen, die nah beieinander liegen, sind korreliert oder haben ein ähnliches Antwortmuster.
# z. B. alkrisk, alkkons, alk30kat liegen eng zusammen, d.h. sie messen ähnliche Dimensionen des Alkoholkonsums/-risikos.
# Variablen, die weit auseinanderliegen, unterscheiden sich stark.
#
# Distanz vom Ursprung (0,0):
# Je weiter eine Variable vom Ursprung entfernt ist, desto stärker trägt sie zur Erklärung der Dimension bei.
# z. B. alkrisk, alkkons, alk30kat und alk30gr liegen weit weg, d.h. sie sind besonders erklärungsstark.
# Variablen wie dsm_al_m oder dsn_al_m liegen sehr nah am Ursprung, d.h. sie haben in dieser Projektion nur geringen Erklärungswert.



# 3D-Scatterplot der Individuen
fig <- plot_ly(
  x = factor_scores[,1],   # Dim.1
  y = factor_scores[,2],   # Dim.2
  z = factor_scores[,3],   # Dim.3
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 4, color = factor_scores[,1], colorscale = "Viridis") # einfärben nach Dim.1
)

fig
