# Commerce international et croissance economique :
# une analyse econometrique a partir des World Development Indicators

# ==================================================
# 0. Installation et chargement des packages
# ==================================================

required_packages <- c(
  "tidyverse", "readr", "tidyr", "dplyr", "ggplot2", "janitor",
  "broom", "car", "lmtest", "sandwich", "zoo", "scales", "purrr"
)

ulib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(ulib)) dir.create(ulib, recursive = TRUE)
.libPaths(c(ulib, .libPaths()))

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org", lib = ulib)
}

invisible(lapply(required_packages, library, character.only = TRUE))

options(scipen = 999)
theme_set(theme_minimal(base_size = 12))

dir.create("r_outputs", showWarnings = FALSE)
dir.create(file.path("r_outputs", "plots"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path("r_outputs", "tables"), showWarnings = FALSE, recursive = TRUE)

cat("Packages charges avec succes.\n")

# ==================================================
# 1. Presentation de la base et des variables
# ==================================================

cat("\n[1] Presentation de la base et des variables\n")

indicators_path <- "Indicators.csv"
country_path <- "Country.csv"

stopifnot(file.exists(indicators_path))
stopifnot(file.exists(country_path))

indicator_map <- c(
  "NY.GDP.MKTP.KD.ZG" = "GDP_growth",
  "NE.TRD.GNFS.ZS" = "Trade",
  "FP.CPI.TOTL.ZG" = "Inflation",
  "SL.UEM.TOTL.ZS" = "Unemployment",
  "NE.GDI.TOTL.ZS" = "Investment",
  "SP.POP.GROW" = "Population"
)

indicator_table <- tibble::tribble(
  ~IndicatorCode, ~IndicatorName, ~Role, ~Interpretation,
  "NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)", "Y", "Croissance economique annuelle",
  "NE.TRD.GNFS.ZS", "Trade (% of GDP)", "X principal", "Ouverture commerciale",
  "FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)", "Controle", "Stabilite macroeconomique",
  "SL.UEM.TOTL.ZS", "Unemployment, total (% of total labor force)", "Controle", "Marche du travail",
  "NE.GDI.TOTL.ZS", "Gross capital formation (% of GDP)", "Controle", "Investissement",
  "SP.POP.GROW", "Population growth (annual %)", "Controle", "Dynamique demographique"
)

readr::write_csv(indicator_table, file.path("r_outputs", "tables", "indicator_table.csv"))

# ==================================================
# 2. Chargement et transformation long -> large
# ==================================================

cat("\n[2] Chargement et transformation long -> large\n")

use_cols <- cols_only(
  CountryName = col_character(),
  CountryCode = col_character(),
  IndicatorName = col_character(),
  IndicatorCode = col_character(),
  Year = col_double(),
  Value = col_double()
)

df_long <- readr::read_csv(indicators_path, col_types = use_cols) %>%
  filter(IndicatorCode %in% names(indicator_map)) %>%
  mutate(variable = dplyr::recode(IndicatorCode, !!!indicator_map))

df_wide <- df_long %>%
  select(CountryName, CountryCode, Year, variable, Value) %>%
  distinct() %>%
  pivot_wider(names_from = variable, values_from = Value) %>%
  select(CountryName, CountryCode, Year, GDP_growth, Trade, Inflation, Unemployment, Investment, Population)

cat("Base longue filtree :", nrow(df_long), "lignes\n")
cat("Base large :", nrow(df_wide), "lignes\n")

# ==================================================
# 3. Nettoyage et pretraitement
# ==================================================

cat("\n[3] Nettoyage et pretraitement\n")

country_meta <- readr::read_csv(
  country_path,
  col_types = cols_only(
    CountryCode = col_character(),
    ShortName = col_character(),
    Region = col_character(),
    IncomeGroup = col_character()
  )
)

country_only_codes <- country_meta %>%
  filter(!is.na(Region)) %>%
  pull(CountryCode)

df_panel <- df_wide
before_country_filter <- nrow(df_panel)

df_panel <- df_panel %>%
  filter(CountryCode %in% country_only_codes) %>%
  mutate(across(c(GDP_growth, Trade, Inflation, Unemployment, Investment, Population), as.numeric))

after_country_filter <- nrow(df_panel)

duplicate_count <- df_panel %>%
  count(CountryCode, Year) %>%
  filter(n > 1) %>%
  nrow()

df_panel <- df_panel %>%
  distinct(CountryCode, Year, .keep_all = TRUE)

period_start <- 1995
period_end <- 2014

numeric_cols <- c("GDP_growth", "Trade", "Inflation", "Unemployment", "Investment", "Population")

df_period <- df_panel %>%
  filter(Year >= period_start, Year <= period_end)

missing_before <- df_period %>%
  summarise(across(all_of(numeric_cols), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing_before")

missing_share <- df_period %>%
  summarise(across(all_of(numeric_cols), ~mean(is.na(.)) * 100)) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing_share_pct")

baseline_df <- df_period %>%
  drop_na(all_of(numeric_cols))

missing_after <- baseline_df %>%
  summarise(across(all_of(numeric_cols), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing_after")

readr::write_csv(missing_before, file.path("r_outputs", "tables", "missing_before.csv"))
readr::write_csv(missing_share, file.path("r_outputs", "tables", "missing_share.csv"))
readr::write_csv(missing_after, file.path("r_outputs", "tables", "missing_after.csv"))
readr::write_csv(baseline_df, file.path("r_outputs", "tables", "trade_growth_clean_panel_professor_R.csv"))

cat("Lignes avant suppression des agregats :", before_country_filter, "\n")
cat("Lignes apres suppression des agregats :", after_country_filter, "\n")
cat("Doublons pays-annee detectes :", duplicate_count, "\n")
cat("Observations sur la periode retenue :", nrow(df_period), "\n")
cat("Observations de base conservees :", nrow(baseline_df), "\n")
cat("Pays conserves :", n_distinct(baseline_df$CountryCode), "\n")

# ==================================================
# 4. Verification que les donnees sont pretes
# ==================================================

cat("\n[4] Verification de la base finale\n")

ready_summary <- tibble(
  test = c(
    "Absence de doublons pays-annee",
    "Variables numeriques",
    "Aucune valeur manquante dans le modele de base",
    "Periode analysee",
    "Nombre final de pays",
    "Nombre final d'observations"
  ),
  resultat = c(
    ifelse(duplicate_count == 0, "Oui", "Non"),
    ifelse(all(sapply(baseline_df[, numeric_cols], is.numeric)), "Oui", "Non"),
    ifelse(sum(is.na(baseline_df[, numeric_cols])) == 0, "Oui", "Non"),
    paste0(period_start, "-", period_end),
    n_distinct(baseline_df$CountryCode),
    nrow(baseline_df)
  )
)

print(ready_summary)
readr::write_csv(ready_summary, file.path("r_outputs", "tables", "ready_summary.csv"))

# ==================================================
# 5. Analyse univariee
# ==================================================

cat("\n[5] Analyse univariee\n")

desc_stats <- baseline_df %>%
  summarise(
    across(
      all_of(numeric_cols),
      list(
        count = ~sum(!is.na(.)),
        mean = ~mean(., na.rm = TRUE),
        median = ~median(., na.rm = TRUE),
        sd = ~sd(., na.rm = TRUE),
        min = ~min(., na.rm = TRUE),
        q25 = ~quantile(., 0.25, na.rm = TRUE),
        q50 = ~quantile(., 0.50, na.rm = TRUE),
        q75 = ~quantile(., 0.75, na.rm = TRUE),
        max = ~max(., na.rm = TRUE),
        skewness = ~{
          x <- .[!is.na(.)]
          n <- length(x)
          if (n < 3) NA_real_ else sum((x - mean(x))^3) / n / (sd(x)^3)
        }
      )
    )
  ) %>%
  pivot_longer(everything()) %>%
  separate(name, into = c("Variable", "Statistique"), sep = "_(?=[^_]+$)") %>%
  pivot_wider(names_from = Statistique, values_from = value)

print(desc_stats)
readr::write_csv(desc_stats, file.path("r_outputs", "tables", "descriptive_stats.csv"))

baseline_long <- baseline_df %>%
  pivot_longer(cols = all_of(numeric_cols), names_to = "Variable", values_to = "Valeur")

p_hist <- ggplot(baseline_long, aes(x = Valeur)) +
  geom_histogram(bins = 30, fill = "#2a9d8f", color = "white") +
  facet_wrap(~Variable, scales = "free", ncol = 3) +
  labs(title = "Histogrammes des variables selectionnees", x = NULL, y = "Frequence")
ggsave(file.path("r_outputs", "plots", "histograms.png"), p_hist, width = 14, height = 8, dpi = 150)

p_box <- ggplot(baseline_long, aes(x = "", y = Valeur)) +
  geom_boxplot(fill = "#e9c46a") +
  facet_wrap(~Variable, scales = "free", ncol = 3) +
  labs(title = "Boites a moustaches des variables selectionnees", x = NULL, y = NULL)
ggsave(file.path("r_outputs", "plots", "boxplots.png"), p_box, width = 14, height = 8, dpi = 150)

iqr_table <- purrr::map_dfr(numeric_cols, function(col) {
  x <- baseline_df[[col]]
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  tibble(
    Variable = col,
    Mediane = median(x, na.rm = TRUE),
    Ecart_type = sd(x, na.rm = TRUE),
    Asymetrie = {
      z <- x[!is.na(x)]
      n <- length(z)
      if (n < 3) NA_real_ else sum((z - mean(z))^3) / n / (sd(z)^3)
    },
    IQR_outliers = sum(x < lower | x > upper, na.rm = TRUE)
  )
})

readr::write_csv(iqr_table, file.path("r_outputs", "tables", "iqr_outliers.csv"))

extreme_table <- purrr::map_dfr(numeric_cols, function(col) {
  baseline_df %>%
    arrange(desc(.data[[col]])) %>%
    slice(1:3) %>%
    transmute(Variable = col, CountryName, Year, Value = .data[[col]])
})

readr::write_csv(extreme_table, file.path("r_outputs", "tables", "extreme_values.csv"))

# ==================================================
# 6. Correlation et analyse bivariate
# ==================================================

cat("\n[6] Correlation et analyse bivariate\n")

corr_matrix <- cor(baseline_df[, numeric_cols], use = "complete.obs")
readr::write_csv(as.data.frame(corr_matrix), file.path("r_outputs", "tables", "correlation_matrix.csv"))

corr_df <- as.data.frame(as.table(corr_matrix))
names(corr_df) <- c("Var1", "Var2", "Correlation")

p_corr <- ggplot(corr_df, aes(Var1, Var2, fill = Correlation)) +
  geom_tile() +
  geom_text(aes(label = round(Correlation, 2)), color = "black") +
  scale_fill_gradient2(low = "#4575b4", mid = "white", high = "#d73027", midpoint = 0) +
  labs(title = "Matrice de correlation", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path("r_outputs", "plots", "correlation_heatmap.png"), p_corr, width = 8, height = 6, dpi = 150)

p_trade <- ggplot(baseline_df, aes(x = Trade, y = GDP_growth)) +
  geom_point(alpha = 0.35) +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Ouverture commerciale et croissance economique", x = "Trade (% of GDP)", y = "GDP growth (annual %)")
ggsave(file.path("r_outputs", "plots", "trade_gdp_scatter.png"), p_trade, width = 8, height = 6, dpi = 150)

pearson_test <- cor.test(baseline_df$Trade, baseline_df$GDP_growth)
capture.output(pearson_test, file = file.path("r_outputs", "tables", "pearson_test.txt"))

# ==================================================
# 7. Regression principale
# ==================================================

cat("\n[7] Regression principale\n")

baseline_model <- lm(
  GDP_growth ~ Trade + Inflation + Unemployment + Investment + Population,
  data = baseline_df
)

baseline_hc3 <- lmtest::coeftest(
  baseline_model,
  vcov. = sandwich::vcovHC(baseline_model, type = "HC3")
)

baseline_hc3_df <- broom::tidy(baseline_hc3)
readr::write_csv(baseline_hc3_df, file.path("r_outputs", "tables", "baseline_hc3.csv"))
capture.output(summary(baseline_model), file = file.path("r_outputs", "tables", "baseline_model_summary.txt"))

# ==================================================
# 8. Diagnostics
# ==================================================

cat("\n[8] Diagnostics\n")

vif_table <- tibble(
  Variable = names(car::vif(baseline_model)),
  VIF = as.numeric(car::vif(baseline_model))
)
readr::write_csv(vif_table, file.path("r_outputs", "tables", "vif_table.csv"))

bp_test <- lmtest::bptest(baseline_model)
capture.output(bp_test, file = file.path("r_outputs", "tables", "breusch_pagan.txt"))

influence_df <- baseline_df %>%
  select(CountryName, CountryCode, Year) %>%
  mutate(
    fitted = fitted(baseline_model),
    residual = resid(baseline_model),
    std_residual = rstandard(baseline_model),
    cooks_distance = cooks.distance(baseline_model),
    leverage = hatvalues(baseline_model)
  )

top_influential <- influence_df %>%
  arrange(desc(cooks_distance)) %>%
  slice(1:10)

readr::write_csv(top_influential, file.path("r_outputs", "tables", "top_influential.csv"))

cook_threshold <- 4 / nrow(influence_df)

p_resid <- ggplot(influence_df, aes(x = fitted, y = residual)) +
  geom_point(alpha = 0.35) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residus en fonction des valeurs ajustees", x = "Valeurs ajustees", y = "Residus")
ggsave(file.path("r_outputs", "plots", "residuals_vs_fitted.png"), p_resid, width = 8, height = 5, dpi = 150)

p_resid_hist <- ggplot(influence_df, aes(x = residual)) +
  geom_histogram(bins = 30, fill = "#457b9d", color = "white") +
  labs(title = "Distribution des residus", x = "Residus", y = "Frequence")
ggsave(file.path("r_outputs", "plots", "residuals_hist.png"), p_resid_hist, width = 8, height = 5, dpi = 150)

# ==================================================
# 9. Test de robustesse
# ==================================================

cat("\n[9] Test de robustesse\n")

robust_df <- baseline_df %>%
  mutate(cooks_distance = cooks.distance(baseline_model)) %>%
  filter(cooks_distance <= cook_threshold) %>%
  select(-cooks_distance)

robust_model <- lm(
  GDP_growth ~ Trade + Inflation + Unemployment + Investment + Population,
  data = robust_df
)

interp_df <- df_period %>%
  arrange(CountryCode, Year) %>%
  group_by(CountryCode) %>%
  mutate(
    Trade = zoo::na.approx(Trade, x = Year, na.rm = FALSE, maxgap = 2),
    Inflation = zoo::na.approx(Inflation, x = Year, na.rm = FALSE, maxgap = 2),
    Unemployment = zoo::na.approx(Unemployment, x = Year, na.rm = FALSE, maxgap = 2),
    Investment = zoo::na.approx(Investment, x = Year, na.rm = FALSE, maxgap = 2),
    Population = zoo::na.approx(Population, x = Year, na.rm = FALSE, maxgap = 2)
  ) %>%
  ungroup() %>%
  drop_na(all_of(numeric_cols))

interp_model <- lm(
  GDP_growth ~ Trade + Inflation + Unemployment + Investment + Population,
  data = interp_df
)

comparison <- tibble(
  Model = c("Baseline complete-case", "Without influential observations", "Interpolation sensitivity"),
  Observations = c(nrow(baseline_df), nrow(robust_df), nrow(interp_df)),
  Trade_coef = c(coef(baseline_model)["Trade"], coef(robust_model)["Trade"], coef(interp_model)["Trade"]),
  Investment_coef = c(coef(baseline_model)["Investment"], coef(robust_model)["Investment"], coef(interp_model)["Investment"]),
  Trade_p_value = c(
    broom::tidy(lmtest::coeftest(baseline_model, vcov. = sandwich::vcovHC(baseline_model, type = "HC3"))) %>% filter(term == "Trade") %>% pull(p.value),
    broom::tidy(lmtest::coeftest(robust_model, vcov. = sandwich::vcovHC(robust_model, type = "HC3"))) %>% filter(term == "Trade") %>% pull(p.value),
    broom::tidy(lmtest::coeftest(interp_model, vcov. = sandwich::vcovHC(interp_model, type = "HC3"))) %>% filter(term == "Trade") %>% pull(p.value)
  ),
  R_squared = c(summary(baseline_model)$r.squared, summary(robust_model)$r.squared, summary(interp_model)$r.squared)
)

readr::write_csv(comparison, file.path("r_outputs", "tables", "robustness_comparison.csv"))

# ==================================================
# 10. Analyse par sous-groupes
# ==================================================

cat("\n[10] Analyse par sous-groupes\n")

country_trade_mean <- baseline_df %>%
  group_by(CountryCode) %>%
  summarise(avg_trade = mean(Trade, na.rm = TRUE), .groups = "drop")

median_trade <- median(country_trade_mean$avg_trade, na.rm = TRUE)

subgroup_df <- baseline_df %>%
  left_join(country_trade_mean, by = "CountryCode") %>%
  mutate(trade_group = if_else(avg_trade >= median_trade, "High-trade", "Low-trade"))

low_trade_df <- subgroup_df %>% filter(trade_group == "Low-trade")
high_trade_df <- subgroup_df %>% filter(trade_group == "High-trade")

low_model <- lm(
  GDP_growth ~ Trade + Inflation + Unemployment + Investment + Population,
  data = low_trade_df
)

high_model <- lm(
  GDP_growth ~ Trade + Inflation + Unemployment + Investment + Population,
  data = high_trade_df
)

subgroup_summary <- subgroup_df %>%
  group_by(trade_group) %>%
  summarise(
    observations = n(),
    countries = n_distinct(CountryCode),
    mean_trade = mean(Trade, na.rm = TRUE),
    mean_growth = mean(GDP_growth, na.rm = TRUE),
    .groups = "drop"
  )

subgroup_comparison <- tibble(
  Group = c("Low-trade", "High-trade"),
  Observations = c(nrow(low_trade_df), nrow(high_trade_df)),
  Countries = c(n_distinct(low_trade_df$CountryCode), n_distinct(high_trade_df$CountryCode)),
  Trade_coef = c(coef(low_model)["Trade"], coef(high_model)["Trade"]),
  Trade_p_value = c(
    broom::tidy(lmtest::coeftest(low_model, vcov. = sandwich::vcovHC(low_model, type = "HC3"))) %>% filter(term == "Trade") %>% pull(p.value),
    broom::tidy(lmtest::coeftest(high_model, vcov. = sandwich::vcovHC(high_model, type = "HC3"))) %>% filter(term == "Trade") %>% pull(p.value)
  ),
  Investment_coef = c(coef(low_model)["Investment"], coef(high_model)["Investment"]),
  Investment_p_value = c(
    broom::tidy(lmtest::coeftest(low_model, vcov. = sandwich::vcovHC(low_model, type = "HC3"))) %>% filter(term == "Investment") %>% pull(p.value),
    broom::tidy(lmtest::coeftest(high_model, vcov. = sandwich::vcovHC(high_model, type = "HC3"))) %>% filter(term == "Investment") %>% pull(p.value)
  ),
  R_squared = c(summary(low_model)$r.squared, summary(high_model)$r.squared)
)

readr::write_csv(subgroup_summary, file.path("r_outputs", "tables", "subgroup_summary.csv"))
readr::write_csv(subgroup_comparison, file.path("r_outputs", "tables", "subgroup_comparison.csv"))

# ==================================================
# 11. Resume final
# ==================================================

cat("\n[11] Resume final\n")

summary_text <- c(
  "Question de recherche : dans quelle mesure l'ouverture commerciale est-elle associee a la croissance economique ?",
  "Le modele principal est un pooled OLS avec erreurs standards robustes HC3.",
  "Le modele de base repose sur un echantillon complete-case pour eviter une imputation excessive.",
  "L'investissement apparait generalement comme une variable positive et significative.",
  "L'effet du commerce est plus fragile et peut dependre des observations influentes et du groupe de pays considere.",
  "Le travail doit etre interprete comme une analyse d'association conditionnelle et non comme une preuve de causalite."
)

writeLines(summary_text, file.path("r_outputs", "tables", "final_summary.txt"))

cat("Execution terminee avec succes.\n")
cat("Les tableaux et graphiques ont ete enregistres dans le dossier r_outputs.\n")
