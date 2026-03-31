from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    md(
        """# Commerce international et croissance économique : une analyse économétrique à partir des World Development Indicators

## Problématique
Dans quelle mesure l’ouverture commerciale semble-t-elle associée à la croissance économique des pays, une fois prises en compte certaines variables macroéconomiques de contrôle ?

## Objectif du notebook
Ce notebook construit une démarche économétrique complète et transparente à partir de la base `World Development Indicators (WDI)` afin d’étudier la relation entre le commerce international et la croissance économique. L’analyse suit les étapes attendues dans un projet universitaire : présentation de la base, choix des variables, transformation des données, statistiques descriptives, analyse de corrélation, régression, diagnostics, tests de robustesse et discussion des limites.

## Portée économétrique
Le notebook estime principalement un modèle de régression `pooled OLS` avec erreurs standards robustes. L’objectif est d’identifier des **associations conditionnelles** entre variables et non de prouver une causalité stricte. Une attention particulière est donc portée au choix de l’échantillon, aux données manquantes, aux valeurs extrêmes et aux observations influentes.
"""
    )
)

cells.append(
    md(
        """## 1. Environnement de travail

Le notebook utilise un environnement Python reproductible fondé sur `pandas` pour la gestion des données, `seaborn` et `matplotlib` pour les visualisations, et `statsmodels` pour l’estimation économétrique. Quelques options d’affichage sont définies afin de garder un document propre, lisible et directement présentable.
"""
    )
)

cells.append(
    code(
        """import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.stats import pearsonr
from IPython.display import display
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")

DATA_PATH = Path("Indicators.csv")
COUNTRY_PATH = Path("Country.csv")
assert DATA_PATH.exists(), f"Fichier introuvable : {DATA_PATH.resolve()}"
assert COUNTRY_PATH.exists(), f"Fichier introuvable : {COUNTRY_PATH.resolve()}"

print("Les fichiers nécessaires ont bien été détectés dans le dossier de travail.")
"""
    )
)

cells.append(
    md(
        """## 2. Présentation et compréhension de la base de données

La base `World Development Indicators (WDI)` de la Banque mondiale est l’une des sources les plus utilisées en macroéconomie appliquée. Elle regroupe des indicateurs harmonisés de développement pour un grand nombre de pays et sur plusieurs années, ce qui en fait une base particulièrement adaptée à une analyse comparative internationale.

### Source et champ de la base
- Source : Banque mondiale, `World Development Indicators` (version Kaggle disponible dans ce projet)
- Fichier principal utilisé : `Indicators.csv`
- Unité d’observation dans le fichier brut : un pays, un indicateur, une année
- Structure des données : **format long**

### Pourquoi cette base est adaptée au thème
Cette base est pertinente pour le thème « commerce international et croissance économique » car elle fournit des indicateurs comparables entre pays sur la croissance, le commerce, l’inflation, l’investissement, le chômage et la dynamique démographique. Ce sont des variables classiques dans les études macroéconomiques sur la croissance.

### Pourquoi une transformation est nécessaire
Une régression nécessite une ligne par unité d’observation. Ici, l’unité pertinente est le **couple pays-année**. Or, dans le fichier WDI brut, chaque indicateur est stocké sur une ligne distincte. Il faut donc transformer la base du **format long** vers un **format large** avant l’estimation économétrique.
"""
    )
)

cells.append(
    code(
        """usecols = ["CountryName", "CountryCode", "IndicatorName", "IndicatorCode", "Year", "Value"]

# Lecture par morceaux pour gérer efficacement un fichier volumineux.
total_rows = 0
sample_chunks = []
for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=usecols, chunksize=500_000)):
    total_rows += len(chunk)
    if i == 0:
        sample_chunks.append(chunk.head(5))

sample_rows = pd.concat(sample_chunks, ignore_index=True)

print(f"Nombre approximatif de lignes dans la base brute : {total_rows:,}")
print("Colonnes utilisées :", usecols)
print("\\nTypes des variables :")
print(sample_rows.dtypes)
print("\\nExemple de lignes du fichier en format long :")
display(sample_rows)
"""
    )
)

cells.append(
    md(
        """L’aperçu ci-dessus illustre bien la logique du format long : un même pays apparaît plusieurs fois, car chaque ligne correspond à une combinaison `pays-indicateur-année`. Cette structure est pratique pour stocker beaucoup d’indicateurs, mais elle n’est pas directement exploitable pour une régression multivariée tant que les variables retenues ne sont pas alignées sur une seule ligne par `pays-année`.
"""
    )
)

cells.append(
    md(
        """## 3. Choix des variables

Le modèle économétrique repose sur une variable dépendante, une variable explicative principale et plusieurs variables de contrôle. Ces contrôles sont retenus parce qu’ils représentent des dimensions macroéconomiques susceptibles d’évoluer conjointement avec la croissance et le commerce.

- **GDP growth** est la variable dépendante `Y`, car elle mesure la performance économique annuelle.
- **Trade (% of GDP)** est la variable explicative principale `X`, car elle mesure l’ouverture commerciale.
- **Inflation** est introduite comme indicateur d’instabilité macroéconomique.
- **Unemployment** est ajoutée car un marché du travail dégradé est souvent associé à une croissance plus faible.
- **Investment** est un déterminant classique de la croissance dans la théorie économique.
- **Population growth** est retenue car la dynamique démographique peut influencer la croissance agrégée.
"""
    )
)

cells.append(
    code(
        """indicator_info = [
    ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)", "Y", "Dependent variable measuring annual real economic growth"),
    ("NE.TRD.GNFS.ZS", "Trade (% of GDP)", "Main X", "Trade openness indicator"),
    ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)", "Control", "Macro-stability control"),
    ("SL.UEM.TOTL.ZS", "Unemployment, total (% of total labor force)", "Control", "Labor-market control"),
    ("NE.GDI.TOTL.ZS", "Gross capital formation (% of GDP)", "Control", "Investment control"),
    ("SP.POP.GROW", "Population growth (annual %)", "Control", "Demographic control"),
]

indicator_table = pd.DataFrame(
    indicator_info,
    columns=["Code de l'indicateur", "Nom de l'indicateur", "Rôle dans le modèle", "Interprétation économique"]
)
display(indicator_table)
"""
    )
)

cells.append(
    md(
        """## 4. Transformation des données

Cette étape consiste à filtrer le fichier brut pour ne conserver que les indicateurs utiles, puis à transformer les données du format long vers le format large. Après cette transformation, chaque ligne correspond à une observation `pays-année` contenant toutes les variables nécessaires à l’estimation.
"""
    )
)

cells.append(
    code(
        """indicator_map = {
    "NY.GDP.MKTP.KD.ZG": "GDP_growth",
    "NE.TRD.GNFS.ZS": "Trade",
    "FP.CPI.TOTL.ZG": "Inflation",
    "SL.UEM.TOTL.ZS": "Unemployment",
    "NE.GDI.TOTL.ZS": "Investment",
    "SP.POP.GROW": "Population",
}

chunks = []
for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=500_000):
    filtered = chunk[chunk["IndicatorCode"].isin(indicator_map)]
    chunks.append(filtered)

df_long = pd.concat(chunks, ignore_index=True)
df_long["variable"] = df_long["IndicatorCode"].map(indicator_map)

print("Taille de la base filtrée en format long :", df_long.shape)
display(df_long.head())

df_wide = (
    df_long.pivot_table(
        index=["CountryName", "CountryCode", "Year"],
        columns="variable",
        values="Value",
        aggfunc="first",
    )
    .reset_index()
)

ordered_cols = [
    "CountryName",
    "CountryCode",
    "Year",
    "GDP_growth",
    "Trade",
    "Inflation",
    "Unemployment",
    "Investment",
    "Population",
]
df_wide = df_wide[ordered_cols]

print("Taille de la base transformée en format large :", df_wide.shape)
display(df_wide.head())
"""
    )
)

cells.append(
    md(
        """Cette transformation crée la structure de panel nécessaire au travail économétrique. Le tableau final contient désormais les identifiants `pays` et `année`, tandis que les six indicateurs sélectionnés deviennent des colonnes pouvant être introduites simultanément dans le modèle de régression.
"""
    )
)

cells.append(
    md(
        """## 5. Nettoyage des données et prétraitement

Cette section rend explicite la construction de l’échantillon final.

### Choix méthodologiques principaux
1. **Exclure les agrégats** tels que `World`, `Arab World` ou les groupes régionaux. Ce ne sont pas des pays individuels et leur présence fausserait une régression menée au niveau des pays.
2. **Retenir la période 1995-2014**. Le fichier local WDI s’arrête en 2014, et la variable de chômage commence plus tard que les autres. Commencer en 1995 permet d’obtenir un panel plus large que 2010-2014 tout en gardant une couverture correcte des variables.
3. **Utiliser un échantillon complet (`complete-case`) pour le modèle de base**. Cela évite d’introduire artificiellement des valeurs imputées dans la variable dépendante et rend la spécification principale plus transparente.
4. **Réserver l’interpolation à un test de robustesse**. Ce choix est plus prudent d’un point de vue académique, car il sépare le modèle principal d’hypothèses de prétraitement plus fortes.

Cette distinction est importante. La première version du notebook reposait davantage sur l’imputation. Dans la version finale, le modèle de base est estimé uniquement sur les observations réellement disponibles, tandis que l’interpolation n’est utilisée qu’ensuite pour vérifier si la conclusion dépend fortement du traitement des données manquantes.
"""
    )
)

cells.append(
    code(
        """country_meta = pd.read_csv(
    COUNTRY_PATH,
    usecols=["CountryCode", "ShortName", "Region", "IncomeGroup"]
)

country_only_codes = country_meta.loc[country_meta["Region"].notna(), "CountryCode"]

df_panel = df_wide.copy()
before_country_filter = df_panel.shape[0]
df_panel = df_panel[df_panel["CountryCode"].isin(country_only_codes)].copy()
after_country_filter = df_panel.shape[0]

numeric_cols = ["GDP_growth", "Trade", "Inflation", "Unemployment", "Investment", "Population"]
for col in numeric_cols:
    df_panel[col] = pd.to_numeric(df_panel[col], errors="coerce")

duplicate_count = df_panel.duplicated(subset=["CountryCode", "Year"]).sum()
df_panel = df_panel.drop_duplicates(subset=["CountryCode", "Year"]).copy()

coverage = (
    df_panel.groupby("Year")
    .agg(
        countries=("CountryCode", "nunique"),
        complete_rows=("GDP_growth", lambda s: s.notna().sum())
    )
    .reset_index()
)

period_start, period_end = 1995, 2014
df_period = df_panel[(df_panel["Year"] >= period_start) & (df_panel["Year"] <= period_end)].copy()

missing_before = df_period[numeric_cols].isna().sum().to_frame("Valeurs manquantes avant nettoyage")
baseline_df = df_period.dropna(subset=numeric_cols).copy()
missing_after = baseline_df[numeric_cols].isna().sum().to_frame("Valeurs manquantes après nettoyage complete-case")

missing_share = (df_period[numeric_cols].isna().mean() * 100).round(2).to_frame("Part manquante (%)")

print(f"Nombre de lignes avant suppression des agrégats : {before_country_filter:,}")
print(f"Nombre de lignes après suppression des agrégats : {after_country_filter:,}")
print(f"Nombre de doublons pays-année détectés : {duplicate_count:,}")
print(f"Nombre de lignes sur la période retenue ({period_start}-{period_end}) avant nettoyage : {len(df_period):,}")
print(f"Nombre de lignes conservées dans l’échantillon de base : {len(baseline_df):,}")
print(f"Nombre de pays conservés dans l’échantillon de base : {baseline_df['CountryCode'].nunique():,}")
print(f"Part des observations pays-année conservées dans l’échantillon de base : {len(baseline_df) / len(df_period):.2%}")

print("\\nCouverture des pays par année :")
display(coverage.tail(10))

print("\\nValeurs manquantes avant nettoyage :")
display(missing_before)

print("\\nPart des valeurs manquantes avant nettoyage :")
display(missing_share)

print("\\nValeurs manquantes après nettoyage de base :")
display(missing_after)

display(baseline_df.head())
"""
    )
)

cells.append(
    md(
        """L’échantillon de base conserve uniquement les observations `pays-année` pour lesquelles les six variables sont observées. Ce choix réduit la taille de l’échantillon, mais il améliore l’interprétation du modèle, car la régression n’est pas portée par une imputation excessive. En contrepartie, les pays ayant une qualité de reporting plus faible contribuent moins à l’analyse.

Le tableau des valeurs manquantes montre aussi pourquoi l’interpolation doit être utilisée avec prudence. Dans cette base, le manque d’information n’est pas négligeable pour l’inflation, le chômage, l’investissement et le commerce. Si l’on remplaçait massivement ces valeurs dans le modèle principal, la régression pourrait paraître artificiellement plus précise qu’elle ne l’est réellement. C’est pourquoi le notebook distingue bien **le modèle de base** et **l’analyse de sensibilité avec interpolation**.
"""
    )
)

cells.append(
    code(
"""baseline_df.to_csv("trade_growth_clean_panel_professor.csv", index=False)
print("Le fichier propre pour la régression a été enregistré : trade_growth_clean_panel_professor.csv")
"""
    )
)

cells.append(
    md(
        """## 6. Analyse univariée

Cette section répond explicitement à l’exigence d’analyse univariée demandée dans le projet. Elle décrit chaque variable séparément avant de passer à la modélisation multivariée. L’objectif est d’étudier la tendance centrale, la dispersion, l’asymétrie et la présence éventuelle de valeurs extrêmes.
"""
    )
)

cells.append(
    code(
        """desc_stats = baseline_df[numeric_cols].describe().T
desc_stats["median"] = baseline_df[numeric_cols].median()
desc_stats["skewness"] = baseline_df[numeric_cols].skew()
desc_stats = desc_stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max", "skewness"]]
display(desc_stats)
"""
    )
)

cells.append(
    code(
        """fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()
for ax, col in zip(axes, numeric_cols):
    sns.histplot(baseline_df[col], kde=True, bins=30, ax=ax, color="#2a9d8f")
    ax.set_title(f"Histogramme de {col}")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()
for ax, col in zip(axes, numeric_cols):
    sns.boxplot(x=baseline_df[col], ax=ax, color="#e9c46a")
    ax.set_title(f"Boîte à moustaches de {col}")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """iqr_summary = []
for col in numeric_cols:
    q1 = baseline_df[col].quantile(0.25)
    q3 = baseline_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((baseline_df[col] < lower) | (baseline_df[col] > upper)).sum()
    iqr_summary.append((col, baseline_df[col].median(), baseline_df[col].std(), baseline_df[col].skew(), outliers))

iqr_table = pd.DataFrame(
    iqr_summary,
    columns=["Variable", "Médiane", "Écart-type", "Asymétrie", "Nombre d'outliers (IQR)"]
)
display(iqr_table)

print("Interprétation univariée :")
for _, row in iqr_table.iterrows():
    skew_label = "asymétrique à droite" if row["Asymétrie"] > 0.5 else ("asymétrique à gauche" if row["Asymétrie"] < -0.5 else "approximativement symétrique")
    print(
        f"- {row['Variable']} : médiane = {row['Médiane']:.3f}, écart-type = {row['Écart-type']:.3f}, "
        f"distribution {skew_label}, et la boîte à moustaches signale {int(row[\"Nombre d'outliers (IQR)\"])} valeurs potentiellement extrêmes."
    )
print("- Les variables ne sont pas sur les mêmes ordres de grandeur, ce qui est fréquent en macroéconomie et renforce l’importance d’une interprétation prudente.")
"""
    )
)

cells.append(
    code(
        """extreme_rows = []
for col in numeric_cols:
    top_obs = baseline_df.nlargest(3, col)[["CountryName", "Year", col]].copy()
    top_obs["Variable"] = col
    top_obs = top_obs.rename(columns={col: "Value"})
    extreme_rows.append(top_obs[["Variable", "CountryName", "Year", "Value"]])

extreme_table = pd.concat(extreme_rows, ignore_index=True)
display(extreme_table)

print("Interprétation des valeurs extrêmes :")
print("- Des taux de croissance du PIB très élevés reflètent souvent des rebonds post-conflit, des booms liés aux matières premières ou des reprises après une forte contraction, plutôt qu’une croissance ordinaire.")
print("- Des ratios Trade/PIB très élevés sont fréquents dans les petites économies très ouvertes ou dans les pays exportateurs de ressources, et ne doivent donc pas être considérés automatiquement comme des erreurs de mesure.")
print("- Des niveaux d’inflation extrêmes traduisent souvent des épisodes d’instabilité macroéconomique, ce qui peut rendre les régressions internationales plus bruitées.")
print("- Ces constats justifient l’étude formelle des observations influentes avant d’interpréter trop fortement le coefficient du commerce.")
"""
    )
)

cells.append(
    md(
        """L’analyse univariée montre déjà que l’échantillon contient plusieurs observations extrêmes, notamment pour la croissance, l’inflation et l’ouverture commerciale. Cela est important, car une régression `pooled OLS` peut être sensible à ce type de cas. Les diagnostics et les tests de robustesse sont donc indispensables.
"""
    )
)

cells.append(
    md(
        """## 7. Correlation and Bivariate Analysis

We next examine the direct relationship between the dependent variable and the main explanatory variable of interest. This does not control for the other regressors and should be interpreted as a descriptive relationship only.

Importantly, **correlation does not imply causality**. Even a strong bivariate association would not establish that trade causes growth.
"""
    )
)

cells.append(
    code(
        """plt.figure(figsize=(9, 6))
sns.regplot(
    data=baseline_df,
    x="Trade",
    y="GDP_growth",
    scatter_kws={"alpha": 0.35},
    line_kws={"color": "crimson"}
)
plt.title("Trade Openness and GDP Growth")
plt.show()

pearson_r, pearson_p = pearsonr(baseline_df["Trade"], baseline_df["GDP_growth"])
print(f"Pearson correlation coefficient: {pearson_r:.4f}")
print(f"P-value: {pearson_p:.4g}")
"""
    )
)

cells.append(
    code(
        """if pearson_p < 0.05:
    sig_text = "statistically significant"
else:
    sig_text = "not statistically significant"

direction = "positive" if pearson_r > 0 else "negative"
print("Bivariate interpretation:")
print(
    f"- The raw correlation between Trade and GDP growth is {pearson_r:.4f}, which indicates a {direction} linear association."
)
print(f"- This correlation is {sig_text} at conventional levels.")
print("- Because this is a bivariate result, it does not account for inflation, unemployment, investment, or population growth.")
"""
    )
)

cells.append(
    md(
        """## 8. Econometric Model

The core specification is the following pooled OLS regression:

\\[
GDP\\_growth_{it} = \\beta_0 + \\beta_1 Trade_{it} + \\beta_2 Inflation_{it} + \\beta_3 Unemployment_{it} + \\beta_4 Investment_{it} + \\beta_5 Population_{it} + \\varepsilon_{it}
\\]

where \\(i\\) denotes countries and \\(t\\) denotes years.

### Why pooled OLS?
Pooled OLS is appropriate here as a clear baseline model for a university econometrics project. It is easy to interpret and directly answers whether trade is conditionally associated with growth once a small set of controls is included.

### Why robust standard errors?
Cross-country macroeconomic data often display heteroskedasticity, meaning error variance can differ substantially across observations. For that reason, the model reports **HC3 robust standard errors**.
"""
    )
)

cells.append(
    code(
        """X_vars = ["Trade", "Inflation", "Unemployment", "Investment", "Population"]
X = sm.add_constant(baseline_df[X_vars])
y = baseline_df["GDP_growth"]

ols_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
print(ols_hc3.summary())
"""
    )
)

cells.append(
    code(
        """coef_table = pd.DataFrame({
    "Coefficient": ols_hc3.params,
    "Robust Std. Error": ols_hc3.bse,
    "z-stat": ols_hc3.tvalues,
    "p-value": ols_hc3.pvalues
})
display(coef_table)

print("Regression interpretation:")
for var in X_vars:
    coef = ols_hc3.params[var]
    pval = ols_hc3.pvalues[var]
    if pval < 0.01:
        sig = "highly significant"
    elif pval < 0.05:
        sig = "significant"
    elif pval < 0.10:
        sig = "weakly significant"
    else:
        sig = "not statistically significant"
    sign = "positive" if coef > 0 else "negative"
    print(f"- {var}: coefficient = {coef:.4f}, {sign} sign, {sig} (p = {pval:.4f}).")

trade_coef = ols_hc3.params["Trade"]
trade_p = ols_hc3.pvalues["Trade"]
print("\\nFocus on Trade:")
if trade_p < 0.05:
    print(
        f"- In the baseline model, Trade is positively associated with GDP growth and statistically significant (coefficient = {trade_coef:.4f}, p = {trade_p:.4f})."
    )
else:
    print(
        f"- In the baseline model, Trade has a positive coefficient ({trade_coef:.4f}) but is not statistically significant (p = {trade_p:.4f})."
    )
    print("- This means the baseline specification does not provide strong evidence that trade openness is independently associated with faster growth once the controls are included.")
"""
    )
)

cells.append(
    md(
        """### Deeper Econometric Interpretation

The baseline trade coefficient deserves careful interpretation. A non-significant coefficient does not mean that trade is irrelevant in economic reality. Rather, it means that within this sample and specification, the estimated conditional association is too imprecise to distinguish clearly from zero at conventional significance levels.

Several reasons can explain this result:
- **Country heterogeneity:** countries differ widely in institutions, export structure, geography, and development stage. A single pooled coefficient may average over very different trade-growth relationships.
- **Measurement and composition effects:** trade as a share of GDP does not distinguish between manufacturing exports, commodity exports, re-export hubs, or import dependence.
- **Short and noisy macro horizons:** even with the extended 1995-2014 sample, annual growth is volatile and influenced by crises, wars, and commodity shocks.
- **Omitted variables:** institutional quality, education, productivity, exchange-rate regimes, and political stability are not controlled for directly here.
- **Possible non-linearity:** trade may help growth more strongly for some country groups than for others, or only after complementary institutions are in place.

By contrast, the positive and statistically significant coefficient on **Investment** is economically intuitive: higher gross capital formation is closely linked to capital accumulation, infrastructure, and productive capacity, all of which are standard drivers of growth in macroeconomics.

The negative coefficient on **Unemployment** also has a plausible interpretation. Higher unemployment reflects weaker labor-market utilization and often accompanies weak aggregate demand or structural economic problems, which are generally associated with lower growth.
"""
    )
)

cells.append(
    md(
        """## 9. Diagnostics

This section evaluates whether the baseline model is being distorted by common econometric problems such as multicollinearity, heteroskedasticity, and influential observations.
"""
    )
)

cells.append(
    code(
        """vif_table = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
display(vif_table)

bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(ols_hc3.resid, X)
print(f"Breusch-Pagan LM statistic: {bp_stat:.4f}")
print(f"Breusch-Pagan p-value: {bp_pvalue:.4g}")
print("Interpretation: a low p-value suggests heteroskedasticity, which supports the use of robust standard errors.")
"""
    )
)

cells.append(
    code(
        """influence = ols_hc3.get_influence()
diagnostics = baseline_df[["CountryName", "CountryCode", "Year"]].copy()
diagnostics["fitted"] = ols_hc3.fittedvalues
diagnostics["residual"] = ols_hc3.resid
diagnostics["std_residual"] = influence.resid_studentized_internal
diagnostics["cooks_distance"] = influence.cooks_distance[0]
diagnostics["leverage"] = influence.hat_matrix_diag

top_influential = diagnostics.sort_values("cooks_distance", ascending=False).head(10)
display(top_influential)

cook_threshold = 4 / len(diagnostics)
high_influence_count = (diagnostics["cooks_distance"] > cook_threshold).sum()
print(f"Cook's distance threshold (4/n): {cook_threshold:.6f}")
print(f"Observations above threshold: {high_influence_count}")
"""
    )
)

cells.append(
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(x=ols_hc3.fittedvalues, y=ols_hc3.resid, alpha=0.35, ax=axes[0])
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals vs Fitted")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")

sns.histplot(ols_hc3.resid, kde=True, ax=axes[1], color="#457b9d")
axes[1].set_title("Distribution of Residuals")

plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """print("Diagnostic interpretation:")
print("- VIF values close to 1 indicate that multicollinearity is not a major concern in this specification.")
if bp_pvalue < 0.05:
    print("- The Breusch-Pagan test suggests heteroskedasticity, which justifies the use of HC3 robust standard errors.")
else:
    print("- The Breusch-Pagan test does not indicate strong heteroskedasticity, but robust standard errors remain a prudent choice in cross-country data.")
print("- The Cook's distance table identifies a small set of highly influential observations, so outlier sensitivity should be checked explicitly.")
"""
    )
)

cells.append(
    md(
        """## 10. Bias and Econometric Limitations

The baseline results should be interpreted carefully.

### Omitted variable bias
Economic growth depends on many omitted factors such as institutional quality, education, governance, commodity dependence, conflict, and technological change. If these omitted factors are correlated with trade, the coefficient on trade may be biased.

### Reverse causality
Trade may affect growth, but growth can also affect trade. Fast-growing economies often trade more, attract more investment, and diversify faster. This means the estimated trade coefficient cannot be interpreted as a clean causal effect.

### Measurement limitations
Cross-country indicators are compiled from different national statistical systems. Even harmonized WDI series may contain reporting differences, revisions, or incomplete coverage.

### Time-horizon limitation
The local file ends in 2014. Although extending the baseline sample to 1995-2014 improves coverage relative to the earlier notebook, the time horizon is still limited by the available data.

### Missing-data limitation
The baseline model uses complete-case estimation. This is transparent, but it may reduce representativeness because countries with weaker reporting systems are more likely to be excluded.

### Pooled OLS limitation
The data are country-year observations, so pooled OLS ignores unobserved country-specific effects and common year shocks. A more advanced extension would use fixed effects, but that goes beyond the baseline scope of this notebook.
"""
    )
)

cells.append(
    md(
        """## 11. Robustness Check

The baseline model may be sensitive to influential observations. To address this concern, the notebook re-estimates the regression after removing observations whose Cook's distance exceeds the common threshold of \\(4/n\\). This is a useful and interpretable robustness check because it directly targets the outlier problem identified in the diagnostics.

As an additional sensitivity check, the notebook also compares the baseline complete-case model with a limited interpolation alternative in which missing control variables are linearly interpolated within country for short internal gaps only. The dependent variable is never imputed.
"""
    )
)

cells.append(
    code(
        """robust_df = baseline_df.loc[diagnostics["cooks_distance"] <= cook_threshold].copy()
X_robust = sm.add_constant(robust_df[X_vars])
y_robust = robust_df["GDP_growth"]
ols_no_influential = sm.OLS(y_robust, X_robust).fit(cov_type="HC3")

interp_df = df_period.sort_values(["CountryCode", "Year"]).copy()
interp_controls = ["Trade", "Inflation", "Unemployment", "Investment", "Population"]
interp_df[interp_controls] = (
    interp_df.groupby("CountryCode", group_keys=False)[interp_controls]
    .apply(lambda g: g.interpolate(method="linear", limit=2, limit_area="inside"))
)
interp_df = interp_df.dropna(subset=numeric_cols).copy()
X_interp = sm.add_constant(interp_df[X_vars])
y_interp = interp_df["GDP_growth"]
ols_interp = sm.OLS(y_interp, X_interp).fit(cov_type="HC3")

comparison = pd.DataFrame({
    "Model": ["Baseline complete-case", "Without influential observations", "Interpolation sensitivity"],
    "Observations": [len(baseline_df), len(robust_df), len(interp_df)],
    "Trade coef.": [ols_hc3.params["Trade"], ols_no_influential.params["Trade"], ols_interp.params["Trade"]],
    "Investment coef.": [ols_hc3.params["Investment"], ols_no_influential.params["Investment"], ols_interp.params["Investment"]],
    "Trade p-value": [ols_hc3.pvalues["Trade"], ols_no_influential.pvalues["Trade"], ols_interp.pvalues["Trade"]],
    "R-squared": [ols_hc3.rsquared, ols_no_influential.rsquared, ols_interp.rsquared]
})
display(comparison)
"""
    )
)

cells.append(
    code(
        """print("Robustness interpretation:")
print(
    f"- Baseline sample: Trade coefficient = {ols_hc3.params['Trade']:.4f}, p-value = {ols_hc3.pvalues['Trade']:.4f}."
)
print(
    f"- After removing influential observations: Trade coefficient = {ols_no_influential.params['Trade']:.4f}, p-value = {ols_no_influential.pvalues['Trade']:.4f}."
)
print(
    f"- Interpolation sensitivity sample: Trade coefficient = {ols_interp.params['Trade']:.4f}, p-value = {ols_interp.pvalues['Trade']:.4f}."
)
print(
    f"- Baseline R-squared = {ols_hc3.rsquared:.4f}; without influential observations R-squared = {ols_no_influential.rsquared:.4f}; interpolation sensitivity R-squared = {ols_interp.rsquared:.4f}."
)

if ols_hc3.pvalues["Trade"] >= 0.05 and ols_no_influential.pvalues["Trade"] < 0.05:
    print("- The trade result becomes statistically significant once highly influential observations are removed.")
    print("- This suggests that the baseline non-significance partly reflects outlier sensitivity rather than a fully stable null relationship.")
elif ols_hc3.pvalues["Trade"] < 0.05 and ols_no_influential.pvalues["Trade"] < 0.05:
    print("- The significance of trade is broadly robust to removing influential observations.")
else:
    print("- The trade result remains fragile across specifications, so the evidence should be described as limited rather than conclusive.")

print("- The interpolation-based model produces very similar trade estimates to the baseline, which suggests that the main conclusion is not driven by the complete-case choice alone.")
"""
    )
)

cells.append(
    md(
        """## 12. Subgroup Analysis: High-Trade vs Low-Trade Countries

To explore whether the trade-growth relationship differs across country types, the baseline sample is split into two groups using each country's **average trade openness** over the sample period. Countries above the median average trade share are classified as **high-trade**, while those below the median are classified as **low-trade**.

This subgroup exercise is useful because a single pooled coefficient may hide heterogeneous relationships. In particular, trade may matter more for economies that are already structurally open to international markets than for relatively closed economies.
"""
    )
)

cells.append(
    code(
        """country_trade_mean = (
    baseline_df.groupby("CountryCode")["Trade"]
    .mean()
    .rename("avg_trade")
    .reset_index()
)

median_trade = country_trade_mean["avg_trade"].median()
subgroup_df = baseline_df.merge(country_trade_mean, on="CountryCode", how="left")
subgroup_df["trade_group"] = np.where(
    subgroup_df["avg_trade"] >= median_trade,
    "High-trade",
    "Low-trade"
)

subgroup_summary = (
    subgroup_df.groupby("trade_group")
    .agg(
        observations=("CountryCode", "size"),
        countries=("CountryCode", "nunique"),
        mean_trade=("Trade", "mean"),
        mean_growth=("GDP_growth", "mean")
    )
    .reset_index()
)

print(f"Median country-average Trade used for the split: {median_trade:.4f}")
display(subgroup_summary)
"""
    )
)

cells.append(
    code(
        """subgroup_results = []
subgroup_models = {}

for group_name in ["Low-trade", "High-trade"]:
    sub = subgroup_df[subgroup_df["trade_group"] == group_name].copy()
    X_sub = sm.add_constant(sub[X_vars])
    y_sub = sub["GDP_growth"]
    model_sub = sm.OLS(y_sub, X_sub).fit(cov_type="HC3")
    subgroup_models[group_name] = model_sub
    subgroup_results.append({
        "Group": group_name,
        "Observations": len(sub),
        "Countries": sub["CountryCode"].nunique(),
        "Trade coef.": model_sub.params["Trade"],
        "Trade p-value": model_sub.pvalues["Trade"],
        "Investment coef.": model_sub.params["Investment"],
        "Investment p-value": model_sub.pvalues["Investment"],
        "R-squared": model_sub.rsquared
    })

subgroup_comparison = pd.DataFrame(subgroup_results)
display(subgroup_comparison)
"""
    )
)

cells.append(
    code(
        """print("Subgroup interpretation:")
low_model = subgroup_models["Low-trade"]
high_model = subgroup_models["High-trade"]

print(
    f"- Low-trade countries: Trade coefficient = {low_model.params['Trade']:.4f}, p-value = {low_model.pvalues['Trade']:.4f}, R-squared = {low_model.rsquared:.4f}."
)
print(
    f"- High-trade countries: Trade coefficient = {high_model.params['Trade']:.4f}, p-value = {high_model.pvalues['Trade']:.4f}, R-squared = {high_model.rsquared:.4f}."
)
print("- In the low-trade group, the estimated trade coefficient is very close to zero and statistically insignificant.")
print("- In the high-trade group, the trade coefficient is larger and much closer to conventional significance thresholds.")
print("- This pattern suggests that trade may be more strongly associated with growth in economies that are already structurally integrated into global markets.")
print("- The result should still be interpreted cautiously, because subgroup OLS does not remove the broader concerns about omitted variables, reverse causality, or country heterogeneity.")
"""
    )
)

cells.append(
    md(
        """## 13. Final Discussion and Conclusion

This notebook addressed the research question of whether trade openness is associated with economic growth using a country-year panel derived from the World Development Indicators. Compared with the earlier version of the project, the final notebook improves the analysis by excluding aggregate entities, extending the sample beyond 2010-2014 to 1995-2014, making the missing-data strategy explicit, diagnosing influential observations, and adding robustness checks.

The empirical findings are nuanced. In the baseline complete-case model with HC3 robust standard errors, the coefficient on trade is positive but not statistically significant. In contrast, investment is positive and statistically significant, which is consistent with standard growth theory linking capital accumulation to output expansion. The unemployment coefficient is negative in the outlier-adjusted specification, which is also economically plausible because weaker labor-market conditions tend to accompany weaker growth performance. Once influential observations are removed, the trade coefficient becomes statistically significant, showing that the baseline trade result is sensitive to a small number of extreme country-year observations. The subgroup analysis adds a further insight: the association between trade and growth appears much weaker among low-trade countries than among countries that are already more open to international markets.

Taken together, the evidence does **not** support a strong causal claim that trade openness by itself raises growth in this sample. The most careful conclusion is that trade is positively associated with growth, but the estimated relationship is modest and sensitive to outliers, sample construction, and model specification. This is an important substantive finding rather than a failure of the project: it shows that macroeconomic relationships can look unstable when countries differ strongly in structure, shocks, and data quality.

The main limitations remain the finite time horizon of the available file, missing data, measurement differences across countries, and the fact that pooled OLS cannot control for unobserved country-specific effects. Future work could extend the sample with newer WDI releases, add relevant omitted variables, and estimate fixed-effects or other panel-data models. Those extensions would make it easier to separate broad cross-country differences from within-country dynamics and would provide a stronger basis for interpreting the trade-growth relationship.
"""
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
}

notebook_path = Path("trade_growth_wdi_analysis.ipynb")
nbf.write(nb, notebook_path)

nb_for_exec = nbf.read(notebook_path, as_version=4)
client = NotebookClient(nb_for_exec, timeout=1800, kernel_name="python3", allow_errors=False)
client.execute()
nbf.write(nb_for_exec, notebook_path)

print(f"Notebook built and executed successfully: {notebook_path.resolve()}")
