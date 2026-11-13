# -*- coding: utf-8 -*-
"""
@author: Claudio Palominos
"""

import re
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# PATHS
path_input     = '/PATH/TO/PRIVATE_DATA/german/texts/'
path_demo      = '/PATH/TO/PRIVATE_DATA/german/'
path_results   = '/PATH/TO/RESULTS/'
path_santander = '/PATH/TO/PRIVATE_DATA/spanish/'
path_london    = '/PATH/TO/PRIVATE_DATA/english/'


""" READ AND PREPARE FILES """

df_ge = pd.read_pickle(path_results+'df_words_dim_sensitivity_german.pkl')
df_sp = pd.read_pickle(path_results+'df_words_dim_sensitivity_spanish.pkl')
df_en = pd.read_pickle(path_results+'df_words_dim_sensitivity_english.pkl')



def mean_vec_by_name(df, vec_col, key="name", mean=True, keep=None):
    """
    Expand a vector column into wide numeric columns.

    - key: str or list/tuple of columns to group by when mean=True.
    - keep: extra columns to carry through when mean=False (e.g., ['bild']).
    """
    # normalize keys
    keys = [key] if isinstance(key, str) else list(key)
    # columns to carry along in the non-aggregated path
    keep = [] if keep is None else list(keep)
    carry_cols = [c for c in (keys + keep) if c in df.columns]
    # expand vector column
    V = (
        pd.DataFrame(
            df[vec_col].apply(lambda v: np.asarray(v, dtype=np.float32)).tolist(),
            index=df.index
        )
        .add_prefix(f"{vec_col}_")
    )
    # join with requested carry-through columns
    out = df[carry_cols].reset_index(drop=True).join(V.reset_index(drop=True))

    if mean:
        # aggregate by the key columns only
        return out.groupby(keys, as_index=False).mean(numeric_only=True)
    else:
        # keep per-row values, including e.g. 'bild'
        return out



# --- GROUPS SPANISH --- #
df_sant = pd.read_excel(path_santander+'new_merged_sentences.xlsx')
# COLUMN NAMES
column_names = {
    'Picture': 'picture',
    'Group': 'group',
    'PAR': 'name'
}
df_sant = df_sant.rename(columns=column_names)
# VALUES
column_values = {
    'picture': {'picture_1': '1', 'picture_2': '2', 'picture_3': '3'},
    'group': {'controles': 'HC', 'pacientes': 'SZ'},
}
df_sant.replace(column_values, inplace=True)
# UNIQUE NAMES
names_sant = df_sant[['name', 'group']].drop_duplicates()
names_sant.reset_index(drop=True, inplace=True)
# GROUPS
groups_spa = df_sant[['name','group']]
groups_spa = groups_spa.drop_duplicates().reset_index(drop=True)
# DEMOGRAPHICS
df_demo_spa = pd.read_pickle(path_santander+'df_demographics_santander.pkl')
df_sp = pd.merge(df_sp, df_demo_spa, on='name', how='inner')

# --- GROUPS ENGLISH --- #
df_lon = pd.read_excel(path_london+'summary.xlsx')
# Filter rows based on conditions
df_lon = df_lon[(df_lon['Role'] == 'interviewee') & (df_lon['Meta'] != 'Off-task')]
# Drop the columns 'Role', 'Meta', and 'Time'
df_lon = df_lon.drop(['Role', 'Meta', 'Time'], axis=1)
# Convert column names to lowercase
df_lon.columns = df_lon.columns.str.lower()
# DEMOGRAPHICS
df_demo_eng = pd.read_pickle(path_london+'df_demographics_london.pkl')
df_en = pd.merge(df_en, df_demo_eng, on='name', how='inner')


df_g = pd.read_excel(path_london+'groups.xlsx')
# MERGE THE DATA WITH GROUPS
df_lon = df_lon.merge(df_g, on='par', how='inner')
df_lon = df_lon[['par', 'group', 'picture', 'utterance']]
df_lon = df_lon.rename(columns = {'par': 'name'})
df_g   = df_g.rename(columns = {'par': 'name'})

# DROP MDD IN GERMAN SAMPLE
df_ge = df_ge[df_ge['group']!='MDD']
g1    = mean_vec_by_name(df_ge, "values_id_fast", mean=False, keep=["bild"])
g2    = mean_vec_by_name(df_ge, "values_id_bert", mean=False, keep=["bild"])
gg    = g1.merge(g2, on=["name", "bild"], how="outer")
df_ge = df_ge.merge(gg, on=["name", "bild"], how="outer")
df_ge = df_ge.drop(columns=['values_id_fast', 'values_id_bert'])

# AVERAGE FOR SPANISH
g1    = mean_vec_by_name(df_sp, "values_id_fast")
g2    = mean_vec_by_name(df_sp, "values_id_bert")
gg    = g1.merge(g2, on="name", how="outer")
df_sp = df_sp.drop(columns=['group','bild','values_id_fast', 'values_id_bert'])
df_sp = df_sp.groupby('name').mean()
df_sp = df_sp.merge(gg, on="name", how="outer")
df_sp = df_sp.merge(groups_spa, on='name', how='left')

# AVERAGE FOR ENGLISH
g1    = mean_vec_by_name(df_en, "values_id_fast")
g2    = mean_vec_by_name(df_en, "values_id_bert")
gg    = g1.merge(g2, on="name", how="outer")
df_en = df_en.drop(columns=['group','bild','values_id_fast', 'values_id_bert'])
df_en = df_en.groupby('name').mean()
df_en = df_en.merge(gg, on="name", how="outer")
df_en = df_en.merge(df_g, on='name', how='left')



""" FUNCTIONS """

from matplotlib.ticker import MaxNLocator 
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

# PLOT ID WITH BAND ACROSS K VALUES
def plot_id_wide(
    df,
    prefix="values_id_fast_",   # e.g., 'values_id_fast_' or 'values_id_bert_'
    start_k=2,                  # k = start_k + column index
    group=None,                 # e.g., 'HC' or 'SSD'; None => all rows
    group_col="group",          # column holding group labels (optional)
    agg="mean",                 # {'mean','median'}
    band="iqr",                 # {'iqr','sd','sem','ci95','minmax'}
    ax=None,
    alpha_band=0.25,
    label=None,
    name=None
):
    # derive model name from prefix
    if "bert" in prefix.lower():
        model_name = "BERT ID"
    elif "fast" in prefix.lower():
        model_name = "FastText ID"
    else:
        model_name = "ID"
    # --- pick the wide columns: prefix + integer suffix ---
    patt = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    cols = []
    for c in df.columns:
        m = patt.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    if not cols:
        raise ValueError(f"No columns found matching '{prefix}<int>' pattern.")
    cols.sort(key=lambda t: t[0])  # by suffix
    idxs, ordered_cols = zip(*cols)
    # --- optional filter by group ---
    data = df
    if group is not None and group_col in df.columns:
        data = df[df[group_col] == group]
    if data.empty:
        raise ValueError("No rows to plot after filtering.")
    # --- compute per-k summaries (ignore NaNs) ---
    k_vals = start_k + np.array(idxs, dtype=int)
    means   = np.array([np.nanmean(data[c].to_numpy(dtype=float)) for c in ordered_cols])
    medians = np.array([np.nanmedian(data[c].to_numpy(dtype=float)) for c in ordered_cols])
    # sd / n for sem, ci95
    sds, ns = [], []
    for c in ordered_cols:
        v = data[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        n = int(m.sum())
        ns.append(n)
        sds.append(np.nanstd(v, ddof=1) if n > 1 else 0.0)
    sds = np.array(sds, dtype=float)
    ns  = np.array(ns, dtype=int)
    sem = np.divide(sds, np.sqrt(np.clip(ns, 1, None)), out=np.zeros_like(sds), where=ns>0)
    # band bounds
    if band == "iqr":
        p25 = np.array([np.nanpercentile(data[c], 25) for c in ordered_cols])
        p75 = np.array([np.nanpercentile(data[c], 75) for c in ordered_cols])
        lower, upper = p25, p75
    elif band == "sd":
        center = means if agg == "mean" else medians
        lower, upper = center - sds, center + sds
    elif band == "sem":
        center = means if agg == "mean" else medians
        lower, upper = center - sem, center + sem
    elif band == "ci95":
        center = means if agg == "mean" else medians
        lower, upper = center - 1.96 * sem, center + 1.96 * sem
    elif band == "minmax":
        mins = np.array([np.nanmin(data[c]) for c in ordered_cols])
        maxs = np.array([np.nanmax(data[c]) for c in ordered_cols])
        lower, upper = mins, maxs
    else:
        raise ValueError("band must be one of {'iqr','sd','sem','ci95','minmax'}")
    # center line
    center = means if agg == "mean" else medians
    # clean finite
    mask = np.isfinite(k_vals) & np.isfinite(center) & np.isfinite(lower) & np.isfinite(upper)
    x, center, lower, upper = k_vals[mask], center[mask], lower[mask], upper[mask]
    # --- plot ---
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    auto_label = label or (f"group={group}" if group is not None else "")
    ax.plot(x, center, linewidth=2, label=auto_label if auto_label else None)
    if (upper > lower).any():
        ax.fill_between(x, lower, upper, alpha=alpha_band)
    else:
        ax.plot(x, lower, linewidth=1)
    ymin = np.nanmin(lower)
    ymax = np.nanmax(upper)
    ax.set_ylim(ymin - 0.15*(ymax - ymin), ymax)
    ax.tick_params(axis='y', pad=6)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Intrinsic Dimensionality (ID)", fontsize=12, labelpad=16) 
    ax.set_title(f"{model_name} versus k — {agg} with {band} band", fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    if auto_label:  # only create a legend if we have a non-empty label
        ax.legend()
    plt.savefig(path_results+f'sensitivity_all_{name}.jpg', dpi=500, bbox_inches='tight', pad_inches=0.1)
    return ax


# GEE Model and plotting function
def gee_model_results(df, y, x=None, baseline='HC', name_data='data'):
    categories = list(df['group'].unique())
    categories.remove(f'{baseline}')
    categories = [f'{baseline}'] + categories
    df['group'] = pd.Categorical(df['group'], categories=categories, ordered=True)
    #formula = f'{y} ~ {x} + group + age + education + female'
    # Start with mandatory variables
    independent_vars = ['group', 'age', 'female']
    # Add optional x
    if x is not None:
        independent_vars.append(f'{x}')
    # Add 'bild' and 'education' if available
    if 'bild' in df.columns:
        df['bild'] = df['bild'].astype('category')
        independent_vars.append('bild')
    if 'education' in df.columns:
        independent_vars.append('education')
    # Join the independent variables into a formula string
    formula = f"{y} ~ {' + '.join(independent_vars)}"
    # Fit GEE model
    model_gee = sm.GEE.from_formula(formula, groups=df['group'], data=df, family=sm.families.Gaussian())
    result = model_gee.fit()
    print(result.summary())
    # Coefficients, confidence intervals, and p-values
    coefs = result.params
    conf_int = result.conf_int()
    p_values = result.pvalues
    # Function to format p-values with significance stars
    def format_pval(pval):
        if pval < 0.001:
            return "<0.001***"
        elif pval < 0.01:
            return f"{pval:.3f}**"
        elif pval < 0.05:
            return f"{pval:.3f}*"
        else:
            return f"{pval:.3f}"
    # Create dataframe for plotting
    coef_df = pd.DataFrame({
        'Coefficient': coefs.index,
        'Estimate': coefs.values,
        'CI_lower': conf_int[0],
        'CI_upper': conf_int[1],
        'p_value': p_values
    })
    # Custom names for the coefficients
    custom_names = {
        'group[T.SZ]': 'SSD',
        'group[T.FES]': 'FEP',
        'group[T.CS]': 'CS',
        'group[T.CHR]': 'CHR',
        'age': 'Age',
        'education': 'Education',
        'female': 'Female',
    }
    if x is not None:
        custom_names[f'{x}'] = 'N tokens'
    # Remove 'Intercept', 'Group Var', and any 'bild' levels
    exclude_terms = ['Intercept', 'Group Var'] + [term for term in coefs.index if term.startswith('bild')]
    coef_df = coef_df[~coef_df['Coefficient'].isin(exclude_terms)]

    # Format names and p-values
    coef_df['Coefficient'] = coef_df.apply(
        lambda row: f"{custom_names.get(row['Coefficient'], row['Coefficient'])} (p={format_pval(row['p_value'])})",
        axis=1
    )
    # Filter out 'Intercept' and 'Group Var'
    coef_df = coef_df[~coef_df['Coefficient'].str.contains('Intercept|Group Var', case=False)]
    # Set font and size
    plt.rcParams['font.family'] = 'Times New Roman'  
    plt.rcParams['font.size'] = 15
    # Plot coefficients with confidence intervals
    plt.figure(figsize=(8, 6))
    sns.pointplot(x="Estimate", y="Coefficient", data=coef_df, linestyle='none', color='b')
    # Plot confidence intervals
    for i in range(coef_df.shape[0]):
        plt.plot([coef_df['CI_lower'].iloc[i], coef_df['CI_upper'].iloc[i]], [i, i], color='gray', lw=2)
    plt.axvline(0, ls='--', color='red')
    plt.xlabel('Estimate')
    plt.ylabel('Coefficients')
    plt.tight_layout()
    # Save the plot
    #plt.savefig(path_results+f'coeff_gee_{name_data}.jpg', dpi=500)
    plt.show()
    return result


# PLOT COEFFICIENT OF GROUP OVER K
def gee_group_coeff_over_k(
    df,
    y_kind="fast",          # {"fast","bert"} or explicit prefix like "values_id_fast_"
    start=2,
    end=30,
    x=None,                 # optional covariate (e.g., 'n_content_u')
    baseline="HC",
    group_col="group",
    gee_groups_col="group", 
    min_rows=5,
    plot=True,
    name_sample = None,
    alpha_sig=0.01,         # <-- significance threshold for star
    star_size=170,          # <-- size of the star markers
):
    """
    For k in [start..end], fit (GEE, Gaussian):
        y_k ~ group + age + female (+ x?) (+ bild?) (+ education?)
    with clusters given by `gee_groups_col`.

    Returns a tidy DataFrame with columns:
        ['k','group','coefficient','p_value']
    and (optionally) plots coefficient vs k (one line per non-baseline group),
    adding a red star marker at points where p_value < alpha_sig.
    """
    # choose prefix
    if y_kind == "fast":
        prefix = "values_id_fast_"
    elif y_kind == "bert":
        prefix = "values_id_bert_"
    else:
        prefix = y_kind  # allow custom prefix
    rows, skips = [], {}
    for k in range(start, end + 1):        # e.g., start=10
        suffix = k - 2                     # because values_0 -> k=2
        if suffix < 0:
            continue
        y = f"{prefix}{suffix}"            # e.g., k=10 -> values_8
        if y not in df.columns:
            skips[k] = "missing outcome column"
            continue
        # 1) filter first: drop NAs in y (and x if used)
        df_k = df.copy().dropna(subset=[y])
        if x is not None:
            if x not in df_k.columns:
                skips[k] = f"missing covariate '{x}'"
                continue
            df_k = df_k.dropna(subset=[x])
        # 2) baseline coding AFTER filtering
        categories = list(df_k[group_col].dropna().unique())
        if baseline not in categories:
            skips[k] = f"baseline '{baseline}' absent after filtering"
            continue
        categories.remove(baseline)
        categories = [baseline] + categories
        df_k[group_col] = pd.Categorical(df_k[group_col], categories=categories, ordered=True)
        # ensure enough data and at least two groups
        if df_k[group_col].nunique() < 2:
            skips[k] = "only one group present after filtering"
            continue
        if len(df_k) < min_rows:
            skips[k] = "too few rows after filtering"
            continue
        # 3) cast bild to categorical AFTER filtering
        if "bild" in df_k.columns:
            df_k["bild"] = df_k["bild"].astype("category")
        # 4) build independent_vars exactly as you do
        independent_vars = ["group", "age", "female"]
        if x is not None:
            independent_vars.append(x)
        if "bild" in df_k.columns:
            independent_vars.append("bild")
        if "education" in df_k.columns:
            independent_vars.append("education")
        # 5) your exact lines for formula + GEE fit (applied to filtered df_k)
        try:
            formula = f"{y} ~ {' + '.join(independent_vars)}"
            model_gee = sm.GEE.from_formula(
                formula, groups=df_k[gee_groups_col], data=df_k, family=sm.families.Gaussian()
            )
            result = model_gee.fit()
        except Exception as e:
            skips[k] = f"GEE fit failed: {type(e).__name__}"
            continue
        # keep only non-baseline group terms
        for term in result.params.index:
            if term.startswith(f"{group_col}[T.") or term.startswith("group[T."):
                grp = term.split("[T.")[-1].rstrip("]")
                rows.append({
                    "k": k,
                    "group": grp,
                    "coefficient": float(result.params[term]),
                    "p_value": float(result.pvalues[term]),
                })
    out = pd.DataFrame(rows).sort_values(["group", "k"]).reset_index(drop=True)
    # optional plot with red star markers for significant points
    if plot and not out.empty:
        plt.figure(figsize=(7, 4.2))
        for g in out["group"].unique():
            gg = out[out["group"] == g]
            # draw the line + circular points
            plt.plot(gg["k"], gg["coefficient"], marker="o", linewidth=1.8, label=g)
            # overlay red stars where significant
            sig = gg[gg["p_value"] < alpha_sig]
            if not sig.empty:
                plt.scatter(
                    sig["k"], sig["coefficient"],
                    marker="*", s=star_size,
                    color="red", edgecolors="none", zorder=3
                )
        plt.axhline(0, ls="--", color="gray")
        plt.xlabel("k")
        plt.ylabel("Group coefficient")
        title_model = ('fastText' if y_kind == 'fast' else ('BERT' if y_kind == 'bert' else prefix))
        plt.title(f"Group effect vs k — {title_model}")
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = [ {"SZ":"SSD","FES":"FEP"}.get(l, l) for l in labels ]
        plt.legend(handles, labels, title="Group", frameon=False)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        # Save the plot
        plt.savefig(path_results+f'coeff_regression_{y_kind}_{name_sample}_k.jpg', dpi=500)
        plt.show()
    return out



""" COEFFICIENTS OF GROUPS AND PLOTS """

# RESULTS WITH CERTAIN K

result_ge_1 = gee_model_results(df_ge, 'values_id_fast_5',  x='n_content_u', baseline='HC', name_data= 'german_unique_id')
result_ge_2 = gee_model_results(df_ge, 'values_id_bert_10',  x='n_tokens', baseline='HC', name_data= 'german_unique_id')

result_sp_1 = gee_model_results(df_sp, 'values_id_fast_7', 'n_content_u', 'HC', 'spanish_unique_id')
result_sp_2 = gee_model_results(df_sp, 'values_id_bert_10', 'n_tokens', 'HC', 'spanish_unique_id')

result_en_1 = gee_model_results(df_en, 'values_id_fast_5', 'n_content_u', 'HC', 'english_unique_id')
result_en_2 = gee_model_results(df_en, 'values_id_bert_10', 'n_tokens', 'HC', 'english_unique_id')


# AVERAGE VALUES IN GERMAN DATA
num_cols  = df_ge.select_dtypes(include='number').columns
other_aggs = {'group':'first', 'education':'first', 'female':'first'}  # adjust to your columns
df_ge = (df_ge
         .groupby('name', as_index=False)
         .agg({**{c:'mean' for c in num_cols}, **other_aggs}))


# PLOT OF COEFFICIENTS

out_fast_ge = gee_group_coeff_over_k(
    df=df_ge,                 
    y_kind="fast",            # uses columns like values_id_fast_k
    start=10, end=25,         # k range
    x="n_content_u",          # optional covariate
    baseline="HC",            # baseline group
    group_col="group",        # column with group labels
    gee_groups_col="group",   # clusters by group
    plot=True,                # draw coefficient vs k
    name_sample='german'      # language sample name in the file
)

out_fast_sp = gee_group_coeff_over_k(
    df=df_sp,                 
    y_kind="fast",            # uses columns like values_id_fast_k
    start=10, end=25,         # k range
    x="n_content_u",          # optional covariate
    baseline="HC",            # baseline group
    group_col="group",        # column with group labels
    gee_groups_col="group",   # clusters by group 
    plot=True,                # draw coefficient vs k
    name_sample='spanish'     # language sample name in the file
)

out_fast_en = gee_group_coeff_over_k(
    df=df_en,                
    y_kind="fast",            # uses columns like values_id_fast_k
    start=10, end=25,         # k range
    x="n_content_u",          # optional covariate
    baseline="HC",            # baseline group
    group_col="group",        # column with group labels
    gee_groups_col="group",   # clusters by group
    plot=True,                # draw coefficient vs k
    name_sample='english'     # language sample name in the file
)


out_bert_ge = gee_group_coeff_over_k(
    df=df_ge,                 # 
    y_kind="bert",            # uses columns like values_id_fast_k
    start=20, end=40,         # k range
    x="n_content_u",          # optional covariate
    baseline="HC",            # baseline group
    group_col="group",        # column with group labels
    gee_groups_col="group",   # clusters by group
    plot=True,                # draw coefficient vs k
    name_sample='german'      # language sample name in the file
)

out_bert_sp = gee_group_coeff_over_k(
    df=df_sp,                 # 
    y_kind="bert",            # uses columns like values_id_fast_k
    start=20, end=40,         # k range
    x="n_content_u",          # optional covariate
    baseline="HC",            # baseline group
    group_col="group",        # column with group labels
    gee_groups_col="group",   # clusters by group
    plot=True,                # draw coefficient vs k
    name_sample='spanish'     # language sample name in the file
)

out_bert_en = gee_group_coeff_over_k(
    df=df_en,                 #
    y_kind="bert",            # uses columns like values_id_fast_k
    start=20, end=40,         # k range
    x="n_content_u",          # optional covariate
    baseline="HC",            # baseline group
    group_col="group",        # column with group labels
    gee_groups_col="group",   # clusters by group
    plot=True,                # draw coefficient vs k
    name_sample='english'     # language sample name in the file
)


""" PLOT & SAVE - ID VERSUS K """

# GERMAN
plot_id_wide(df_ge, prefix="values_id_fast_", band="sd", name='german_fasttext')
plot_id_wide(df_ge, prefix="values_id_bert_", band="sd", name='german_bert')

# SPANISH
plot_id_wide(df_sp, prefix="values_id_fast_", band="sd", name='spanish_fasttext')
plot_id_wide(df_sp, prefix="values_id_bert_", band="sd", name='spanish_bert')

# ENGLISH
plot_id_wide(df_en, prefix="values_id_fast_", band="sd", name='english_fasttext')
plot_id_wide(df_en, prefix="values_id_bert_", band="sd", name='english_bert')



""" COMBINE FIGURES IN PANEL """

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

# FIGURE S4

# Load images
img_A = Image.open(path_results+'sensitivity_all_german_fasttext.jpg')
img_B = Image.open(path_results+'sensitivity_all_german_bert.jpg')

img_C = Image.open(path_results+'sensitivity_all_spanish_fasttext.jpg')
img_D = Image.open(path_results+'sensitivity_all_spanish_bert.jpg')

img_E = Image.open(path_results+'sensitivity_all_english_fasttext.jpg')
img_F = Image.open(path_results+'sensitivity_all_english_bert.jpg')

# Create 3x2 grid (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(8, 12), gridspec_kw={'hspace': -0.6})
fig.tight_layout()
# List of images and labels
images = [img_A, img_B, img_C, img_D, img_E, img_F]
labels = ['A', 'B', 'C', 'D', 'E', 'F']
# Plot images
for ax, img, label in zip(axes.flat, images, labels):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=17, va='top', ha='left')
plt.subplots_adjust(hspace=0.01)
plt.savefig(path_results+"FigureS4.jpg", dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()



# FIGURE S5

# Load images
img_A = Image.open(path_results+'coeff_regression_fast_german_k.jpg')
img_B = Image.open(path_results+'coeff_regression_bert_german_k.jpg')

img_C = Image.open(path_results+'coeff_regression_fast_spanish_k.jpg')
img_D = Image.open(path_results+'coeff_regression_bert_spanish_k.jpg')

img_E = Image.open(path_results+'coeff_regression_fast_english_k.jpg')
img_F = Image.open(path_results+'coeff_regression_bert_english_k.jpg')

# Create 3x2 grid (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(8, 12), gridspec_kw={'hspace': -0.6})
fig.tight_layout()
# List of images and labels
images = [img_A, img_B, img_C, img_D, img_E, img_F]
labels = ['A', 'B', 'C', 'D', 'E', 'F']
# Plot images
for ax, img, label in zip(axes.flat, images, labels):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=17, va='top', ha='left')
plt.subplots_adjust(hspace=0.01)
plt.savefig(path_results+"FigureS5.jpg", dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
