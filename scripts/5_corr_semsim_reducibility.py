"""
@author: u164233
"""

# IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM


# PICTURES
# German:  pict 1: boy, pict 2: countryside, pict 4: couple, pict 6: old woman
# Spanish: pict 1: countryside, pict2: couple, pict3: bridge


# PATH INPUT
path_input     = '/PATH/TO/PRIVATE_DATA/german/texts/'
path_demo      = '/PATH/TO/PRIVATE_DATA/german/'
path_results   = '/PATH/TO/RESULTS/'
path_santander = '/PATH/TO/PRIVATE_DATA/spanish/'
path_london    = '/PATH/TO/PRIVATE_DATA/english/'


""" READ DATA """

# GERMAN REDUCTION
df_ge = pd.read_pickle(path_results+'df_words_dim_reduction_german.pkl')
# ONLY SSD AND HC
df_ge = df_ge[df_ge['group']!='MDD']
df_ge = df_ge.rename(columns={'bild': 'picture'})


# SPANISH REDUCTION
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
groups_spa = groups_spa.drop_duplicates()
# READ NUMBER OF COMPONENTS AND AVERAGE
df_spa = pd.read_pickle(path_results+'df_words_dim_reduction_spanish.pkl')
# DEMOGRAPHICS
df_demo_spa = pd.read_pickle(path_santander+'df_demographics_santander.pkl')
df_spa = pd.merge(df_spa, df_demo_spa, on='name', how='inner')
df_spa.loc[df_spa['group'] == 'SZ', 'group'] = 'SSD'
df_spa = df_spa.rename(columns={'bild': 'picture'})
df_spa['name'] = df_spa['name'].str.replace('DS_', '', regex=False)


# ENGLISH REDUCTION

# Import the .xlsx file into a DataFrame
df_lon = pd.read_excel(path_london+'summary.xlsx')
# Filter rows based on conditions
df_lon = df_lon[(df_lon['Role'] == 'interviewee') & (df_lon['Meta'] != 'Off-task')]
# Drop the columns 'Role', 'Meta', and 'Time'
df_lon = df_lon.drop(['Role', 'Meta', 'Time'], axis=1)
# Convert column names to lowercase
df_lon.columns = df_lon.columns.str.lower()
# GROUPS
# import the .xlsx of groups
df_g = pd.read_excel(path_london+'groups.xlsx')
# MERGE THE DATA WITH GROUPS
df_lon = df_lon.merge(df_g, on='par', how='inner')
df_lon = df_lon[['par', 'group', 'picture', 'utterance']]
df_lon = df_lon.rename(columns = {'par': 'name'})
df_g   = df_g.rename(columns = {'par': 'name'})
# READ NUMBER OF COMPONENTS AND AVERAGE
df_eng = pd.read_pickle(path_results+'df_words_dim_reduction_english.pkl')
# DEMOGRAPHICS
df_demo_eng = pd.read_pickle(path_london+'df_demographics_london.pkl')
df_eng = pd.merge(df_eng, df_demo_eng, on='name', how='inner')
df_eng = df_eng.rename(columns={'bild': 'picture'})



""" GRAPHS CORRELATIONS AND REGRESSIONS """
import matplotlib

# Set Times New Roman and font size globally
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14

# SCATTER PLOT
def scatter_plot_group(df, var1, var2, group=None, pict=None):
    # Filter the DataFrame based on 'group' and 'pict'
    if group is not None:
        df = df[df['group'] == group]
    if pict is not None:
        df = df[df['bild'] == pict]
    # Scatter plot with different colors for different groups
    sns.set_palette("colorblind")
    if group is None:
        grouped = df.groupby('group')
        for name, group_data in grouped:
            plt.scatter(group_data[var1], group_data[var2], label=f'{name}')
        plt.legend()
    else:
        sns.scatterplot(data=df, x=var1, y=var2, hue='group', palette='colorblind')
    # Add labels and title
    plt.xlabel(f'{var1}')
    plt.ylabel(f'{var2}')
    plt.title(f'{var2} vs {var1}')
    corr, p_value = pearsonr(df[var1], df[var2])
    print(f'Correlation coefficient between {var1} and {var2}: {corr}')
    print(f'P-value of this correlation: {p_value}')
    # Display correlation if any
    if p_value < 0.05:  # Adjust the threshold
        sns.regplot(x=df[var1], y=df[var2], scatter=False)
    #plt.savefig(path_results+'scatter_corr.jpg', dpi=500)
    plt.show()


# MATRIX OF CORRELATIONS
def correlation_and_pvalue_matrix(df, variables, group=None, group_column='group'):
    """
    Computes Pearson correlation coefficients and p-values for a given list of 4 variables in a DataFrame,
    optionally filtered by a group.
    Parameters:
    - df: pandas DataFrame containing the data
    - variables: list of 4 column names (strings)
    - group: group value to filter the DataFrame (optional)
    - group_column: name of the column indicating the group (default: 'group')
    Returns:
    - corr_matrix: 4x4 DataFrame of Pearson correlation coefficients
    - pval_matrix: 4x4 DataFrame of corresponding p-values
    """
    if len(variables) != 5:
        raise ValueError("Exactly 5 variables must be provided.")
    # Filter by group if specified
    if group is not None:
        df = df[df[group_column] == group]
    corr_matrix = pd.DataFrame(np.zeros((5, 5)), columns=variables, index=variables)
    pval_matrix = pd.DataFrame(np.zeros((5, 5)), columns=variables, index=variables)
    for i in range(5):
        for j in range(5):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
                pval_matrix.iloc[i, j] = 0.0
            else:
                corr, pval = pearsonr(df[variables[i]], df[variables[j]])
                corr_matrix.iloc[i, j] = corr
                pval_matrix.iloc[i, j] = pval
    return corr_matrix, pval_matrix


# HEATMAP OF CORRELATIONS
def plot_correlations(corr_matrix, pval_matrix, alpha=0.05, label_map=None, name=None):
    """
    Plots a heatmap of Pearson correlation coefficients, showing only significant values in the upper triangle.

    Parameters:
    - corr_matrix: DataFrame of correlation coefficients
    - pval_matrix: DataFrame of corresponding p-values
    - alpha: significance threshold (default: 0.05)
    - title: title of the plot
    """
    # Apply label mapping if provided
    if label_map:
        corr_matrix = corr_matrix.rename(index=label_map, columns=label_map)
        pval_matrix = pval_matrix.rename(index=label_map, columns=label_map)
    # Create a mask for non-significant values and the lower triangle + diagonal
    #mask = np.tril(np.ones_like(corr_matrix, dtype=bool)) | (pval_matrix >= alpha)
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1) | (
        (pval_matrix >= alpha) & ~np.eye(len(pval_matrix), dtype=bool)
    )
    # Set up the plot
    plt.figure(figsize=(8, 6))
    sns.set(style='white')
    # Draw the heatmap
    ax = sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        #cmap='coolwarm',
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        cbar=True,
        square=True,
        linewidths=0,
        linecolor='gray'
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path_results+f'heatmap_corr_{name}.jpg', dpi=1000)
    plt.show()


# MIXED-EFFECTS MODEL
def mixed_model_results(df, y, x, baseline, name_data):
    categories = list(df['group'].unique())
    categories.remove(f'{baseline}')
    categories = [f'{baseline}'] + categories
    df['group'] = pd.Categorical(df['group'], categories=categories, ordered=True)
    #formula = f'{y} ~ {x} + n_comp_90 + group + age + education + female'
    independent_vars = [f'{x}', 'n_comp_90 + group + age + female']  # List of independent variables
    if 'education' in df.columns:
        independent_vars.append('education')
    # Join the independent variables into a formula string
    formula = f"{y} ~ {' + '.join(independent_vars)}"
    #model_r = MixedLM.from_formula(f'{y} ~ group + bild + age + education + female + {x}', df, groups=df["name"])
    model_r = MixedLM.from_formula(formula, df, groups=df["name"])
    result = model_r.fit()
    print(result.summary())
    # Coefficients and confidence intervals
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
    # Dataframe for plotting
    coef_df = pd.DataFrame({
        'Coefficient': coefs.index,
        'Estimate': coefs.values,
        'CI_lower': conf_int[0],
        'CI_upper': conf_int[1],
        'p_value': p_values
    })
    # Mapping original coefficient names to custom names
    custom_names = {
        'group[T.SSD]': 'SSD',
        'bild[T.2]': 'Picture 2',
        'bild[T.4]': 'Picture 4',
        'bild[T.6]': 'Picture 6',
        'age': 'Age',
        'n_comp_90': 'N Components',
        'education': 'Education',
        'female': 'Female',
        f'{x}': 'N content words'
    }
    #coef_df['Coefficient'] = coef_df['Coefficient'].replace(custom_names)
    # Apply the custom names and append p-values
    coef_df['Coefficient'] = coef_df.apply(
        lambda row: f"{custom_names.get(row['Coefficient'], row['Coefficient'])} (p={format_pval(row['p_value'])})", axis=1
    )
    coef_df = coef_df[~coef_df['Coefficient'].str.contains('Intercept|Group Var', case=False)]
    # Setting font and size
    plt.rcParams['font.family'] = 'Times New Roman'  
    plt.rcParams['font.size'] = 15
    # Coefficients with confidence intervals
    plt.figure(figsize=(8, 6))
    sns.pointplot(x="Estimate", y="Coefficient", data=coef_df, linestyle='none', color='b')
    for i in range(coef_df.shape[0]):
       plt.plot([coef_df['CI_lower'].iloc[i], coef_df['CI_upper'].iloc[i]], [i, i], color='gray', lw=2)
    plt.axvline(0, ls='--', color='red')
    #plt.title('Coefficients with Confidence Intervals')
    plt.xlabel('Estimate')
    plt.ylabel('Coefficients')
    plt.tight_layout()
    #plt.savefig(path_results+f'coeff_comp_{name_data}.jpg', dpi=500)
    plt.show()
    print(result)
    return result


def mixed_model_results(df, y, x, baseline, name_data):
    # Drop rows with missing values in essential columns
    required_cols = ['name', 'group', x, 'age', 'female', y]
    df = df.dropna(subset=required_cols).copy()
    # Reset index to avoid indexing issues in MixedLM
    df = df.reset_index(drop=True)
    categories = list(df['group'].unique())
    categories.remove(f'{baseline}')
    categories = [f'{baseline}'] + categories
    df['group'] = pd.Categorical(df['group'], categories=categories, ordered=True)
    # Build independent variables dynamically
    independent_vars = [f'{x}', 'group', 'picture', 'age', 'female']
    if 'education' in df.columns:
        independent_vars.append('education')
        df = df.dropna(subset=['education'])
    # Build formula
    formula = f"{y} ~ {' + '.join(independent_vars)}"
    # Fit mixed-effects model
    model_r = MixedLM.from_formula(formula, df, groups=df["name"])
    result = model_r.fit()
    print(result.summary())
    # Prepare data for plotting
    coefs = result.params
    conf_int = result.conf_int()
    p_values = result.pvalues

    def format_pval(pval):
        if pval < 0.001:
            return "<0.001***"
        elif pval < 0.01:
            return f"{pval:.3f}**"
        elif pval < 0.05:
            return f"{pval:.3f}*"
        else:
            return f"{pval:.3f}"

    coef_df = pd.DataFrame({
        'Coefficient': coefs.index,
        'Estimate': coefs.values,
        'CI_lower': conf_int[0],
        'CI_upper': conf_int[1],
        'p_value': p_values
    })
    # Custom names for known coefficients
    custom_names = {
        'group[T.SSD]': 'SSD',
        'age': 'Age',
        'n_comp_90': 'N Components',
        'education': 'Education',
        'female': 'Female'
        }
    # Apply custom names and p-values
    coef_df['Coefficient'] = coef_df.apply(
        lambda row: f"{custom_names.get(row['Coefficient'], row['Coefficient'])} (p={format_pval(row['p_value'])})",
        axis=1
    )
    coef_df = coef_df[~coef_df['Coefficient'].str.contains('Intercept|Group Var', case=False)]
    # Plot
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(8, 6))
    sns.pointplot(x="Estimate", y="Coefficient", data=coef_df, linestyle='none', color='b')
    for i in range(coef_df.shape[0]):
        plt.plot([coef_df['CI_lower'].iloc[i], coef_df['CI_upper'].iloc[i]], [i, i], color='gray', lw=2)
    plt.axvline(0, ls='--', color='red')
    plt.xlabel('Estimate')
    plt.ylabel('Coefficients')
    plt.tight_layout()
    plt.show()
    print(result)
    return result



""" 1. CORRELATIONS """


# LABELS
variables = ['n_comp_90', 'var_exp_2', 'id_bert', 'mean_semsim', 'dispersion']
label_map = {
    'mean_semsim': 'Semantic sim',
    'm_dispersion': 'mov_dispersion',  
    'dispersion': 'Dispersion', 
    'n_comp_90': 'N Comp 90',
    'var_exp_2': 'Var exp 2',
    'id_bert': 'BERT ID'
}


# GERMAN HEATMAP
corr_SSD, pvals_SSD = correlation_and_pvalue_matrix(df_ge, variables, group='SSD')
corr_HC, pvals_HC   = correlation_and_pvalue_matrix(df_ge, variables, group='HC')
plot_correlations(corr_SSD, pvals_SSD, alpha=0.05, label_map=label_map, name='German_SSD')
plot_correlations(corr_HC, pvals_HC, alpha=0.05, label_map=label_map, name='German_HC')


# GERMAN SCATTER PLOTS
scatter_plot_group(df_ge, 'mean_semsim', 'n_comp_90', group='SSD')
scatter_plot_group(df_ge, 'mean_semsim', 'id_bert', group='SSD')
scatter_plot_group(df_ge, 'mean_semsim', 'n_comp_90_bert', group='SSD')
scatter_plot_group(df_ge, 'mean_semsim', 'n_comp_90', group='HC')
scatter_plot_group(df_ge, 'mean_semsim', 'id_bert', group='HC')
scatter_plot_group(df_ge, 'mean_semsim', 'n_comp_90_bert', group='HC')
scatter_plot_group(df_ge, 'dispersion', 'n_comp_90')
scatter_plot_group(df_ge, 'm_dispersion', 'n_comp_90', group='SSD')
scatter_plot_group(df_ge, 'm_dispersion', 'n_comp_90', group='HC')



# SPANISH HEATMAP
corr_SSD, pvals_SSD = correlation_and_pvalue_matrix(df_spa, variables, group='SSD')
corr_HC, pvals_HC   = correlation_and_pvalue_matrix(df_spa, variables, group='HC')
plot_correlations(corr_SSD, pvals_SSD, alpha=0.05, label_map=label_map, name='Spanish_SSD')
plot_correlations(corr_HC, pvals_HC, alpha=0.05, label_map=label_map, name='Spanish_HC')


# SPANISH SCATTERPLOTS
scatter_plot_group(df_spa, 'mean_semsim', 'n_comp_90', group='SSD')
scatter_plot_group(df_spa, 'mean_semsim', 'id_bert', group='SSD')
scatter_plot_group(df_spa, 'mean_semsim', 'n_comp_90_bert', group='SSD')
scatter_plot_group(df_spa, 'mean_semsim', 'n_comp_90', group='HC')
scatter_plot_group(df_spa, 'mean_semsim', 'id_bert', group='HC')
scatter_plot_group(df_spa, 'mean_semsim', 'n_comp_90_bert', group='HC')
scatter_plot_group(df_spa, 'dispersion', 'n_comp_90')
scatter_plot_group(df_spa, 'm_dispersion', 'n_comp_90', group='SSD')
scatter_plot_group(df_spa, 'm_dispersion', 'n_comp_90', group='HC')



# ENGLIS HEATMAPS
corr_FEP, pvals_FEP = correlation_and_pvalue_matrix(df_eng, variables, group='FES')
corr_CHR, pvals_CHR = correlation_and_pvalue_matrix(df_eng, variables, group='CHR')
corr_CS, pvals_CS   = correlation_and_pvalue_matrix(df_eng, variables, group='CS')
corr_HC, pvals_HC   = correlation_and_pvalue_matrix(df_eng, variables, group='HC')
plot_correlations(corr_FEP, pvals_FEP, alpha=0.05, label_map=label_map, name='English_FEP')
plot_correlations(corr_CHR, pvals_CHR, alpha=0.05, label_map=label_map, name='English_CHR')
plot_correlations(corr_CS, pvals_CS, alpha=0.05, label_map=label_map, name='English_CS')
plot_correlations(corr_HC, pvals_HC, alpha=0.05, label_map=label_map, name='English_HC')


# ENGLISH SCATTERPLOTS
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90', group='FES')
scatter_plot_group(df_eng, 'mean_semsim', 'id_bert', group='FES')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90_bert', group='FES')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90', group='CHR')
scatter_plot_group(df_eng, 'mean_semsim', 'id_bert', group='CHR')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90_bert', group='CHR')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90', group='CS')
scatter_plot_group(df_eng, 'mean_semsim', 'id_bert', group='CS')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90_bert', group='CS')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90', group='HC')
scatter_plot_group(df_eng, 'mean_semsim', 'id_bert', group='HC')
scatter_plot_group(df_eng, 'mean_semsim', 'n_comp_90_bert', group='HC')
scatter_plot_group(df_eng, 'dispersion', 'n_comp_90')
scatter_plot_group(df_eng, 'm_dispersion', 'n_comp_90', group='FES')
scatter_plot_group(df_eng, 'm_dispersion', 'n_comp_90', group='CHR')
scatter_plot_group(df_eng, 'm_dispersion', 'n_comp_90', group='CS')
scatter_plot_group(df_eng, 'm_dispersion', 'n_comp_90', group='HC')



""" 2. REGRESSION SEMSIM ~ N COMP 90 """

# SELECT VARIABLES
var_use = ['name', 'group', 'picture', 'mean_semsim', 'max_semsim', 'n_comp_90', 'var_exp_2', 'id_bert', 'n_content_u', 'n_tokens', 'age', 'female']
df_ge  = df_ge[var_use]
df_spa = df_spa[var_use]
df_eng = df_eng[var_use]

# APPEND DATAFRAMES
df_all = pd.concat([df_ge, df_spa, df_eng], ignore_index=True)

# REGRESS
result_all = mixed_model_results(df_all, 'mean_semsim', 'n_comp_90', 'HC', 'semsim_ncomp')

result_ge  = mixed_model_results(df_ge, 'mean_semsim', 'id_bert', 'HC', 'semsim_ncomp')

result_spa = mixed_model_results(df_spa, 'mean_semsim', 'id_bert', 'HC', 'semsim_ncomp')

result_eng = mixed_model_results(df_eng, 'mean_semsim', 'id_bert', 'HC', 'semsim_ncomp')

