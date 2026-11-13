# -*- coding: utf-8 -*-
"""
@author: u164233
"""

# IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
from PIL import Image


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
df_spa = df_spa.groupby('name').mean(numeric_only=True)
df_spa = df_spa.merge(groups_spa, on='name', how='left')
df_spa = df_spa[(df_spa['group'] == 'HC') | (df_spa['group'] == 'SZ')]
# DEMOGRAPHICS
df_demo_spa = pd.read_pickle(path_santander+'df_demographics_santander.pkl')
df_spa = pd.merge(df_spa, df_demo_spa, on='name', how='inner')


# Calculating statistics
from scipy import stats
mean_education = df_spa['education'].mean()
df_spa['education'].fillna(mean_education, inplace=True)
summary = df_spa.groupby('group').agg(
    count=('group', 'size'),
    avg_age=('age', 'mean'),
    std_age=('age', 'std'),
    avg_education=('education', 'mean'),
    std_education=('education', 'std'),
    avg_c_words=('n_content_u', 'mean'),
    std_c_words=('n_content_u', 'std'),
    proportion_female=('female', lambda x: x.mean())
).reset_index()
# Performing t-tests for age and education
group_A = df_spa[df_spa['group'] == 'SZ']
group_B = df_spa[df_spa['group'] == 'HC']
t_test_age = stats.ttest_ind(group_A['age'], group_B['age'])
t_test_education = stats.ttest_ind(group_A['education'], group_B['education'])
t_test_n_words = stats.ttest_ind(group_A['n_content_u'], group_B['n_content_u'])
# Performing chi-square test for female proportion
contingency_table = pd.crosstab(df_spa['group'], df_spa['female'])
chi2, p, _, _ = stats.chi2_contingency(contingency_table)
# Adding t-test results to summary
summary['t_test_age_statistic'] = t_test_age.statistic
summary['t_test_age_p_value'] = t_test_age.pvalue
summary['t_test_education_statistic'] = t_test_education.statistic
summary['t_test_education_p_value'] = t_test_education.pvalue
summary['t_test_n_words_statistic'] = t_test_n_words.statistic
summary['t_test_n_words_p_value'] = t_test_n_words.pvalue
summary['chi2_statistic'] = chi2
summary['chi2_p_value'] = p
print(summary)



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
df_eng = df_eng.groupby('name').mean(numeric_only=True)
df_eng = df_eng.merge(df_g, on='name', how='left')
# DEMOGRAPHICS
df_demo_eng = pd.read_pickle(path_london+'df_demographics_london.pkl')
df_eng = pd.merge(df_eng, df_demo_eng, on='name', how='inner')

# MIX CS and FEP
#df_eng['group'] = df_eng['group'].str.strip()
#df_eng.loc[df_eng['group'] == 'CS', 'group'] = 'SSD'
#df_eng.loc[df_eng['group'] == 'FES', 'group'] = 'SSD'


# DEMOGRAPHICS STATS
summary = df_eng.groupby('group').agg(
    count=('group', 'size'),
    avg_age=('age', 'mean'),
    std_age=('age', 'std'),
    proportion_female=('female', lambda x: x.mean())
).reset_index()
# Performing t-tests for age and education
group_A = df_eng[df_eng['group'] == 'HC']
group_B = df_eng[df_eng['group'] == 'CS']
group_C = df_eng[df_eng['group'] == 'CHR']
group_D = df_eng[df_eng['group'] == 'FES']
t_test_age_1 = stats.ttest_ind(group_A['age'], group_B['age'])
t_test_age_2 = stats.ttest_ind(group_A['age'], group_C['age'])
t_test_age_3 = stats.ttest_ind(group_A['age'], group_D['age'])
t_test_age_4 = stats.ttest_ind(group_B['age'], group_C['age'])
t_test_age_5 = stats.ttest_ind(group_B['age'], group_D['age'])
t_test_age_6 = stats.ttest_ind(group_C['age'], group_D['age'])
# Performing chi-square test for female proportion
contingency_table = pd.crosstab(df_eng['group'], df_eng['female'])
chi2, p, _, _ = stats.chi2_contingency(contingency_table)
# Adding t-test results to summary
summary['t_test_age_statistic_1'] = t_test_age_1.statistic
summary['t_test_age_p_value_1'] = t_test_age_1.pvalue
summary['t_test_age_statistic_2'] = t_test_age_2.statistic
summary['t_test_age_p_value_2'] = t_test_age_2.pvalue
summary['t_test_age_statistic_3'] = t_test_age_3.statistic
summary['t_test_age_p_value_3'] = t_test_age_3.pvalue
summary['t_test_age_statistic_4'] = t_test_age_4.statistic
summary['t_test_age_p_value_4'] = t_test_age_4.pvalue
summary['t_test_age_statistic_5'] = t_test_age_5.statistic
summary['t_test_age_p_value_5'] = t_test_age_5.pvalue
summary['t_test_age_statistic_6'] = t_test_age_6.statistic
summary['t_test_age_p_value_6'] = t_test_age_6.pvalue
summary['chi2_statistic'] = chi2
summary['chi2_p_value'] = p
print(summary)
p_values_summary = summary[['group', 't_test_age_p_value_1',
                            't_test_age_p_value_2',
                            't_test_age_p_value_3',
                            't_test_age_p_value_4',
                            't_test_age_p_value_5',
                            't_test_age_p_value_6',
                            'chi2_p_value']]
p_values_summary.columns = ['group', 
                             'HC - CS (p-value)', 
                             'HC - CHR (p-value)', 
                             'HC - FEP (p-value)', 
                             'CS - CHR (p-value)', 
                             'CS - FEP (p-value)', 
                             'CHR - FEP (p-value)', 
                             'chi2 (p-value)']


""" FUNCTIONS """

# MIXED-EFFECTS MODEL
def mixed_model_results(df, y, x=None, baseline='HC', name_data='data'):
    categories = list(df['group'].unique())
    categories.remove(f'{baseline}')
    categories = [f'{baseline}'] + categories
    df['group'] = pd.Categorical(df['group'], categories=categories, ordered=True)
    #formula = f'{y} ~ {x} + group + age + education + female'
    # Start with mandatory variables
    independent_vars = ['group', 'age', 'female']
    # Optionally include 'x' if provided
    if x is not None:
        independent_vars.append(f'{x}')
    if 'bild' in df.columns:
        df['bild'] = df['bild'].astype('category')
        independent_vars.append('bild')
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
        'education': 'Education',
        'female': 'Female',
    }
    if x is not None:
        custom_names[f'{x}'] = 'N tokens'
    # Exclude coefficients from display before renaming
    exclude_terms = ['Intercept', 'Group Var'] + [col for col in coefs.index if col.startswith('bild')]
    coef_df = coef_df[~coef_df['Coefficient'].isin(exclude_terms)]
    # Now apply custom names and p-values formatting
    coef_df['Coefficient'] = coef_df.apply(
        lambda row: f"{custom_names.get(row['Coefficient'], row['Coefficient'])} (p={format_pval(row['p_value'])})", axis=1
    )
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
    plt.savefig(path_results+f'coeff_comp_{name_data}.jpg', dpi=500)
    plt.show()
    print(result)
    return result


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
    plt.savefig(path_results+f'coeff_gee_{name_data}.jpg', dpi=500)
    plt.show()
    return result





""" SCATTER CORRELATIONS """


def scatter_with_correlation(df, x_var, y_var, title=None):
    """
    Plots a scatter plot of two variables in a DataFrame using Seaborn 
    and prints the Pearson correlation coefficient.

    Parameters:
    - df: pandas DataFrame
    - x_var: str, name of the column for x-axis
    - y_var: str, name of the column for y-axis
    - title: optional str, plot title
    """
    sns.scatterplot(x=x_var, y=y_var, data=df)
    plt.title(title or f'Scatter plot of {x_var} vs {y_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True)
    plt.show()
    corr = df[x_var].corr(df[y_var])
    print(f"Correlation between {x_var} and {y_var}: {corr:.2f}")


scatter_with_correlation(df_ge, 'n_content_u', 'id_fast')              # 0.92
scatter_with_correlation(df_ge, 'n_content_u', 'id_fast_corr_matrix')  # 0.58
scatter_with_correlation(df_ge, 'n_content_u', 'id_fast_sim_matrix')   # 0.79

scatter_with_correlation(df_ge, 'n_tokens', 'n_comp_90_bert')          # 0.93
scatter_with_correlation(df_ge, 'n_tokens', 'var_exp_2_bert')          # -0.53

scatter_with_correlation(df_ge, 'n_tokens', 'id_bert')              # 0.26
scatter_with_correlation(df_ge, 'n_tokens', 'id_bert_corr_matrix')  # 0.55
scatter_with_correlation(df_ge, 'n_tokens', 'id_bert_sim_matrix')   # 0.54


""" RESULTS """

# GERMAN
result_ge_1 = mixed_model_results(df_ge, 'n_comp_90', 'n_content_u', 'HC', 'german_unique')
result_ge_2 = mixed_model_results(df_ge, 'var_exp_2', 'n_content_u', 'HC', 'german_unique_var')
result_ge_3 = mixed_model_results(df_ge, 'id_fast', 'n_content_u', 'HC', 'german_unique_id')



result_ge_6 = mixed_model_results(df_ge, 'n_comp_90_bert', 'n_tokens', 'HC', 'german_bert')
result_ge_7 = mixed_model_results(df_ge, 'var_exp_2_bert', 'n_tokens', 'HC', 'german_bert_var')
result_ge_8 = mixed_model_results(df_ge, 'id_bert', x=None, baseline='HC', name_data='german_bert_id')




# SPANISH
result_sp_1 = gee_model_results(df_spa, 'n_comp_90', 'n_content_u', 'HC', 'spanish_unique')
result_sp_2 = gee_model_results(df_spa, 'var_exp_2', 'n_content_u', 'HC', 'spanish_unique_var')
result_sp_3 = gee_model_results(df_spa, 'id_fast', 'n_content_u', 'HC', 'spanish_unique_id')


result_sp_6 = gee_model_results(df_spa, 'n_comp_90_bert', 'n_tokens', 'HC', 'spanish_bert')
result_sp_7 = gee_model_results(df_spa, 'var_exp_2_bert', 'n_tokens', 'HC', 'spanish_bert_var')
result_sp_8 = gee_model_results(df_spa, 'id_bert', x=None, baseline='HC', name_data='spanish_bert_id')

# ENGLISH
result_en_1 = gee_model_results(df_eng, 'n_comp_90', 'n_content_u', 'HC', 'english_unique')
result_en_2 = gee_model_results(df_eng, 'var_exp_2', 'n_content_u', 'HC', 'english_unique_var')
result_en_3 = gee_model_results(df_eng, 'id_fast', 'n_content_u', 'HC', 'english_unique_id')


result_en_6 = gee_model_results(df_eng, 'n_comp_90_bert', 'n_tokens', 'HC', 'english_bert')
result_en_7 = gee_model_results(df_eng, 'var_exp_2_bert', 'n_tokens', 'HC', 'english_bert_var')
result_sp_8 = gee_model_results(df_eng, 'id_bert', x=None, baseline='HC', name_data='english_bert_id')





""" COMBINE FIGURES IN PANEL """

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

# FIGURE 1

# Load images
img1 = Image.open(path_results+'coeff_comp_german_unique.jpg')
img2 = Image.open(path_results+'coeff_gee_spanish_unique.jpg')
img3 = Image.open(path_results+'coeff_gee_english_unique.jpg')

# Create vertical panel: 3 rows, 1 column
fig, axes = plt.subplots(3, 1, figsize=(4, 12), gridspec_kw={'hspace': -0.4})

# Add images and panel labels
for ax, img, label in zip(axes, [img1, img2, img3], ['A', 'B', 'C']):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='left')

plt.subplots_adjust(hspace=0.01)
plt.savefig(path_results+"Figure1.jpg", dpi=500, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()


# FIGURE 2

# Load images
img1 = Image.open(path_results+'coeff_comp_german_unique_var.jpg')
img2 = Image.open(path_results+'coeff_gee_spanish_unique_var.jpg')
img3 = Image.open(path_results+'coeff_gee_english_unique_var.jpg')

# Create vertical panel: 3 rows, 1 column
fig, axes = plt.subplots(3, 1, figsize=(4, 12), gridspec_kw={'hspace': -0.4})

# Add images and panel labels
for ax, img, label in zip(axes, [img1, img2, img3], ['A', 'B', 'C']):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='left')

plt.subplots_adjust(hspace=0.01)
plt.savefig(path_results+"Figure2.jpg", dpi=500, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()


# FIGURE 3

# Load images
img_A = Image.open(path_results + 'coeff_comp_german_bert_id.jpg')   # A
img_B = Image.open(path_results + 'coeff_comp_german_bert.jpg')      # B
img_C = Image.open(path_results + 'coeff_gee_spanish_bert_id.jpg')   # C
img_D = Image.open(path_results + 'coeff_gee_spanish_bert.jpg')      # D
img_E = Image.open(path_results + 'coeff_gee_english_bert_id.jpg')   # E
img_F = Image.open(path_results + 'coeff_gee_english_bert.jpg')      # F

# Create 3x2 grid (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(8, 12), gridspec_kw={'hspace': -0.4})

# List of images and labels
images = [img_A, img_B, img_C, img_D, img_E, img_F]
labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Plot images
for ax, img, label in zip(axes.flat, images, labels):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='left')

plt.subplots_adjust(wspace=0.01, hspace=0.1)
plt.savefig(path_results + "Figure3.jpg", dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

# ADD SUPPLEMENTARY FIGURES