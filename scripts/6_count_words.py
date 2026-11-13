# -*- coding: utf-8 -*-
"""
@author: u164233
"""

# PATH INPUT
path_input     = '/PATH/TO/PRIVATE_DATA/german/texts/'
path_demo      = '/PATH/TO/PRIVATE_DATA/german/'
path_results   = '/PATH/TO/RESULTS/'
path_santander = '/PATH/TO/PRIVATE_DATA/spanish/'
path_london    = '/PATH/TO/PRIVATE_DATA/english/'

# PICTURES
# German:  pict 1: boy, pict 2: countryside, pict 4: couple, pict 6: old woman
# Spanish: pict 1: countryside, pict2: couple, pict3: bridge

# IMPORT PACKAGES
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import itertools



""" GERMAN """
df_ge = pd.read_pickle(path_results+'df_words_dim_reduction_german.pkl')
# ONLY SSD AND HC
df_ge = df_ge[df_ge['group']!='MDD']

# SUMMARY DIFFERENCES
summary = df_ge.groupby(['group', 'bild']).agg(
    avg_words=('n_content', 'mean'),
    std_words=('n_content', 'std'),
).reset_index()
t_test_words_stats = []
t_test_words_pvals = []
bild_categories = df_ge['bild'].unique()
# Loop through each 'bild' and perform t-tests between SSD and HC for each variable
for bild in bild_categories:
    # Filter data by 'bild'
    df_bild = df_ge[df_ge['bild'] == bild]
    # Separate into SSD and HC groups within each 'bild'
    group_A = df_bild[df_bild['group'] == 'SSD']
    group_B = df_bild[df_bild['group'] == 'HC']  
    # Perform t-test for n_conte (words) only if both groups have enough data
    if len(group_A) > 1 and len(group_B) > 1:
        t_test_words = stats.ttest_ind(group_A['n_content'], group_B['n_content'], nan_policy='omit')
    else:
        # If insufficient data for comparison, store None values
        t_test_words = None  
    t_test_words_stats.append(t_test_words.statistic if t_test_words else None)
    t_test_words_pvals.append(t_test_words.pvalue if t_test_words else None)
summary['t_test_words_statistic'] = t_test_words_stats*2
summary['t_test_words_p_value'] = t_test_words_pvals*2



# GRAPH DIFFERENCES IN WORDS
plt.rcParams['font.family'] = 'Times New Roman'  
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 6))
sns.boxplot(x='bild', y='n_content', hue='group', data=df_ge, palette='Set2')
plt.title('Word count by picture in German sample', fontsize=14)
plt.xlabel('Picture', fontsize=12)
plt.ylabel('Word count', fontsize=12)
plt.legend(title='Group', loc='upper right')
plt.tight_layout()
plt.savefig(path_results+'word_differences_german.jpg', dpi=500)
plt.show()



""" SPANISH """

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

# SUMMARY DIFFERENCES
summary = df_spa.groupby(['group']).agg(
    avg_words=('n_content', 'mean'),
    std_words=('n_content', 'std'),
).reset_index()
t_test_words_stats = []
t_test_words_pvals = []
group_A = df_spa[df_spa['group'] == 'SZ']
group_B = df_spa[df_spa['group'] == 'HC']  
if len(group_A) > 1 and len(group_B) > 1:
    t_test_words = stats.ttest_ind(group_A['n_content'], group_B['n_content'], nan_policy='omit')
    t_test_words_stats.append(t_test_words.statistic)
    t_test_words_pvals.append(t_test_words.pvalue)
else:
    t_test_words_stats.append(None)
    t_test_words_pvals.append(None)
summary['t_test_words_statistic'] = t_test_words_stats*2
summary['t_test_words_p_value'] = t_test_words_pvals*2



# GRAPH DIFFERENCES IN WORDS
plt.rcParams['font.family'] = 'Times New Roman'  
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 6))
sns.boxplot(y='n_content', hue='group', data=df_ge, palette='Set2')
plt.title('Average word count in Spanish sample', fontsize=14)
plt.xlabel('Picture', fontsize=12)
plt.ylabel('Word count', fontsize=12)
plt.legend(title='Group', loc='upper right')
plt.tight_layout()
plt.savefig(path_results+'word_differences_spanish.jpg', dpi=500)
plt.show()



""" ENGLISH """

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



# DEMOGRAPHICS STATS
summary = df_eng.groupby('group').agg(
    avg_words=('n_content', 'mean'),
).reset_index()
# Performing t-tests for age and education
group_A = df_eng[df_eng['group'] == 'HC']
group_B = df_eng[df_eng['group'] == 'CS']
group_C = df_eng[df_eng['group'] == 'CHR']
group_D = df_eng[df_eng['group'] == 'FES']
t_test_age_1 = stats.ttest_ind(group_A['n_content'], group_B['n_content'])
t_test_age_2 = stats.ttest_ind(group_A['n_content'], group_C['n_content'])
t_test_age_3 = stats.ttest_ind(group_A['n_content'], group_D['n_content'])
t_test_age_4 = stats.ttest_ind(group_B['n_content'], group_C['n_content'])
t_test_age_5 = stats.ttest_ind(group_B['n_content'], group_D['n_content'])
t_test_age_6 = stats.ttest_ind(group_C['n_content'], group_D['n_content'])
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
print(summary)
p_values_summary = summary[['group', 't_test_age_p_value_1',
                            't_test_age_p_value_2',
                            't_test_age_p_value_3',
                            't_test_age_p_value_4',
                            't_test_age_p_value_5',
                            't_test_age_p_value_6']]
p_values_summary.columns = ['group', 
                             'HC - CS (p-value)', 
                             'HC - CHR (p-value)', 
                             'HC - FEP (p-value)', 
                             'CS - CHR (p-value)', 
                             'CS - FEP (p-value)', 
                             'CHR - FEP (p-value)']

# P-VALUES
# Pairwise comparisons (example using t-tests)
groups = df_eng['group'].unique()
comparisons = list(itertools.combinations(groups, 2))
for group1, group2 in comparisons:
    data1 = df_eng[df_eng['group'] == group1]['n_content']
    data2 = df_eng[df_eng['group'] == group2]['n_content']
    t_stat, p_value = stats.ttest_ind(data1, data2)
    print(f'Comparison between {group1} and {group2}: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}')


# GRAPH DIFFERENCES IN WORDS
df_eng['group'] = df_eng['group'].replace({'FES': 'FEP'})
plt.rcParams['font.family'] = 'Times New Roman'  
plt.rcParams['font.size'] = 14
plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='n_content', data=df_eng, palette='Set2', dodge=False) 
#sns.boxplot(x='group', y='n_content', data=df_eng, palette='Set2')
plt.title('Average word count by group in English sample', fontsize=14)
plt.xlabel('Group', fontsize=12)
plt.ylabel('Word Count', fontsize=12)
plt.tight_layout()
plt.savefig(path_results+'word_differences_english.jpg', dpi=500)
plt.show()
