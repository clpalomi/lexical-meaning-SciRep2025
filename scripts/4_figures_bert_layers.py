"""
@author: u164233
"""

# IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import f_oneway, ttest_ind
import statsmodels.formula.api as smf
import numpy as np


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

# GERMAN
df_ge = pd.read_pickle(path_results+'df_intrinsic_dim_bert_german.pkl')
# ONLY SSD AND HC
df_ge = df_ge[df_ge['group']!='MDD']
# ONLY ID BERT LAYERS
df_ge = df_ge[['name', 'group', 'picture',
                    'id_bert_1', 'id_bert_2', 'id_bert_3', 'id_bert_4',
                    'id_bert_5', 'id_bert_6', 'id_bert_7', 'id_bert_8',
                    'id_bert_9', 'id_bert_10', 'id_bert_11', 'id_bert_12']]


# SPANISH
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
df_spa = pd.read_pickle(path_results+'df_intrinsic_dim_bert_spanish.pkl')
# DEMOGRAPHICS
df_demo_spa = pd.read_pickle(path_santander+'df_demographics_santander.pkl')
df_spa = pd.merge(df_spa, df_demo_spa, on='name', how='inner')
df_spa = df_spa.rename(columns={'bild': 'picture'})
# ONLY ID BERT LAYERS
df_spa = df_spa[['name', 'group', 'picture',
                    'id_bert_1', 'id_bert_2', 'id_bert_3', 'id_bert_4',
                    'id_bert_5', 'id_bert_6', 'id_bert_7', 'id_bert_8',
                    'id_bert_9', 'id_bert_10', 'id_bert_11', 'id_bert_12']]
# change name SZ to SSD
df_spa['group'] = df_spa['group'].replace('SZ', 'SSD')


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
df_eng = pd.read_pickle(path_results+'df_intrinsic_dim_bert_english.pkl')
#df_eng = df_eng.groupby('name').mean(numeric_only=True)
#df_eng = df_eng.merge(df_g, on='name', how='left')
# DEMOGRAPHICS
df_demo_eng = pd.read_pickle(path_london+'df_demographics_london.pkl')
df_eng = pd.merge(df_eng, df_demo_eng, on='name', how='inner')
df_eng = df_eng.rename(columns={'bild': 'picture'})
# ONLY ID BERT LAYERS
df_eng = df_eng[['name', 'group', 'picture',
                    'id_bert_1', 'id_bert_2', 'id_bert_3', 'id_bert_4',
                    'id_bert_5', 'id_bert_6', 'id_bert_7', 'id_bert_8',
                    'id_bert_9', 'id_bert_10', 'id_bert_11', 'id_bert_12']]
# change name FES to FEP
df_eng['group'] = df_eng['group'].replace('FES', 'FEP')
# ONLY FEP AND HC
df_eng = df_eng[(df_eng['group']=='FEP') | (df_eng['group']=='HC')]


""" GRAPHS AND REGRESSIONS FUNCTIONS """


def get_pval_mixed_effect(df, layer_col, group_col='group', subject_col='name', baseline='HC'):
    # Ensure 'group' is categorical with baseline
    df[group_col] = pd.Categorical(df[group_col], categories=[baseline] + [g for g in df[group_col].unique() if g != baseline], ordered=True)
    try:
        model = smf.mixedlm(f"{layer_col} ~ {group_col}", df, groups=df[subject_col])
        result = model.fit()
        # Return p-value of the clinical group (not intercept)
        pval = result.pvalues.get(f'{group_col}[T.{[g for g in df[group_col].unique() if g != baseline][0]}]', np.nan)
        return pval
    except Exception as e:
        print(f"Model failed for {layer_col}: {e}")
        return np.nan
    

def collapse_and_plot_by_picture(df, picture):
    # Identify the id_bert_n columns
    id_bert_cols = [col for col in df.columns if col.startswith("id_bert_")]
    # Collapse the dataframe by taking the mean for each group-picture pair
    df_avg = df.groupby(["group", "picture"], as_index=False)[id_bert_cols].mean()
    # Filter for the selected picture
    df_f = df_avg[df_avg["picture"] == picture]
    # Plot the results
    plt.figure(figsize=(8, 5))
    # Get the x-axis values (1 to 12) from column names
    x_values = [int(col.split("_")[-1]) for col in id_bert_cols]
    # Plot each group-picture pair
    for _, row in df_f.iterrows():
        plt.plot(x_values, row[id_bert_cols], marker='o', label=f'Group {row["group"]}, Pic {row["picture"]}')
    plt.xlabel("BERT layer")
    plt.ylabel("Average ID")
    plt.title("Averaged ID values by group and picture")
    plt.xticks(x_values)
    plt.legend()
    #plt.savefig(path_results+'bert_layers_red_id_picture.jpg', dpi=500)
    plt.show()


def collapse_and_plot_av(df, name, alpha=0.05):
    # ----- Prep data -----
    # Identify the id_bert_n columns
    id_bert_cols = [col for col in df.columns if col.startswith("id_bert_")]
    # Collapse the dataframe by taking the mean for each group-picture pair
    df_avg = df.groupby(["group"], as_index=False)[id_bert_cols].mean()
    # Get the x-axis values (1 to 12) from column names
    x_values = [int(col.split("_")[-1]) for col in id_bert_cols]
    

    # Perform ANOVA (or t-test) for each BERT layer
    significant_layers = []
    groups = df['group'].unique()
    for i, col in enumerate(id_bert_cols):
        values_by_group = [df[df['group'] == g][col].dropna() for g in groups]
        if len(groups) == 2:
            stat, p = ttest_ind(*values_by_group)
        else:
            stat, p = f_oneway(*values_by_group)
        if p < alpha:
            significant_layers.append((i+1, p))  # i+1 for layer index


    # Plot the results
    plt.figure(figsize=(8, 5))
    # Ensure HC is plotted first (so it always gets the first color in the cycle),
    # then plot the rest in alphabetical order to keep legends stable.
    ordered_groups = []
    if 'HC' in df_avg['group'].values:
        ordered_groups.append('HC')
    ordered_groups += [g for g in sorted(df_avg['group'].unique()) if g != 'HC']

    for g in ordered_groups:
        row = df_avg[df_avg['group'] == g].iloc[0]
        # No explicit color -> uses Matplotlib's cycle; HC takes the first color consistently
        plt.plot(x_values, row[id_bert_cols], marker='o', label=g, linewidth=2)

    # Add significance markers
    for x, p in significant_layers:
        y_max = df_avg[[f"id_bert_{x}"]].max().max()
        plt.text(x, y_max + 0.05, '*', ha='center', fontsize=14, color='red')
    """
    # Mark significance via mixed models
    for i, col in enumerate(id_bert_cols):
        p = get_pval_mixed_effect(df[['name', 'group', col]].dropna(), layer_col=col)
        if p is not None and p < alpha:
            y_max = df_avg[col].max()
            plt.text(i + 1, y_max + 0.05, '*', ha='center', fontsize=14, color='red')
    """
    plt.xlabel("BERT layer")
    plt.ylabel("Average ID")
    plt.title(f"Average ID by BERT layer and group - {name} sample")
    plt.xticks(x_values)
    plt.legend()
    plt.savefig(path_results+f'bert_layers_{name}_avg.jpg', dpi=500)
    plt.show()



""" FIGURES """

# GERMAN
df_ge_avg   = collapse_and_plot_av(df_ge, 'German')
df_ge_avg_1 = collapse_and_plot_by_picture(df_ge, '1')
df_ge_avg_2 = collapse_and_plot_by_picture(df_ge, '2')
df_ge_avg_4 = collapse_and_plot_by_picture(df_ge, '4')
df_ge_avg_6 = collapse_and_plot_by_picture(df_ge, '6')

# SPANISH
df_spa_avg   = collapse_and_plot_av(df_spa, 'Spanish')
df_spa_avg_1 = collapse_and_plot_by_picture(df_spa, '1')
df_spa_avg_2 = collapse_and_plot_by_picture(df_spa, '2')
df_spa_avg_3 = collapse_and_plot_by_picture(df_spa, '3')

# ENGLISH
df_eng_avg   = collapse_and_plot_av(df_eng, 'English')
df_eng_avg_1 = collapse_and_plot_by_picture(df_eng, '1')
df_eng_avg_2 = collapse_and_plot_by_picture(df_eng, '2')
df_eng_avg_3 = collapse_and_plot_by_picture(df_eng, '3')




""" COMBINE FIGURES IN PANEL """

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'

# FIGURE 4

# Load images
img1 = Image.open(path_results+'bert_layers_German_avg.jpg')
img2 = Image.open(path_results+'bert_layers_Spanish_avg.jpg')
img3 = Image.open(path_results+'bert_layers_English_avg.jpg')

# Create vertical panel: 3 rows, 1 column
fig, axes = plt.subplots(3, 1, figsize=(4, 12), gridspec_kw={'hspace': -0.6})

# Add images and panel labels
for ax, img, label in zip(axes, [img1, img2, img3], ['A', 'B', 'C']):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='left')

plt.subplots_adjust(hspace=0.01)
plt.savefig(path_results+"Figure4.jpg", dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()


