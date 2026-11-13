# -*- coding: utf-8 -*-
"""
DEMOGRAPHICS
@author: u164233
"""

# PATH INPUT
path_input     = '/PATH/TO/PRIVATE_DATA/german/texts/'
path_demo      = '/PATH/TO/PRIVATE_DATA/german/'
path_results   = '/PATH/TO/RESULTS/'
path_santander = '/PATH/TO/PRIVATE_DATA/spanish/'
path_london    = '/PATH/TO/PRIVATE_DATA/english/'


import pyreadstat
import numpy as np
import pandas as pd


# SPANISH
df, meta = pyreadstat.read_sav(path_santander+"Data_Santander_Discouse.sav")
df.rename(columns={'codigoFamiliar': 'name', 'EdadEvaluacion': 'age', 'years_edu': 'education', 'sexo': 'female'}, inplace=True)
df['age'] = np.floor(df['age']).astype(int)
df['female'] = df['female'].apply(lambda x: 1 if x == 2 else 0)
df['name'] = df['name'].apply(lambda x: 'DS_' + str(x))
df = df[['name', 'age', 'education', 'female']]
df.to_pickle(path_santander+'df_demographics_santander.pkl')






# ENGLISH
df_demo_eng = pd.read_excel(path_london+'demo.xlsx')
df_demo_eng.rename(columns={'ID': 'name', 'AgeScan1': 'age', 'Gender': 'female'}, inplace=True)
df_demo_eng['female'] = df_demo_eng['female'].apply(lambda x: 1 if x == 2 else 0)
df_demo_eng = df_demo_eng[['name', 'age', 'female']]
df_demo_eng['age'] = df_demo_eng['age'].astype(int)
df_demo_eng.to_pickle(path_london+'df_demographics_london.pkl')



