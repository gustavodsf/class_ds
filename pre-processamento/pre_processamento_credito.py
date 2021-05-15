# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:35:53 2021

@author: gusta
"""
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.describe()

# base.loc[base['age'] < 0]
# base.drop('age', 1, inplace=True)
# base.drop(base[base.age < 0 ].index, inplace=True)

# base.mean()
# base.age.mean()
# base['age'][base.age > 0].mean()

base.loc[base.age < 0, 'age'] =  np.nan
# base.loc[base.age.isna(), 'age'] = base.age.mean()

# pd.isnull(base['age'])
# base[pd.isnull(base['age'])] | base[base.age.isna()] | base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

