# -*- coding: utf-8 -*-
"""
Created on Mon May  3 21:14:36 2021

@author: gusta
"""
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')

base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[: , 1:4].values
classe = base.iloc[: , 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[: , 0:3])
previsores[: , 0:3] = imputer.transform(previsores[:, 0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
prev_train, prev_test, classe_train, class_test = train_test_split(previsores, classe, test_size = 0.2 , random_state=42)


from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(prev_train, classe_train)
previsao = classificador.predict(prev_test)

from sklearn.metrics  import confusion_matrix, accuracy_score
precisao = accuracy_score(class_test, previsao)
matriz = confusion_matrix(class_test, previsao)