# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:26:06 2021

@author: gusta
"""

import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[: , 0:14].values
classe = base.iloc[: , 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_previsores = LabelEncoder();
previsores[: , 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[: , 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[: , 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[: , 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[: , 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[: , 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[: , 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[: , 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

oneHotEncoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncoder.fit_transform(previsores).toarray()

labeencoder_classe = LabelEncoder()
classe = labeencoder_classe.fit_transform(classe)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
pred_train, pred_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.15, random_state = 0);

from sklearn.tree import DecisionTreeClassifier
classificador = DecisionTreeClassifier( criterion='entropy' , random_state = 0)
classificador.fit(pred_train, classe_train)
previsoes = classificador.predict(pred_test)

from sklearn.metrics import confusion_matrix, accuracy_score;

precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

