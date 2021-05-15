# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:17:27 2021

@author: gusta
"""

import pandas as pd

base = pd.read_csv('census.csv')

previsores = base.iloc[: , 0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

conlumn_transform = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = conlumn_transform.fit_transform(previsores).toarray()

labelEncoder_classe = LabelEncoder()
classe = labelEncoder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
prev_train, prev_test, class_train, class_test = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(prev_train, class_train)
previsao = classificador.predict(prev_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(class_test, previsao)
matrix =  confusion_matrix(class_test, previsao)