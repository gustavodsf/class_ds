# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:03:17 2021

@author: gusta
"""

import pandas as  pd
base =  pd.read_csv('risco_credito.csv');

previsores  = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder();

base.columns

previsores[:, 0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelEncoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelEncoder.fit_transform(previsores[:, 3])


from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion="entropy")
classificador.fit(previsores, classe)
print(classificador.feature_importances_)


export.export_graphviz(classificador,
                       out_file = 'arvore_dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = classificador.classes_,
                       filled=True,
                       leaves_parallel=True
                       )


resultado  = classificador.predict([[0,0,1,2],[3,0,0,0]])
print(classificador.classes_)