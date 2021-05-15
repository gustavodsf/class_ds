import pandas as pd

df_census = pd.read_csv('census.csv')

previsores = df_census.iloc[:, 0:14].values
classe = df_census.iloc[: , 14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_previsores = LabelEncoder()
classe_encoded = labelencoder_previsores.fit(classe)
classe = classe_encoded.transform(classe)

from sklearn.compose import ColumnTransformer

oneHotEncoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
previsores = oneHotEncoder.fit_transform(previsores).toarray()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

