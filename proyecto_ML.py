#!/usr/bin/env python
# coding: utf-8

# In[296]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from yellowbrick.regressor import PredictionError
from sklearn.ensemble import RandomForestRegressor


# In[297]:


dataset = pd.read_csv("2023_Precios_Casas_RM.csv")
dataset=dataset.drop(['id','Realtor','Ubicacion','Price_USD','Price_CLP'], axis=1)

dataset.info()


# In[298]:


dataset.isna().sum()


# In[299]:


dataset = dataset.dropna().reset_index(drop=True)


# In[300]:


#Como medida de seguridad, eliminalos las filas duplicadas.
dataset = dataset.drop_duplicates().reset_index(drop=True)


# In[301]:


dataset.isna().sum()


# Ordenar por Precio_UF

# In[302]:


dataset = dataset.rename(columns = {'Built Area' : 'built_Area', 'Total Area' : 'total_Area'})
dataset = dataset.sort_values(by="Price_UF").reset_index(drop=True)


# Eliminamos precios con valores extremadamente bajos

# In[303]:


dataset['Price_UF'].describe()


# In[304]:


dataset[dataset['Price_UF']>90000]


# In[305]:


dataset = dataset.drop(dataset[dataset['Price_UF']>90000].index).reset_index(drop=True)
dataset


# In[306]:


dataset[dataset['Price_UF'] < 1000]


# In[307]:


dataset= dataset.loc[dataset['Price_UF'] > 1000 ]
print(dataset.shape)
dataset=dataset.reset_index(drop=True)
dataset


# In[308]:


dataset['Price_UF'].describe()


# Analisis de la caracteristica 'Bath' (baños)

# In[309]:


dataset['Baths'].value_counts()


# In[310]:


dataset['Baths'].describe()


# In[311]:


dataset[dataset['Baths'] > 7]


# In[312]:


dataset = dataset.drop(dataset[dataset['Baths'] > 7].index).reset_index(drop=True)
dataset['Baths'].describe()


# Analisis de la caracteristica 'Dorms'

# In[313]:


dataset['Dorms'].value_counts()


# In[314]:


dataset['Dorms'].describe()


# In[315]:


dataset[dataset['Dorms'] > 8]


# In[316]:


dataset = dataset.drop(dataset[dataset['Dorms'] >= 12].index).reset_index(drop=True)
dataset['Dorms'].describe()


# Analisis de la caracteristica 'total_Area'

# In[317]:


dataset['total_Area'].describe()


# In[318]:


dataset = dataset.sort_values(by="total_Area").reset_index(drop=True)
dataset[dataset['total_Area'] > 3000]


# In[319]:


dataset = dataset.drop(dataset[dataset['total_Area'] > 3000].index).reset_index(drop=True)
dataset['total_Area'].describe()


# Eliminamos las filas que tengas Built_Area >= Total_Area

# In[320]:


dataset[dataset['built_Area'] >= dataset['total_Area']].count()


# In[321]:


dataset = dataset.drop(dataset[dataset['built_Area'] >= dataset['total_Area']].index).reset_index(drop=True)
dataset['total_Area'].describe()


# Analisis de la caracteristica Built Area

# In[322]:


dataset['built_Area'].describe()


# In[323]:


dataset[dataset['built_Area'] < 30]


# In[324]:


dataset = dataset.drop(dataset[dataset['built_Area'] < 30].index).reset_index(drop=True)
dataset['built_Area'].describe()


# In[325]:


dataset[dataset['built_Area'] > 500]


# In[326]:


dataset = dataset.drop(dataset[dataset['built_Area'] > 500].index).reset_index(drop=True)
dataset['built_Area'].describe()


# Analisis de la caracteristica 'Parking'

# In[327]:


print(dataset['Parking'].value_counts())


# In[328]:


print(dataset['Parking'].describe())


# In[329]:


dataset = dataset.drop(dataset[dataset['Parking'] >= 10].index).reset_index(drop=True)
dataset['Parking'].describe()


# In[330]:


dataset.info()


# Correlacion Entre Variables

# In[331]:


correlacion=dataset.drop(['Comuna'], axis=1).corr()
correlacion.info()
correlacion


# In[332]:


dataset.info()


# In[333]:


colores = sns.light_palette('salmon', as_cmap=True)
mask = np.triu(correlacion)

with sns.axes_style("white"):
    plt.figure(figsize=(11, 6))
    sns.heatmap(correlacion, cmap=colores, mask=mask, square=True, annot=True, fmt='.2f')
##plt.show()


# Aplicamos One-Hot para las variables vategoricas

# In[334]:


dataset = pd.get_dummies(dataset, columns=['Comuna'], prefix='Comuna')
dataset.head()


# In[335]:


data = dataset.drop('Price_UF', axis=1)
labels = dataset['Price_UF']
data


# Separacion de los datos

# In[336]:


x_train,x_test,y_train,y_test=train_test_split(data, labels, test_size=0.2)


# Entrenamiento con Regrasion Linear

# In[337]:


model_lineal = LinearRegression()
model_lineal.fit(x_train, y_train)


# % Resultado
# 

# In[338]:


model_lineal.score(x_test,y_test)


# In[339]:


y_pred = model_lineal.predict(x_test)


# In[340]:


fig, ax = plt.subplots(figsize=(8, 8))
pev = PredictionError(model_lineal)
pev.fit(x_train, y_train)
pev.score(x_test, y_test)
pev.poof()


# Arbol de decision para Regresion

# In[341]:


model_tree = DecisionTreeRegressor()
model_tree.fit(x_train, y_train)


# In[342]:


model_tree.score(x_test,y_test)


# In[343]:


predicciones = model_tree.predict(x_test)
predicciones


# In[344]:


fig, ax = plt.subplots(figsize=(8, 8))
pev = PredictionError(model_tree)
pev.fit(x_train, y_train)
pev.score(x_test, y_test)
pev.poof()


# Random Forest Regresion

# In[345]:


rf_model = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=10)


# In[346]:


rf_model.fit(x_train, y_train)


# In[347]:


rf_model.score(x_test,y_test)


# In[348]:


prediccion_rf = rf_model.predict(x_test)


# In[349]:


fig, ax = plt.subplots(figsize=(8, 8))
pev = PredictionError(rf_model)
pev.fit(x_train, y_train)
pev.score(x_test, y_test)
pev.poof()

