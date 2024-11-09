"""
Module to analysis user behavior dataset
"""

#!/usr/bin/env python
# coding: utf-8

# Análise de Dados

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("./user_behavior_dataset.csv")

# Análise primaria do dataset
# display(df)
print("-----------------------------\nData types:\n", df.dtypes)
print("-----------------------------\nDescribe:\n", df.describe())
print("-----------------------------\nVerificando se há valores NaN:\n", df.isna().sum())
print("-----------------------------\nVerificando se há valores nulos:\n",
      df.isnull().sum())

std_deviation = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                    'Battery Drain (mAh/day)', 'Number of Apps Installed',
                    'Data Usage (MB/day)', 'Age']].std()

print("-----------------------------\nDesvio Padrao:\n", std_deviation)
print("-----------------------------\nIdades Unicas:\n", df['Age'].unique())


# Análise Exploratória
#

# In[65]:

print("-----------------------------\nIntervalo de idades:\n",
      df["Age"].min(), "a", df["Age"].max())
print("-----------------------------\nMédia das idades:\n",
      round(df["Age"].mean(), 2))
print("-----------------------------\nMediana das idades:\n",
      round(df["Age"].median(), 2))
print("-----------------------------\nModa das idades:\n",
      round(df["Age"].mode(), 2))
print("-----------------------------\nMédia dos apps instalados:\n",
      round(df["Number of Apps Installed"].mean(), 2))
print("-----------------------------\nMédia de Tempo de Tela Ligada (horas/dia):\n",
      round(df["Screen On Time (hours/day)"].mean(), 2))
print("-----------------------------\nMédia do uso de dados (MB/Dia):\n",
      round(df["Data Usage (MB/day)"].mean(), 2))


# Análise Visual
#

# In[66]:

sns.set(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle("Análise Exploratória do User Behavior Dataset", fontsize=16)
sns.histplot(df['App Usage Time (min/day)'],
             kde=True, color="skyblue", ax=axes[0, 0])
axes[0, 0].set_title("Distribuição do tempo de uso do aplicativo (min/dia)")
sns.histplot(df['Screen On Time (hours/day)'],
             kde=True, color="salmon", ax=axes[0, 1])
axes[0, 1].set_title("Distribuição do tempo de tela ligado (horas/dia)")
sns.histplot(df['Battery Drain (mAh/day)'], kde=True,
             color="lightgreen", ax=axes[1, 0])
axes[1, 0].set_title("Distribuição do consumo da bateria (mAh/dia)")
sns.histplot(df['Number of Apps Installed'],
             kde=True, color="coral", ax=axes[1, 1])
axes[1, 1].set_title("Distribuição do número de aplicativos instalados")
sns.histplot(df['Age'], kde=True, color="purple", ax=axes[2, 0])
axes[2, 0].set_title("Distribuição de idade")
sns.countplot(x='User Behavior Class', hue='User Behavior Class',
              data=df, palette="viridis", ax=axes[2, 1], legend=False)
axes[2, 1].set_title("User Behavior Class Distribution")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='App Usage Time (min/day)',
                y='Screen On Time (hours/day)', hue='User Behavior Class', palette='coolwarm')
plt.title('Tempo de uso do aplicativo x tempo de tela por classe de comportamento')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', hue='Gender', palette='Set2', legend=False)
plt.title('Distribuição de Gênero')
plt.show()


# Preparando dados para Treinamento
#

# In[67]:


# Preparando o dataset para treinamento
df_prepared = df.drop(columns='User ID')

le = LabelEncoder()
df_prepared['Gender'] = le.fit_transform(df_prepared['Gender'])
df_prepared['Operating System'] = le.fit_transform(
    df_prepared['Operating System'])
df_prepared['Device Model'] = le.fit_transform(df_prepared['Device Model'])

df_prepared.head()


# KNeighbors

# In[68]:

X = df_prepared.drop(columns=['User Behavior Class'])
y = df_prepared['User Behavior Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Predizer no conjunto de teste
y_pred_knn = knn.predict(X_test)

# Relatório de classificação
print("K-Nearest Neighbors")
print(f"Acertou {((y_test == y_pred_knn).sum())} de {len(X_test)}")
print(classification_report(y_test, y_pred_knn))


# Random Forest Tree

# In[69]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, rf_pred),
            annot=True, cmap='Blues', fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.show()


# In[ ]:


# In[70]:


print(classification_report(y_test, y_pred_knn))
print(classification_report(y_test, rf_pred))
