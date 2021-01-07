##### Import packages
# Basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

# To avoid warnings´
import warnings
warnings.filterwarnings("ignore")





##### Import data
# Check the path before running it

df = pd.read_pickle('data_modelling_memred.pkl')





##### I create a numerical series -- I need to predict 17th month

dict_meses = dict()
for num,value in enumerate(df.fecha_dato.unique(),1):
    dict_meses[value] = num
    
df["month_counter"] = df['fecha_dato'].map(dict_meses)





##### Due to imputation strategy, I reapply previous filter to prevent unexpected values 

info_17 = df.ncodpers.value_counts()[df.ncodpers.value_counts() == 17].index
df = df[df['ncodpers'].isin(info_17)].reset_index(drop = True)


# # Prediction Model for `ind_cco_fin_ult1`




##### I extract the other 15 rows from the first product purchase column and transpose it

df_product = pd.DataFrame()
for i in range(0,17):
    x = df.query(f'month_counter == {i}')["ind_cco_fin_ult1"].tolist()
    df1 = pd.DataFrame(x).T
    df_product = pd.concat([df1,df_product], axis=0)
    
df_product = df_product.T
df_product.columns = range(1,17)
df_product





##### I keep only relevant columns (product column will have a special processing)

df_pred = df[['month', 'sexo', 'age', 'antiguedad','tiprel_1mes', 'canal_entrada', 'nomprov', 'renta', 'segmento']].copy()

df_pred.head()





##### See which ones need data preparation based on unique values

for col in df_pred.columns:
    print(f'{col}: {df_pred[col].unique()}')





##### Data binning to age, renta and antiguedad

fig, ax = plt.subplots(1,3, figsize = [25, 5])
fig.suptitle('Data binning', fontsize = 18)

# Age
ax[0].hist(df_pred.age)
ax[0].set_title('Age distribution', fontsize = 12)

bins = np.linspace(df_pred.age.min(), df_pred.age.max(), 5)
# Young = 0 | Adult = 1 | Senior = 2 | Elderly = 3
df_pred['age_binning'] = pd.cut(df_pred['age'], bins, labels = [0,1,2,3], include_lowest=True)
del df_pred['age']

# Renta
ax[1].hist(df_pred.renta)
ax[1].set_title('Income distribution', fontsize = 12)

bins = np.linspace(df_pred.renta.min(), df_pred.renta.max(), 5)
# Low = 0 | Middle = 1 | Middle-High = 2 | High = 3
df_pred['renta_binning'] = pd.cut(df_pred['renta'], bins, labels = [0,1,2,3], include_lowest=True)
del df_pred['renta']

# Antiguedad
ax[2].hist(df_pred.antiguedad)
ax[2].set_title('Seniority distribution', fontsize = 12)

bins = np.linspace(df_pred.antiguedad.min(), df_pred.antiguedad.max(), 5)
# Baja = 0 | Media = 1 | Media Alta = 2 | Alta = 3
df_pred['antiguedad_binning'] = pd.cut(df_pred['antiguedad'], bins, labels = [0,1,2,3], include_lowest=True)
del df_pred['antiguedad']





##### To dummy to sexo, tiprel_1mes and canal_entrada

for col in ["sexo", "tiprel_1mes","canal_entrada"]:
    df_pred = df_pred.join(pd.get_dummies(df_pred[col], prefix = col))
    
for col in ["sexo", "tiprel_1mes","canal_entrada", "sexo_V", "tiprel_1mes_I"]:
    del df_pred[col]





##### Label Encoding to nomprov and segmento

for col in ['nomprov', 'segmento']:
    LabelEncoding = LabelEncoder()
    df_pred[col] = LabelEncoding.fit_transform(df_pred[col].values)





##### Now I join both DataFrames

df_product = df_product.join(df_pred)
df_product





##### Creation of X and y

X = np.asarray(df_product.values)
y = np.asarray(df.query(f'month_counter == 17')["ind_cco_fin_ult1"]) # target is last month





##### Creation of X and y train/test

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)





##### Undersampling to create sintetic data to improve class balance.
# Increase minority class size until its size represent 80% of major class size

undersampling = RandomUnderSampler(sampling_strategy=0.8) 
X_balance, y_balance = undersampling.fit_resample(X, y)
Counter(y_balance)





##### Creation of X and y split -- train and test applying undersampling

X_train_balance, X_test_balance, y_train_balance, y_test_balance = train_test_split(X_balance, y_balance, test_size=0.4)


# ## Logistic Regression




##### Logistic Regression 

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)

print('Accuracy score: ',accuracy_score(y_test,yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict product purchase next month')
plt.show()


# ### With Undersampling




##### Logistic Regression with Undersampling

clf = LogisticRegression(solver='liblinear')
clf.fit(X_train_balance, y_train_balance)
yhat = clf.predict(X_test_balance)
print('Accuracy score: ',accuracy_score(y_test_balance, yhat))





##### Classification Report

print(classification_report(y_test_balance, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test_balance, y_test_balance,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict product purchase next month')
plt.show()


# ## Random Forest Classifier




##### Random Forest Classifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print('Accuracy score: ',accuracy_score(y_test,yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict product purchase next month')
plt.show()





##### Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(df_product.columns)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ### With Undersampling




##### Random Forest Classifier with Undersampling

clf = RandomForestClassifier()
clf.fit(X_train_balance, y_train_balance)
yhat = clf.predict(X_test_balance)
print('Accuracy score: ',accuracy_score(y_test_balance,yhat))





##### Classification Report

print(classification_report(y_test_balance, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test_balance, y_test_balance,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict product purchase next month')
plt.show()





##### Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(df_product.columns)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ## XGB Classifier




##### XGB Classifier

clf = XGBClassifier() 
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print('Accuracy score: ',accuracy_score(y_test, yhat))





##### Classification Report

print(classification_report(y_test, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test, y_test,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict product purchase next month')
plt.show()





##### Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(df_product.columns)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()


# ### With Undersampling




##### XGB Classifier

clf = XGBClassifier() 
clf.fit(X_train_balance, y_train_balance)
yhat = clf.predict(X_test_balance)
print('Accuracy score: ',accuracy_score(y_test_balance, yhat))





##### Classification Report

print(classification_report(y_test_balance, yhat, digits=3))





##### Confussion Matrix

plot_confusion_matrix(clf, X_test_balance, y_test_balance,
                             display_labels=[0,1],
                             cmap=plt.cm.Blues,
                             normalize="true")
plt.title('Confussion matrix on how the \n model can predict product purchase next month')
plt.show()





##### Feature Importance

features_importance = clf.feature_importances_
features_array = np.array(df_product.columns)
features_array_ordered = features_array[(features_importance).argsort()[::-1]]
features_array_ordered

plt.figure(figsize=(16,10))
sns.barplot(y = features_array, x = features_importance, orient='h', order=features_array_ordered[:50])

plt.show()

