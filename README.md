# Ex-07-Feature-Selection
## AIM :
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation :
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM :
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


## CODE :
#### Importing Packages.
```python
import numpy as np
import pandas as pd
data=pd.read_csv('/content/titanic_dataset.csv')
y=data['Survived']
X=data.copy()
del X['Survived']
```
#### Handling Missing values.
```python
data.isnull().mean().sort_values(ascending=True)
X['body_null']=np.where(X.body.isnull(),1,0)
X.body.fillna(0,inplace=True)
temp_cabin=X.cabin.str.split(expand=True)[0]
X['cabin_char']=temp_cabin.str[0]
del X['cabin']

X['cabin_char']=np.where(X.cabin_char.isnull(),'M',X.cabin_char)

temp_boat=X.boat.str.rsplit(expand=True).rename(columns={0:'boat_1',1:'boat_2',2:'boat_3'})
X['boat_char']=np.where(temp_boat.boat_1.str.isdigit(),np.nan,temp_boat.boat_1)
X['boat_char']=np.where(X.boat_char.isnull(),'M',X.boat_char)
del X['boat']
```
#### Label Encoding.
```python
X['name']=X.name.str.split(',',expand=True)[1].str.split('.',expand=True)[0]
X['name']=X['name'].map({' Mrs': 0,  ' Mr': 1, ' Miss': 2, ' Master': 3, ' Col': 1, ' Mme': 0, ' Dr': 4, ' Major': 1, ' Capt': 1, ' Lady': 0, ' Sir': 1, ' Mlle': 0, ' Dona': 0, ' Jonkheer': 1, ' the Countess': 0, ' Don': 1, ' Rev': 1, ' Ms': 0})
X['sex']=X['sex'].map({'male':0,'female':1})
X['embarked']=X.embarked.map({'M':0,'S':1,'C':2,'Q':3})
X['cabin_char']=X.cabin_char.map({k:i for i,k in enumerate(X.cabin_char.unique())})
X['boat_char']=X.boat_char.map({k:i for i,k in enumerate(X.boat_char.unique())})
```
#### One Hot Encoding.
```python
temp_name=pd.get_dummies(X.name,drop_first=True)
X=pd.concat([X,temp_name],axis=1)
del X['name']
temp_embarked=pd.get_dummies(X.embarked,drop_first=True).rename(columns={1:5,2:6,3:7})
X=pd.concat([X,temp_embarked],axis=1)
del X['embarked']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=50)
```
#### Feature Selection.
```python
# Basic Methods
X_train_filter=X_train.copy()
y_train_filter=y_train.copy()
X_test_filter=X_test.copy()
y_test_filter=y_test.copy()

from sklearn.feature_selection import VarianceThreshold
sel=VarianceThreshold(threshold=0.01)
sel.fit(X_train_filter)
sel.transform(X_train)
X_train_filter.columns[sel.get_support()]
del X_train_filter[4]
del X_test_filter[4]

from xgboost import XGBClassifier
model_filter=XGBClassifier()
model_filter.fit(X_train_filter,y_train_filter)


y_pred_filter=model_filter.predict(X_test_filter)

from sklearn.metrics import confusion_matrix
metric_filter=confusion_matrix(y_test_filter,y_pred_filter)

accuracy_filter_basic=(metric_filter[0][0]+metric_filter[1][1])/sum(sum(metric_filter))*100
print('Accuracy using filter : ',accuracy_filter_basic)
```
```python
#Correlation Methods
X_train_filter=X_train.copy()
y_train_filter=y_train.copy()
X_test_filter=X_test.copy()
y_test_filter=y_test.copy()

corrmat={}
for i in X_train_filter.columns.values:
    corrmat[i]=X_train_filter[i].corr(y_train_filter)
del X_train_filter[7]
del X_train_filter[4]
del X_train_filter[3]
del X_train_filter['parch']
del X_train_filter['sibsp']
del X_train_filter['age']
del X_test_filter[7]
del X_test_filter[4]
del X_test_filter[3]
del X_test_filter['parch']
del X_test_filter['sibsp']
del X_test_filter['age']

from xgboost import XGBClassifier
model_filter=XGBClassifier()
model_filter.fit(X_train_filter,y_train_filter)


y_pred_filter=model_filter.predict(X_test_filter)

from sklearn.metrics import confusion_matrix
metric_filter=confusion_matrix(y_test_filter,y_pred_filter)

accuracy_filter_basic=(metric_filter[0][0]+metric_filter[1][1])/sum(sum(metric_filter))*100
print('Accuracy using filter : ',accuracy_filter_basic)
```
#### Statistical Filter Methods.
```python
#Information gain / Mutual information
X_train_filter=X_train.copy()
y_train_filter=y_train.copy()
X_test_filter=X_test.copy()
y_test_filter=y_test.copy()

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest,SelectPercentile
#mi=mutual_info_classif(X_train_filter,y_train_filter)
select=SelectKBest(mutual_info_classif,k=12).fit(X_train_filter,y_train_filter)
X_train_filter=X_train_filter[X_train_filter.columns[select.get_support()].values]
X_test_filter=X_test_filter[X_test_filter.columns[select.get_support()].values]
from xgboost import XGBClassifier
model_filter=XGBClassifier()
model_filter.fit(X_train_filter,y_train_filter)


y_pred_filter=model_filter.predict(X_test_filter)

from sklearn.metrics import confusion_matrix
metric_filter=confusion_matrix(y_test_filter,y_pred_filter)

accuracy_filter_basic=(metric_filter[0][0]+metric_filter[1][1])/sum(sum(metric_filter))*100
print('Accuracy using filter : ',accuracy_filter_basic)
```
```python
#Fisher score / chi square
#Information gain / Mutual information
X_train_filter=X_train.copy()
y_train_filter=y_train.copy()
X_test_filter=X_test.copy()
y_test_filter=y_test.copy()

from sklearn.feature_selection import SelectKBest, chi2
select=SelectKBest(chi2,k=10).fit(X_train_filter,y_train_filter)
X_train_filter=X_train_filter[X_train_filter.columns[select.get_support()].values]
X_test_filter=X_test_filter[X_test_filter.columns[select.get_support()].values]
from xgboost import XGBClassifier
model_filter=XGBClassifier()
model_filter.fit(X_train_filter,y_train_filter)


y_pred_filter=model_filter.predict(X_test_filter)

from sklearn.metrics import confusion_matrix
metric_filter=confusion_matrix(y_test_filter,y_pred_filter)

accuracy_filter_basic=(metric_filter[0][0]+metric_filter[1][1])/sum(sum(metric_filter))*100
print('Accuracy using filter : ',accuracy_filter_basic)
```
```python
# Univariate / ANOVA
X_train_filter=X_train.copy()
y_train_filter=y_train.copy()
X_test_filter=X_test.copy()
y_test_filter=y_test.copy()

from sklearn.feature_selection import SelectKBest, f_classif
select=SelectKBest(f_classif,k=11).fit(X_train_filter,y_train_filter)
X_train_filter=X_train_filter[X_train_filter.columns[select.get_support()].values]
X_test_filter=X_test_filter[X_test_filter.columns[select.get_support()].values]
from xgboost import XGBClassifier
model_filter=XGBClassifier()
model_filter.fit(X_train_filter,y_train_filter)


y_pred_filter=model_filter.predict(X_test_filter)

from sklearn.metrics import confusion_matrix
metric_filter=confusion_matrix(y_test_filter,y_pred_filter)

accuracy_filter_basic=(metric_filter[0][0]+metric_filter[1][1])/sum(sum(metric_filter))*100
print('Accuracy using filter : ',accuracy_filter_basic)
```
#### Wrapper Methods.
```python
#Forward feature selection
X_train_wrapper=X_train.copy()
y_train_wrapper=y_train.copy()
X_test_wrapper=X_test.copy()
y_test_wrapper=y_test.copy()
from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs1=SFS(XGBClassifier(n_jobs=4),k_features=9,forward=True,floating=False,verbose=2,scoring='roc_auc',cv=3)
sfs1.fit(np.array(X_train_wrapper),y_train_wrapper)
```
```python
#Backward feature selection
X_train_wrapper=X_train.copy()
y_train_wrapper=y_train.copy()
X_test_wrapper=X_test.copy()
y_test_wrapper=y_test.copy()
from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
sfs1=SFS(XGBClassifier(n_jobs=4),k_features=11,forward=False,floating=False,verbose=2,scoring='roc_auc',cv=3)
sfs1.fit(np.array(X_train_wrapper),y_train_wrapper)
```
```python
#Exhaustive feature selection
X_train_wrapper=X_train.copy()
y_train_wrapper=y_train.copy()
X_test_wrapper=X_test.copy()
y_test_wrapper=y_test.copy()
from xgboost import XGBClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
sfs1=EFS(XGBClassifier(n_jobs=4),min_features=4,max_features=5,scoring='roc_auc',print_progress=True,cv=2)
sfs1.fit(np.array(X_train_wrapper),y_train_wrapper)
```
#### Embedded Methods.
```python
X_train_embedded=X_train.copy()
y_train_embedded=y_train.copy()
X_test_embedded=X_test.copy()
y_test_embedded=y_test.copy()
del X_train_embedded['body_null']
del X_train_embedded[2]
del X_train_embedded[3]
del X_train_embedded[7]
del X_test_embedded['body_null']
del X_test_embedded[2]
del X_test_embedded[3]
del X_test_embedded[7]

from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train_embedded,y_train_embedded)
print(xgb.feature_importances_*100)

y_pred_embedded=xgb.predict(X_test_embedded)

from sklearn.metrics import confusion_matrix
metric_embedded=confusion_matrix(y_test_embedded,y_pred_embedded)

accuracy_embedded=(metric_embedded[0][0]+metric_embedded[1][1])/sum(sum(metric_embedded))*100
print('Accuracy using embedded : ',accuracy_embedded)
```
## OUPUT :
![image](./Screenshot%20from%202023-05-16%2015-57-12.png)
![image](./Screenshot%20from%202023-05-16%2015-57-35.png)
![image](./Screenshot%20from%202023-05-16%2015-57-42.png)
![image](./Screenshot%20from%202023-05-16%2015-57-58.png)
![image](./Screenshot%20from%202023-05-16%2015-58-04.png)
![image](./Screenshot%20from%202023-05-16%2015-58-17.png)
![image](./Screenshot%20from%202023-05-16%2015-58-22.png)
![image](./Screenshot%20from%202023-05-16%2015-58-28.png)
![image](./Screenshot%20from%202023-05-16%2015-58-33.png)
![image](./Screenshot%20from%202023-05-16%2015-59-01.png)
## RESULT :
Thus, various feature selection techniques have been performed on the given data set.
