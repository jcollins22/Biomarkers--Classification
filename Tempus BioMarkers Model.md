```python
import json
import pandas as pd
import numpy as np
```


```python
#import data
biomarkers = pd.DataFrame(pd.read_csv("biomarkers.csv"))
targets = pd.DataFrame(pd.read_csv("targets.csv"))
patients = open("patient_profiles.json")
```


```python
#check proper import
print(type(biomarkers))
print(type(targets))
print(type(patients))
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    <class '_io.TextIOWrapper'>



```python
#fix input problem
patients_str = patients.read()
patients_json = json.loads(patients_str)[0]
type(patients_json)
```




    dict




```python
#unfold nested dictionaries into dataframe
patients = patients_json
patients = pd.DataFrame(patients)
patients = pd.json_normalize(patients["patient_profiles"])
```


```python
bioTarget = pd.merge(biomarkers, targets, on='biomarker_id')
fullFile = pd.merge(bioTarget, patients, on="patient_id")
fullFile.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>biomarker_id</th>
      <th>BM00000</th>
      <th>BM00001</th>
      <th>BM00002</th>
      <th>BM00003</th>
      <th>BM00004</th>
      <th>BM00005</th>
      <th>BM00006</th>
      <th>BM00007</th>
      <th>BM00008</th>
      <th>...</th>
      <th>patient_id</th>
      <th>target_label</th>
      <th>demographics.gender</th>
      <th>demographics.age</th>
      <th>status.disease_sub_type</th>
      <th>status.comorbidity_index</th>
      <th>status.cohort_qualifier</th>
      <th>status.smoking_status</th>
      <th>status.months_since_diagnosis</th>
      <th>demographics.race</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101219d6e</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1c9af69ad</td>
      <td>0</td>
      <td>Male</td>
      <td>69</td>
      <td>A</td>
      <td>NaN</td>
      <td>True</td>
      <td>current</td>
      <td>0</td>
      <td>Black or African American</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1039b491e</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>11c7927cb</td>
      <td>1</td>
      <td>Male</td>
      <td>68</td>
      <td>A</td>
      <td>NaN</td>
      <td>True</td>
      <td>current</td>
      <td>33</td>
      <td>White</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10612bd12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>5a30e008</td>
      <td>0</td>
      <td>Female</td>
      <td>40</td>
      <td>A</td>
      <td>NaN</td>
      <td>True</td>
      <td>current</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1068da327</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>bbb9a595</td>
      <td>0</td>
      <td>Male</td>
      <td>60</td>
      <td>A</td>
      <td>NaN</td>
      <td>True</td>
      <td>current</td>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>107fb394c</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1504ec8e9</td>
      <td>0</td>
      <td>Male</td>
      <td>66</td>
      <td>B</td>
      <td>NaN</td>
      <td>True</td>
      <td>never</td>
      <td>13</td>
      <td>White</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 15168 columns</p>
</div>




```python
fullFile.isna().sum()
```




    biomarker_id                       0
    BM00000                            0
    BM00001                            0
    BM00002                            0
    BM00003                            0
                                    ... 
    status.comorbidity_index         178
    status.cohort_qualifier            0
    status.smoking_status              0
    status.months_since_diagnosis      0
    demographics.race                 44
    Length: 15168, dtype: int64




```python
#most of cohort does not have comorbidty index
del fullFile["status.comorbidity_index"]

#fix NaN
fullFile['demographics.race'].fillna("unknown")
```




    0             Black or African American
    1                                 White
    2                               unknown
    3                               unknown
    4                                 White
                         ...               
    226           Black or African American
    227                               White
    228                             unknown
    229    American Indian or Alaska Native
    230                               White
    Name: demographics.race, Length: 231, dtype: object




```python
#one hot encoding
smoking_onehot = pd.get_dummies(fullFile["status.smoking_status"])
race_onehot = pd.get_dummies(fullFile['demographics.race'])
subtype_onehot = pd.get_dummies(fullFile["status.disease_sub_type"])
gender_onehot = pd.get_dummies(fullFile["demographics.gender"])
cohort_onehot = pd.get_dummies(fullFile["status.cohort_qualifier"])
patients_onehot = smoking_onehot.join(race_onehot)
patients_onehot1 = patients_onehot.join(subtype_onehot)
patients_onehot2 = patients_onehot1.join(gender_onehot)
patients_onehot3 = patients_onehot2.join(cohort_onehot)
patients_onehot3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>current</th>
      <th>former</th>
      <th>never</th>
      <th>unknown</th>
      <th>American Indian or Alaska Native</th>
      <th>Asian</th>
      <th>Black or African American</th>
      <th>Native Hawaiian or Other Pacific Islander</th>
      <th>White</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>Female</th>
      <th>Male</th>
      <th>True</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_onehot = patients_onehot3.join(fullFile)
full_onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>current</th>
      <th>former</th>
      <th>never</th>
      <th>unknown</th>
      <th>American Indian or Alaska Native</th>
      <th>Asian</th>
      <th>Black or African American</th>
      <th>Native Hawaiian or Other Pacific Islander</th>
      <th>White</th>
      <th>A</th>
      <th>...</th>
      <th>BM15156</th>
      <th>patient_id</th>
      <th>target_label</th>
      <th>demographics.gender</th>
      <th>demographics.age</th>
      <th>status.disease_sub_type</th>
      <th>status.cohort_qualifier</th>
      <th>status.smoking_status</th>
      <th>status.months_since_diagnosis</th>
      <th>demographics.race</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1c9af69ad</td>
      <td>0</td>
      <td>Male</td>
      <td>69</td>
      <td>A</td>
      <td>True</td>
      <td>current</td>
      <td>0</td>
      <td>Black or African American</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>11c7927cb</td>
      <td>1</td>
      <td>Male</td>
      <td>68</td>
      <td>A</td>
      <td>True</td>
      <td>current</td>
      <td>33</td>
      <td>White</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>5a30e008</td>
      <td>0</td>
      <td>Female</td>
      <td>40</td>
      <td>A</td>
      <td>True</td>
      <td>current</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>bbb9a595</td>
      <td>0</td>
      <td>Male</td>
      <td>60</td>
      <td>A</td>
      <td>True</td>
      <td>current</td>
      <td>21</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1504ec8e9</td>
      <td>0</td>
      <td>Male</td>
      <td>66</td>
      <td>B</td>
      <td>True</td>
      <td>never</td>
      <td>13</td>
      <td>White</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 15185 columns</p>
</div>




```python
#remove unneeded columns
del full_onehot["demographics.gender"]
del full_onehot["demographics.age"]
del full_onehot["status.cohort_qualifier"]
del full_onehot["demographics.race"]
del full_onehot["patient_id"]
del full_onehot["biomarker_id"]
del full_onehot["status.smoking_status"]
del full_onehot["status.disease_sub_type"]
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /usr/local/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2645             try:
    -> 2646                 return self._engine.get_loc(key)
       2647             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'demographics.gender'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-35-1f453949a6d6> in <module>
          1 #remove unneeded columns
    ----> 2 del full_onehot["demographics.gender"]
          3 del full_onehot["demographics.age"]
          4 del full_onehot["status.cohort_qualifier"]
          5 del full_onehot["demographics.race"]


    /usr/local/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py in __delitem__(self, key)
       3757             # there was no match, this call should raise the appropriate
       3758             # exception:
    -> 3759             self._data.delete(key)
       3760 
       3761         # delete from the caches


    /usr/local/anaconda3/lib/python3.7/site-packages/pandas/core/internals/managers.py in delete(self, item)
       1000         Delete selected item (items if non-unique) in-place.
       1001         """
    -> 1002         indexer = self.items.get_loc(item)
       1003 
       1004         is_deleted = np.zeros(self.shape[0], dtype=np.bool_)


    /usr/local/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2646                 return self._engine.get_loc(key)
       2647             except KeyError:
    -> 2648                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2649         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2650         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'demographics.gender'



```python
#create a test vs train random split of data
X = full_onehot
Y = full_onehot["target_label"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
data_test = X_test
data_test["target_label"] = y_test
```

    /usr/local/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      import sys



```python
del X_train["target_label"]
del X_test["target_label"]
```


```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
#evaluate the classification tree model
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    [[48  8]
     [ 9  5]]
                  precision    recall  f1-score   support
    
               0       0.84      0.86      0.85        56
               1       0.38      0.36      0.37        14
    
        accuracy                           0.76        70
       macro avg       0.61      0.61      0.61        70
    weighted avg       0.75      0.76      0.75        70
    



```python
#import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#create object of the classifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train
neigh.fit(X_train, y_train)
#predict
pred = neigh.predict(data_test)
#evaluate
print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))
```

    KNeighbors accuracy score :  0.6714285714285714



```python
#import
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)
#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(X_train, y_train).predict(data_test)
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(y_test, pred, normalize = True))
```

    LinearSVC accuracy :  0.8142857142857143



```python
#import
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(X_train, y_train).predict(data_test)
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))
```

    Naive-Bayes accuracy :  0.8



```python

```
