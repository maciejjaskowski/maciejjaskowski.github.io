---
title: Pandas + Scikit learn
---

I still struggle to decide whether to use R or Python + Pandas + SciKit. For now I dive into Python. The Python way has definitely a steeper learning curve. On the other hand there is a couple of goodies that, once you learn how to use them, might pay off greatly.

Below I show an example on how nicely you structure your code with the help of `sklearn.pipeline.Pipeline` and `sklearn.preprocessing.*`.


```python
from __future__ import division
import csv as csv
import numpy as np

import pandas as pd
from pandas import get_dummies
from pandas import DataFrame

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn_pandas import DataFrameMapper
```

Let's load the data from Kaggle Titanic competition.


{% highlight python %}
df_train = pd.read_csv('train.csv', header = 0, index_col = 'PassengerId')
df_test = pd.read_csv('test.csv', header = 0, index_col = 'PassengerId')
df = pd.concat([df_train, df_test], keys=["train", "test"])
{% endhighlight %}


```python
df['Title'] = df['Name'].apply(lambda c: c[c.index(',') + 2 : c.index('.')])
title_min_map = pd.Series({ 
  'Mr': 'Mr', 'Mrs': 'Mrs', 'Miss': 'Miss', 'Master': 'Master', 
  'Don': 'LikelyDead', 'Rev': 'LikelyDead', 'Dr': 'Mr', 'Mme': 'LikelyAlive', 
  'Ms': 'LikelyAlive', 'Major': 'Mr', 'Capt': 'Mr', 'Lady': 'LikelyAlive', 
  'Sir': 'Mr', 'Mlle': 'LikelyAlive', 'Col': 'Mr', 'the Countess': 'LikelyAlive', 
  'Jonkheer': 'LikelyDead', 'Dona': 'LikelyAlive'})
df['TitleMin'] = df['Title'].map(title_min_map)

df['LastName'] = df['Name'].apply(lambda n: n[0:n.index(',')])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].mode()[0]
df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].mode()[0]
df['FamilyID'] = df['LastName'] + ':' + df['FamilySize'].apply(str)
df.loc[df['FamilySize'] <= 2, 'FamilyID'] = 'Small_Family'

df['AgeOriginallyNaN'] = df['Age'].isnull().astype(int)
medians_by_title = pd.DataFrame(df.groupby('TitleMin')['Age'].median()) \
  .rename(columns = {'Age': 'AgeFilledMedianByTitle'})
df = df.merge(medians_by_title, left_on = 'TitleMin', right_index = True) \
  .sort_index(level = 0).sort_index(level = 1)
df_train = df[:len(df.ix['train'])]
df_test  = df[len(df.ix['train']):]
```


```python
def featurize(features):
  transformations = [
                            ('Embarked', LabelBinarizer()),
                            ('Fare', None),
                            ('Parch', None),
                            ('Pclass', LabelBinarizer()),
                            ('Sex', LabelBinarizer()),
                            ('SibSp', None),                                       
                            ('Title', LabelBinarizer()),
                            ('TitleMin', LabelBinarizer()),
                            ('FamilySize', None),
                            ('FamilyID', LabelBinarizer()),
                            ('AgeOriginallyNaN', None),
                            ('AgeFilledMedianByTitle', None)]

  return DataFrameMapper(filter(lambda x: x[0] in df.columns, transformations))

```


```python
features = ['Survived', 'Sex', 'Title', 'FamilySize', 'AgeFilledMedianByTitle',
            'Embarked', 'Pclass', 'FamilyID', 'AgeOriginallyNaN']

pipeline = Pipeline([('featurize', featurize(features)), ('forest', RandomForestClassifier())])
```


```python
model = pipeline.fit(X = df_train[df_train.columns.drop('Survived')], y = df_train['Survived'])
```


```python
model.predict(df_test)
```




    array([ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,
            0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  0.,
            1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,
            1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,
            1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,
            1.,  1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,
            0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,
            0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,
            1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
            0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
            1.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,
            0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,
            1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,
            0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,
            1.,  0.,  1.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,
            0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
            0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
            1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,
            0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,
            0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  1.,  1.,
            0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,
            1.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
            0.,  1.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,
            0.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,
            0.,  1.])


