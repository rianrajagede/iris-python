from __future__ import print_function
from builtins import range

"""
SECTION 1 : Load and setup data for training
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
datatrain = pd.read_csv('../Datasets/iris/iris.csv')

# Change string value to numeric
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)

# Change dataframe to array
datatrain_array = datatrain.values

# Split x and y (feature and target)
X_train, X_test, y_train, y_test = train_test_split(datatrain_array[:,:4],
                                                    datatrain_array[:,4],
                                                    test_size=0.2)

"""
SECTION 2 : Build and Train Model

Random Forest model with 500 trees
"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500, random_state=113)

# Train the model
rfc.fit(X_train, y_train)

# Test the model
print(rfc.score(X_test,y_test))

sl = 5.8
sw = 4
pl = 1.2
pw = 0.2
data = [[sl,sw,pl,pw]]
print(rfc.predict(data))



