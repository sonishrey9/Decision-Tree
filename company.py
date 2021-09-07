# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# %%
data = pd.read_csv("Company_Data.csv")
data


# %%
data.isna().sum()


# %%
data.info()


# %%
#converting into binary
lb = LabelEncoder()


# %%
for i in data.Sales:

    if i <= 5 :
        data.Sales = data.Sales.replace([i], "Low")
    elif i >=10:
        data.Sales = data.Sales.replace([i], "High")
    else:
        data.Sales = data.Sales.replace([i], "Medium")


# %%
categorical_features=[feature for feature in data.columns if data[feature].dtypes=='O']
categorical_features


# %%
#converting into binary
lb = LabelEncoder()

for i in categorical_features:
    data[i] = lb.fit_transform(data[i])

data


# %%
data.info()


# %%
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# %%
data.columns.to_list()


# %%
#split dataset in features and target variable
feature_cols = ['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']
X = data[feature_cols] # Features
y = data.Sales # Target variable


# %%
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test

# %% [markdown]
# ## **Building Decision Tree Model**

# %%
from sklearn.tree import DecisionTreeClassifier


# %%
# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion='entropy')

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

# #Predict the response for test dataset
y_pred = clf.predict(X_test)


# %%
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# %% [markdown]
# ## **Random Forest**

# %%
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(criterion = 'entropy',n_estimators=100)
model1.fit(X_train, y_train)


# %%
model1.score(X_test,y_test)


# %%
import six
import sys
sys.modules['sklearn.externals.six'] = six

# %% [markdown]
# ## **Optimizing Decision Tree Performance**

# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# %%



