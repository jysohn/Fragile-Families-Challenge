#!/usr/bin/env python
# coding: utf-8

# # Imports and Preprocessing

# In[3]:


import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data_raw = pd.read_csv('output.csv')


# ## X: Only select constructed variables

# In[5]:


# Get all the indeces of non-constructed variables
i = 0
non_constructed = []
for column in data_raw.columns:
    if column[0] != 'c':
        non_constructed.append(i)
    i += 1


# In[6]:


# Drop all non-constructed variables
X = data_raw.drop(data_raw.columns[non_constructed], axis=1)


# ## y_train: Drop NaN columns

# In[7]:


y_train_raw = pd.read_csv('train.csv')
y_train_lay = y_train_raw[["challengeID", "layoff"]].dropna()


# ## X_train and y_train: Align the two dataframes

# In[8]:


challengeIDsLAY = y_train_lay.challengeID
X_train_lay = X[X["challengeID"].isin(challengeIDsLAY)]
X_train_lay = X_train_lay.set_index("challengeID")
y_train_lay = y_train_lay.set_index("challengeID")


# # Logistic Regression, SVM (Layoff)

# In[9]:


from sklearn.linear_model import LogisticRegression
clf_logit = LogisticRegression(solver="liblinear")
clf_logit.fit(X_train_lay, y_train_lay.values.ravel())

from sklearn import svm
clf_svm = svm.SVC(gamma = 'auto').fit(X_train_lay, y_train_lay.values.ravel())


# In[10]:


# Testing
y_test_raw = pd.read_csv("test.csv")

y_test_lay = y_test_raw[["challengeID", "layoff"]].dropna()
y_test_lay = y_test_lay

challengeIDsLAY_test = y_test_lay.challengeID

X_test_lay = X[X["challengeID"].isin(challengeIDsLAY_test)]
X_test_lay = X_test_lay.set_index("challengeID")
y_test_lay = y_test_lay.set_index("challengeID")
X_test_lay.shape, y_test_lay.shape


# In[11]:


temp = []
for x in y_test_lay.layoff.values:
    temp.append(np.bool_(x))
y_test_lay = np.array(temp)


# In[16]:


#feature selection (comment out to see results w/o feature selection)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
clf_logit1 = LogisticRegression(solver="liblinear")
temp = 0.0
for i in range(590):
    kbest = SelectKBest(chi2, k=i+1).fit(X_train_lay, y_train_lay)
    X_test_lay1 = kbest.transform(X_test_lay)
    X_train_lay1 = kbest.transform(X_train_lay)
    clf_logit1.fit(X_train_lay1, y_train_lay.values.ravel())
    score = clf_logit1.score(X_test_lay1, y_test_lay)
    if (score > temp):
        temp = score
        index = i
print("Selected " + str(index+1) + " features: " + str(temp))


# In[13]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train_lay, y_train_lay.values.ravel())
model = SelectFromModel(clf, prefit=True)
X_test_lay2 = model.transform(X_test_lay)
X_train_lay2 = model.transform(X_train_lay)

X_test_lay2.shape, X_train_lay2.shape


# In[32]:


from sklearn.linear_model import LogisticRegression
clf_logit2 = LogisticRegression(solver="liblinear")
clf_logit2.fit(X_train_lay2, y_train_lay.values.ravel())
score1 = clf_logit2.score(X_test_lay2, y_test_lay)
print(score1)


# In[70]:


from sklearn import svm
clf_svm1 = svm.SVC(gamma = 'auto').fit(X_train_lay1, y_train_lay.values.ravel())


# In[71]:


clf_logit.score(X_test_lay, y_test_lay)


# In[72]:


clf_logit1.score(X_test_lay1, y_test_lay)


# In[73]:


clf_svm.score(X_test_lay, y_test_lay)


# In[74]:


clf_svm1.score(X_test_lay1, y_test_lay)


# In[1]:


clf.score(X_test_lay2, y_test_lay)


# In[33]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_pred_lay1 = clf_logit.predict(X_test_lay)
y_pred_lay1 = y_pred_lay1 == 1
accuracy_score(y_test_lay, y_pred_lay1)

y_pred_lay2 = clf_svm.predict(X_test_lay)
y_pred_lay2 = y_pred_lay2 == 1
accuracy_score(y_test_lay, y_pred_lay2)

y_pred_lay_extratree = clf_logit2.predict(X_test_lay2)
y_pred_lay_extratree = y_pred_lay_extratree == 1
accuracy_score(y_test_lay, y_pred_lay_extratree)

print(pd.DataFrame(confusion_matrix(y_test_lay, y_pred_lay1, labels=[0, 1]), index=['True: 0', 'True: 1'], columns=['Predicted: 0', 'Predicted: 1']))
print(pd.DataFrame(confusion_matrix(y_test_lay, y_pred_lay2, labels=[0, 1]), index=['True: 0', 'True: 1'], columns=['Predicted: 0', 'Predicted: 1']))
print(pd.DataFrame(confusion_matrix(y_test_lay, y_pred_lay_extratree, labels=[0, 1]), index=['True: 0', 'True: 1'], columns=['Predicted: 0', 'Predicted: 1']))


# In[ ]:


#feature selection (comment out to see results w/o feature selection)
from sklearn.feature_selection import SelectKBest, chi2
kbest = SelectKBest(chi2, k=100).fit(X_train, y_train)
X_train = kbest.transform(X_train)
X_test = kbest.transform(X_test)

