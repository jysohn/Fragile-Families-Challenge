# Imports
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

data_raw = pd.read_csv('output.csv')

# Raw Baseline
# Format Training Data
y_train_raw = pd.read_csv('train.csv')
y_train_raw = y_train_raw[["challengeID", "layoff"]].dropna()

challengeIDsLAY = y_train_raw.challengeID
X_train_raw = data_raw[data_raw["challengeID"].isin(challengeIDsLAY)]
X_train_raw = X_train_raw.select_dtypes([np.number])

X_train_raw = X_train_raw.set_index("challengeID")
y_train_raw = y_train_raw.set_index("challengeID")

# Format Test Data
y_test_raw = pd.read_csv('test.csv')
y_test_raw = y_test_raw[['challengeID', 'layoff']].dropna()

challengeIDsTest = y_test_raw.challengeID
X_test_raw = data_raw[data_raw["challengeID"].isin(challengeIDsTest)]
X_test_raw = X_test_raw.set_index("challengeID")
X_test_raw = X_test_raw.select_dtypes([np.number])

y_test_raw = y_test_raw.set_index("challengeID")

temp = []
for x in y_test_raw.layoff.values:
    temp.append(np.bool_(x))
y_test_raw = np.array(temp)

# Basic Logit with no selection at all
from sklearn.linear_model import LogisticRegression
clf_logit_raw = LogisticRegression(solver="liblinear")
clf_logit_raw.fit(X_train_raw, y_train_raw.values.ravel())

y_pred_lay_raw = clf_logit_raw.predict(X_test_raw)
#y_pred_lay_raw = y_pred_lay_raw == 1

y_pred_lay_raw_score = clf_logit_raw.decision_function(X_test_raw)

print("Accuracy: " + str(accuracy_score(y_test_raw, y_pred_lay_raw)))
print("MSE: " + str(mean_squared_error(y_test_raw, y_pred_lay_raw)))
print("ROC-AUCE: " + str(roc_auc_score(y_test_raw, y_pred_lay_raw_score)))

# Select the Constructed Variables
# Get all the indeces of non-constructed variables
i = 0
non_constructed = []
for column in data_raw.columns:
    if column[0] != 'c':
        non_constructed.append(i)
    i += 1
# Drop all non-constructed variables
X = data_raw.drop(data_raw.columns[non_constructed], axis=1)

# Drop NAN Rows
y_train_raw = pd.read_csv('train.csv')
y_train_gpa = y_train_raw[["challengeID", "gpa"]].dropna()
y_train_lay = y_train_raw[["challengeID", "layoff"]].dropna()

# Align Train and Test Data
challengeIDsLAY = y_train_lay.challengeID
X_train_lay = X[X["challengeID"].isin(challengeIDsLAY)]
X_train_lay = X_train_lay.set_index("challengeID")
y_train_lay = y_train_lay.set_index("challengeID")

# Logit on Constructed
from sklearn.linear_model import LogisticRegression
clf_logit = LogisticRegression(solver="liblinear")
clf_logit.fit(X_train_lay_up, y_train_lay_up.ravel())

# SVM on Constructed
from sklearn import svm
clf_svm = svm.SVC(gamma='auto').fit(X_train_lay_up, y_train_lay_up.ravel())

# Format Test Data
# Testing
y_test_raw = pd.read_csv("test.csv")

y_test_lay = y_test_raw[["challengeID", "layoff"]].dropna()
y_test_lay = y_test_lay

challengeIDsLAY_test = y_test_lay.challengeID

X_test_lay = X[X["challengeID"].isin(challengeIDsLAY_test)]
X_test_lay = X_test_lay.set_index("challengeID")
y_test_lay = y_test_lay.set_index("challengeID")
X_test_lay.shape, y_test_lay.shape

temp = []
for x in y_test_lay.layoff.values:
    temp.append(np.bool_(x))
y_test_lay = np.array(temp)

# Analyze Performance of Baseline Models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_pred_lay1 = clf_logit.predict(X_test_lay)
#y_pred_lay1 = y_pred_lay1 == 1
y_pred_lay1_score = clf_logit.decision_function(X_test_lay)
print("Logit accuracy: " + str(accuracy_score(y_test_lay, y_pred_lay1)))
print("Logit MSE: " + str(mean_squared_error(y_test_lay, y_pred_lay1)))
print("Logit ROC-AUC: " + str(roc_auc_score(y_test_lay, y_pred_lay1_score)))

y_pred_lay2 = clf_svm.predict(X_test_lay)
#y_pred_lay2 = y_pred_lay2 == 1
y_pred_lay2_score = clf_svm.decision_function(X_test_lay)
print("SVM accuracy: " + str(accuracy_score(y_test_lay, y_pred_lay2)))
print("SVM MSE: " + str(mean_squared_error(y_test_lay, y_pred_lay2)))
print("SVM ROC-AUC: " + str(roc_auc_score(y_test_lay, y_pred_lay2_score)))

y_pred_lay3 = clf_rf.predict(X_test_lay)
y_pred_lay2 = y_pred_lay2 == 1

# Plot ROC Curves
y_score_logit = clf_logit.decision_function(X_test_lay)
y_score_svm = clf_svm.decision_function(X_test_lay)
y_score_raw_logit = clf_logit_raw.decision_function(X_test_raw)

from sklearn.metrics import roc_curve
fpr3, tpr3, threshold3 = roc_curve(y_test_lay, y_score_logit)
fpr4, tpr4, threshold4 = roc_curve(y_test_lay, y_score_svm)
fpr5, tpr5, threshold5 = roc_curve(y_test_lay, y_score_raw_logit)

plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr5, tpr5, label='Logit (All Variables)', color='b')
plt.plot(fpr3, tpr3, label='Logit (Constructed Variables)', color='r')

plt.plot(fpr4, tpr4, label='SVM', color='y')


# Recursive Feature Elimination with Cross Validation
from matplotlib import pyplot as plt
plt.title('ROC Curves')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

logit = LogisticRegression(solver="liblinear", max_iter=200)
rfecv = RFECV(estimator=logit, step=1, min_features_to_select=1, cv=StratifiedKFold(3),
              scoring='accuracy')
rfecv.fit(X_train_lay, y_train_lay.values.ravel())

print("Optimal number of features : %d" % rfecv.n_features_)

# See RFE Graph
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.title("RFE Cross Validation Score by Number of Features")
plt.ylabel("Cross validation score (Classification Accuracy)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# See highest accuracy variables according to RFE
rfecv.ranking_
X_train_lay.columns
feature_rankings = pd.DataFrame(rfecv.ranking_, X_train_lay.columns)
feature_rankings.sort_values(by=0).head(n=35)

# 1 variable case (RFE)
X_train_lay_1 = X_train_lay[["cf2fint"]]
clf_logit_1 = LogisticRegression(solver="liblinear")
clf_logit_1.fit(X_train_lay_1, y_train_lay.values.ravel())

# Extra Trees
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=1000)
clf = clf.fit(X_train_lay, y_train_lay.values.ravel())
model = SelectFromModel(clf, prefit=True)
X_test_lay2 = model.transform(X_test_lay)
X_train_lay2 = model.transform(X_train_lay)

X_test_lay2.shape, X_train_lay2.shape

clf_logit_extra_trees = LogisticRegression(solver="liblinear")
clf_logit_extra_trees.fit(X_train_lay2, y_train_lay.values.ravel())
y_pred_lay_extra_trees = clf_logit_extra_trees.predict(X_test_lay2)

# Select K-Best
# feature selection (comment out to see results w/o feature selection)
from sklearn.feature_selection import SelectKBest, chi2
clf_logit1 = LogisticRegression(solver="liblinear")
mse = 1.0
for i in range(550):
    kbest = SelectKBest(chi2, k=i+1).fit(X_train_lay, y_train_lay)
    X_test_lay1 = kbest.transform(X_test_lay)
    X_train_lay1 = kbest.transform(X_train_lay)
    clf_logit1.fit(X_train_lay1, y_train_lay.values.ravel())
    y_pred_lay_k = clf_logit1.predict(X_test_lay1)
    mse = mean_squared_error(y_test_lay, y_pred_lay_k)
#     if (score > temp):
    if (mse < temp):
        temp = mse
        index = i
print("Selected " + str(index+1) + " features: " + str(temp))

# PCA
from sklearn.decomposition import PCA

# Fitting the PCA algorithm with our Data
pca = PCA().fit(X_train_lay)
# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.xlim(0, 20)
plt.ylabel('Variance (%)')  # for each component
plt.title('Fragile Families Explained Variance by Number of Components')
plt.show()

# PCA Actual Model
pca = PCA(n_components=30)
pca.fit(X_train_lay)
X_train_lay_pca = pca.transform(X_train_lay)
X_test_lay_pca = pca.transform(X_test_lay)
clf_logit_pca = LogisticRegression(solver="liblinear", max_iter=400)
clf_logit_pca.fit(X_train_lay_pca, y_train_lay.values.ravel())
y_pred_lay_pca = clf_logit_pca.predict(X_test_lay_pca)
mse = mean_squared_error(y_test_lay, y_pred_lay_pca)
print(pd.DataFrame(confusion_matrix(y_test_lay, y_pred_lay_pca, labels=[
      0, 1]), index=['True: 0', 'True: 1'], columns=['Predicted: 0', 'Predicted: 1']))
print("PCA MSE: " + str(mean_squared_error(y_test_lay, y_pred_lay_pca)))

# Results of Feature Selection Methods
from sklearn.metrics import roc_auc_score

y_score_logit = clf_logit.decision_function(X_test_lay)
y_score_extra = clf_logit_extra_trees.decision_function(X_test_lay2)
y_score_kbest = clf_logit_kbest.decision_function(kbest.transform(X_test_lay))
y_score_pca = clf_logit_pca.decision_function(X_test_lay_pca)
y_score_rfe = clf_logit_1.decision_function(X_test_lay_1)

from sklearn.metrics import roc_curve
fpr1, tpr1, threshold1 = roc_curve(y_test_lay, y_score_logit)
fpr2, tpr2, threshold2 = roc_curve(y_test_lay, y_score_extra)
fpr3, tpr3, threshold3 = roc_curve(y_test_lay, y_score_kbest)
fpr4, tpr4, threshold4 = roc_curve(y_test_lay, y_score_pca)
fpr5, tpr5, threshold5 = roc_curve(y_test_lay, y_score_rfe)

plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr1, tpr1, label='All constructed variables', color='r')
plt.plot(fpr2, tpr2, label='Extra trees', color='y')
plt.plot(fpr3, tpr3, label='K-best', color='b')
plt.plot(fpr4, tpr4, label='PCA', color='m')
plt.plot(fpr5, tpr5, label='RFE', color='skyblue')

from matplotlib import pyplot as plt
plt.title('ROC Curves with Different Feature Selection Methods (Logit)')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

print("All constructed variables ROC Score = " +
      str(roc_auc_score(y_test_lay, y_score_logit)))
print("Extra Trees ROC Score = " + str(roc_auc_score(y_test_lay, y_score_extra)))
print("K-Best ROC Score = " + str(roc_auc_score(y_test_lay, y_score_kbest)))
print("PCA ROC Score = " + str(roc_auc_score(y_test_lay, y_score_pca)))
