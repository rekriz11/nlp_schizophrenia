import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes	import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams

## load the data
df = pd.read_csv('wolf-worden_language_measures_20200211.csv')

## define X and y
TLC = df.iloc[:,22:44] 
X = TLC

df['group'] = np.where(df['group']=="SSD", 1, 0)
y = df['group']

accList = np.empty(0)
probaList = np.empty(0)
testList = np.empty(0)
predList = np.empty(0)
train_pred_list = np.empty(0)

## define train_test_split
loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
	X_train, X_test, y_train, y_test = X.iloc[train_index,:], X.iloc[test_index, :], y[train_index], y[test_index]

	## impute missing values
	imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	X_train = imputer.fit_transform(X_train)
	X_test = imputer.transform(X_test)

	## scale features
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	## train the model
	clf = GaussianNB()
	clf.fit(X_train, y_train)

	## check train accuracy
	train_acc = accuracy_score(y_train, clf.predict(X_train))
	train_pred_list = np.append(train_pred_list, train_acc)

	## predict 
	y_pred = clf.predict(X_test)
	probas = clf.predict_proba(X_test)[0, 1]
	probaList = np.append(probaList, probas)

	acc = accuracy_score(y_test,y_pred)
	testList = np.append(testList, y_test)
	predList = np.append(predList, y_pred)
	accList = np.append(accList, acc)

## check the overall performance
print("mean acc in train:", np.mean(train_pred_list))
print("mean acc in test:", np.mean(accList))
auc_score = roc_auc_score(testList, probaList)
print("auc:", auc_score)
print(classification_report(testList, predList))
print(confusion_matrix(testList, predList))

#### plot ROC curve 
## parameters for the plot
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 13
rcParams['axes.labelsize'] = 'x-large'

fpr, tpr, thersholds = roc_curve(testList, probaList)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.plot(fpr, tpr, color='b', label=r'ROC (AUC = {0:0.2f})'
		''.format(auc_score), lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
#plt.savefig('AUC_TLC_only.png', format='png', dpi=300)

