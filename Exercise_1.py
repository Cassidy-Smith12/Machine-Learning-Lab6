import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay


train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

y_train = np.array(train.iloc[:, 0])
X_train = np.array(train.drop('label', axis=1))
y_test = np.array(test.iloc[:, 0])
X_test = np.array(test.drop('label', axis=1))


logReg = LogisticRegression(solver='lbfgs').fit(X_train, y_train)

pred = logReg.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred)}")

print(f"Classification Report:\n{classification_report(y_test, pred)}")

conf_mat = confusion_matrix(y_test, pred)
print(f"Confusion Matrix:\n{conf_mat}")

ConfustionMatrix = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=logReg.classes_)
ConfustionMatrix.plot()
plt.show()

