import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

y_train = np.array(train_df.iloc[:, 0])
X_train = np.array(train_df.drop('label', axis=1))
y_test = np.array(test_df.iloc[:, 0])
X_test = np.array(test_df.drop('label', axis=1))

logReg = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)

img = cv2.imread('trousers.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('bag.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (28, 28))
img2 = cv2.resize(img2, (28, 28))

img = img / 255.0
img2 = img2 / 255.0

cv2.imshow('Trouser Image', img)
cv2.imshow('Bag Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = img.reshape(1,28*28)
img2 = img2.reshape(1,28*28)

img_pred = logReg.predict(img)
img2_pred = logReg.predict(img2)

label_dict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle Boot'}

print(f"The prediction for the trouser image: {label_dict[img_pred[0]]}")
print(f"The prediction for the bag image: {label_dict[img2_pred[0]]}")


