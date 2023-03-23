from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
import joblib

# Loading MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
m = X_train.shape[0]
# Data normalization
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Searching best hyperparameters
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_,grid_search.best_score_)

# Model training
knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf = knn_clf.fit(X_train,y_train)
prediction = knn_clf.predict(X_test)
print(accuracy_score(y_test,prediction))

# Data augmentation
datagen = ImageDataGenerator(
    width_shift_range= 2.0,
    height_shift_range= 2.0,
)
datagen.fit(X_train.reshape(X_train.shape[0], 28, 28, 1))
data_generator = datagen.flow(X_train.reshape(X_train.shape[0], 28, 28, 1),shuffle=False, batch_size=1)
X_train_aug = [data_generator.next() for i in range(0, m * 4)]
X_train_aug = np.asarray(X_train_aug).reshape(m * 4, 28 * 28)
X_train_aug = np.vstack((X_train, X_train_aug))
y_train_aug = np.concatenate((y_train, y_train, y_train, y_train,y_train))

# Model training after data augmentation
knn_clf = knn_clf.fit(X_train_aug,y_train_aug)
prediction = knn_clf.predict(X_test)
print(accuracy_score(y_test,prediction))

# Saving model
filename = "KNN_clf.joblib"
joblib.dump(knn_clf, filename)