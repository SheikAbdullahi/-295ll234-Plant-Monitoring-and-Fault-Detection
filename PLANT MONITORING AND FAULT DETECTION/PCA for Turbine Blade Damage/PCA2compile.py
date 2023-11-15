from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the MATLAB file
data = loadmat('/Users/sheik/Downloads/AllData_ExTxS.mat')

# Extracting and unfolding 'act_1_train' data
act_1_train = data['act_1_train']
act_1_train_unfolded = act_1_train.reshape(act_1_train.shape[0], -1)

# Applying PCA to the unfolded training data
pca = PCA()
pca.fit(act_1_train_unfolded)

# Transforming the training and test data
act_1_test = data['act_1_test']
act_1_test_unfolded = act_1_test.reshape(act_1_test.shape[0], -1)
pca_test_data = np.dot(act_1_test_unfolded, pca.components_.T)

# Feature Selection: Choosing number of components for 95% variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1

# Reducing training and testing data to the first 11 principal components
pca_train_data = pca.transform(act_1_train_unfolded)[:, :n_components_95]
pca_test_data_reduced = pca_test_data[:, :n_components_95]

# Assuming labels are available in the data file
train_labels = data.get('act_1_train_labels', np.zeros(act_1_train.shape[0]))
test_labels = data.get('act_1_test_labels', np.zeros(act_1_test.shape[0]))

# Splitting the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(pca_train_data, train_labels, test_size=0.3, random_state=42)

# Training a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Validating the model
val_predictions = dt_classifier.predict(X_val)
conf_matrix_val = confusion_matrix(y_val, val_predictions)
accuracy_val = accuracy_score(y_val, val_predictions)
classification_rep_val = classification_report(y_val, val_predictions)

# Testing the model on test data
test_predictions = dt_classifier.predict(pca_test_data_reduced)
conf_matrix_test = confusion_matrix(test_labels, test_predictions)
accuracy_test = accuracy_score(test_labels, test_predictions)
classification_rep_test = classification_report(test_labels, test_predictions)

# Displaying results for validation and test sets
print("Validation Set Results:")
print("Confusion Matrix:\n", conf_matrix_val)
print("Accuracy:", accuracy_val)
print("Classification Report:\n", classification_rep_val)

print("\nTest Set Results:")
print("Confusion Matrix:\n", conf_matrix_test)
print("Accuracy:", accuracy_test)
print("Classification Report:\n", classification_rep_test)
