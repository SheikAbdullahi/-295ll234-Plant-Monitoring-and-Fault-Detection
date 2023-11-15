from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

# Load the MATLAB file
data = loadmat('/Users/sheik/Downloads/AllData_ExTxS.mat')

# Display the keys to understand the structure of the loaded data
data_keys = data.keys()
data_keys

# Extracting the 'act_1_train' array from the loaded data
act_1_train = data['act_1_train']

# Unfolding the 'act_1_train' array to a 2D array
act_1_train_unfolded = act_1_train.reshape(act_1_train.shape[0], -1)

# Applying PCA to the unfolded training data
pca = PCA()
pca.fit(act_1_train_unfolded)

# The transformation matrix (P) is the components_ attribute of the PCA object
transformation_matrix = pca.components_

# Extracting the 'act_1_test' array from the loaded data
act_1_test = data['act_1_test']

# Unfolding the 'act_1_test' array to a 2D array
act_1_test_unfolded = act_1_test.reshape(act_1_test.shape[0], -1)

# Projecting the unfolded testing data onto the PCA model using the transformation matrix
pca_test_data = np.dot(act_1_test_unfolded, transformation_matrix.T)

# Displaying the shape of the transformed test data 
pca_test_data.shape

# Explained variance ratio of each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance ratio
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plotting the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_explained_variance, marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()

# Selecting the number of components that explain a significant amount of variance (e.g., 95%)
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
n_components_95, cumulative_explained_variance[n_components_95 - 1]

# Reducing the training and testing data to the first 11 principal components
pca_train_data = pca.transform(act_1_train_unfolded)[:, :n_components_95]
pca_test_data_reduced = pca_test_data[:, :n_components_95]

# Assuming the labels for the training data are available in the MATLAB data file
# Extracting labels for training data (assuming they are stored under a key like 'act_1_train_labels')
train_labels = data.get('act_1_train_labels', np.zeros(act_1_train.shape[0]))

# Splitting the PCA-transformed training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(pca_train_data, train_labels, test_size=0.3, random_state=42)

# Training a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predicting on the validation set
val_predictions = dt_classifier.predict(X_val)

# Evaluating the model on the validation set
conf_matrix = confusion_matrix(y_val, val_predictions)
accuracy = accuracy_score(y_val, val_predictions)
classification_rep = classification_report(y_val, val_predictions)

# Displaying results
conf_matrix, accuracy, classification_rep


# Extracting labels for testing data (assuming they are stored under a key like 'act_1_test_labels')
test_labels = data.get('act_1_test_labels', np.zeros(act_1_test.shape[0]))

# Predicting on the test set using the trained Decision Tree Classifier
test_predictions = dt_classifier.predict(pca_test_data_reduced)

# Evaluating the model on the test set
test_conf_matrix = confusion_matrix(test_labels, test_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)
test_classification_rep = classification_report(test_labels, test_predictions)

# Displaying results for the test set
test_conf_matrix, test_accuracy, test_classification_rep

