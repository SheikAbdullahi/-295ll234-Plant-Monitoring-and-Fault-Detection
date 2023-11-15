from scipy.io import loadmat
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load the MATLAB file
data = loadmat('/Users/sheik/Downloads/AllData_ExTxS.mat')

# Display the keys to understand the structure of the loaded data
data.keys()

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

# Plotting the first two PCA components of the testing data
plt.figure(figsize=(12, 8))
plt.scatter(pca_test_data[:, 0], pca_test_data[:, 1], alpha=0.7)
plt.title("First Two PCA Components of Testing Data")
plt.xlabel("First Component (PCA1)")
plt.ylabel("Second Component (PCA2)")
plt.grid(True)
plt.show()

# Calculating the Q-statistic (Squared Prediction Error) for each testing data point
# Q-statistic is the sum of the squared differences between the original data and the PCA-reconstructed data

# Reconstructing the data from the PCA components
reconstructed_data = np.dot(pca_test_data, transformation_matrix)

# Calculating the squared differences
squared_differences = np.square(act_1_test_unfolded - reconstructed_data)

# Summing the squared differences for each data point to get the Q-statistic
q_statistic = np.sum(squared_differences, axis=1)

# Plotting the Q-statistic
plt.figure(figsize=(12, 8))
plt.plot(q_statistic, marker='o', linestyle='-', markersize=5)
plt.title("Q-statistic for Each Testing Data Point")
plt.xlabel("Data Point Index")
plt.ylabel("Q-statistic Value")
plt.grid(True)


# Calculating the T2-statistic (Hotelling's T-square) for each testing data point
# T2-statistic is the sum of the squared scaled scores for each PCA component
# Each score is typically scaled by the inverse of its corresponding eigenvalue

# Eigenvalues of the PCA components
eigenvalues = pca.explained_variance_

# Scaling the PCA scores by the inverse of the eigenvalues
scaled_scores = pca_test_data / np.sqrt(eigenvalues)

# Calculating the T2-statistic for each data point (sum of squared scaled scores)
t2_statistic = np.sum(np.square(scaled_scores), axis=1)

# Plotting the T2-statistic
plt.figure(figsize=(12, 8))
plt.plot(t2_statistic, marker='o', linestyle='-', markersize=5)
plt.title("T2-statistic for Each Testing Data Point")
plt.xlabel("Data Point Index")
plt.ylabel("T2-statistic Value")
plt.grid(True)
plt.show()
