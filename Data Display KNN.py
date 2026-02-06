import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load data
train_x = sio.loadmat(r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Train_x.mat")['Train_x']
train_y = sio.loadmat(r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Train_y.mat")['Train_y'].ravel()
test_x = sio.loadmat(r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Test_x.mat")['Test_x']
test_y = sio.loadmat(r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Test_y.mat")['Test_y'].ravel()

# Plot train_x signals
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
for i, ax in enumerate(axs.flat):
    ax.plot(train_x[i+1])
    ax.set_title(f'Train_x[{i+1}]')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Signal')
plt.tight_layout()
plt.show()

# Evaluate KNN classifier for different number of neighbors
k_range = range(1, 26)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x, train_y)
    predict_result = knn.predict(test_x)
    accuracy_scores.append(accuracy_score(test_y, predict_result))
    precision_scores.append(precision_score(test_y, predict_result, average='macro'))
    recall_scores.append(recall_score(test_y, predict_result, average='macro'))
    f1_scores.append(f1_score(test_y, predict_result, average='macro'))

# Create subplots for evaluation metrics
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot accuracy score
axs[0,0].plot(k_range, accuracy_scores)
axs[0,0].set_title('Accuracy')
axs[0,0].set_xlabel('Number of Neighbors')
axs[0,0].set_ylabel('Score')

# Plot precision score
axs[0,1].plot(k_range, precision_scores)
axs[0,1].set_title('Precision')
axs[0,1].set_xlabel('Number of Neighbors')
axs[0,1].set_ylabel('Score')

# Plot recall score
axs[1,0].plot(k_range, recall_scores)
axs[1,0].set_title('Recall')
axs[1,0].set_xlabel('Number of Neighbors')
axs[1,0].set_ylabel('Score')

# Plot F1-score
axs[1,1].plot(k_range, f1_scores)
axs[1,1].set_title('F1-Score')
axs[1,1].set_xlabel('Number of Neighbors')
axs[1,1].set_ylabel('Score')

# Adjust layout and display plot
plt.tight_layout()
plt.show()


# Print evaluation metrics for k=5
k = 5
print(f'Evaluation metrics for k={k}:')
print(f'Accuracy: {accuracy_scores[k-1]:.3f}')
print(f'Precision: {precision_scores[k-1]:.3f}')
print(f'Recall: {recall_scores[k-1]:.3f}')
print(f'F1-Score: {f1_scores[k-1]:.3f}')

# Plot the evaluation metrics against the number of neighbors

plt.plot(k_range, accuracy_scores, label='Accuracy')
plt.plot(k_range, precision_scores, label='Precision')
plt.plot(k_range, recall_scores, label='Recall')
plt.plot(k_range, f1_scores, label='F1-Score')
plt.xlabel('Number of Neighbors')
plt.ylabel('Score')
plt.title('KNN Evaluation Metrics')
plt.legend()
plt.show()
