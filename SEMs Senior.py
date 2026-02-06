
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

SEMfile1=r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Train_x.mat"
data_x=sio.loadmat(SEMfile1)
train_x=data_x['Train_x']

SEMfile1=r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Train_y.mat"
data_y=sio.loadmat(SEMfile1)
train_y=data_y['Train_y']
train_y=train_y.reshape((1000,))

SEMfile1=r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Test_x.mat"
data_x=sio.loadmat(SEMfile1)
test_x=data_x['Test_x']

SEMfile1=r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Test_y.mat"
data_y=sio.loadmat(SEMfile1)
test_y=data_y['Test_y']
test_y=test_y.reshape((400,))

x=list(range(0,300))
y1=train_x[1,]
plt.plot(x,y1)
plt.show()

y2=train_x[2,]
plt.plot(x,y2)
plt.show()

y3=train_x[3,]
plt.plot(x,y3)
plt.show()

y4=train_x[501,]
plt.plot(x,y4)
plt.show()

y5=train_x[502,]
plt.plot(x,y5)
plt.show()

y6=train_x[503,]
plt.plot(x,y6)
plt.show()

# Define the range of neighbors to try
k_range = range(1, 26)

# Initialize lists to store evaluation metric scores
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Train and evaluate the KNN classifier for each value of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x, train_y)
    predict_result = knn.predict(test_x)
    accuracy_scores.append(accuracy_score(test_y, predict_result))
    precision_scores.append(precision_score(test_y, predict_result, average='macro'))
    recall_scores.append(recall_score(test_y, predict_result, average='macro'))
    f1_scores.append(f1_score(test_y, predict_result, average='macro'))

# Print the evaluation metric scores for k=5
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
