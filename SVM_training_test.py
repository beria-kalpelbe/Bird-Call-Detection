import numpy as np
import matplotlib.pyplot as plt
from Functions import *
from Contants import *
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix



with open(basedir_data+'x_dataset.npy', 'rb') as f:
    X_dataset = np.load(f)
with open(basedir_data+'y_dataset.npy', 'rb') as f:
    Y_dataset = np.load(f)

print(X_dataset.shape)
print(Y_dataset.shape)

# Converting categorical string labels ('gibbons' and 'no-gibbon) to 0s and 1s
for index, call_type in enumerate(call_order):
    Y_dataset = np.where(Y_dataset == call_type, index, Y_dataset)
Y_dataset = Y_dataset.astype(int)

print(X_dataset.shape)
print(Y_dataset.shape)

"""
Model training
"""
# Standardize the feature values (mean=0, variance=1)
scaler = StandardScaler()
new_dataset = np.zeros((X_dataset.shape[0], X_dataset.shape[1]*X_dataset.shape[2]))
for i in range(len(new_dataset)):
  a = scaler.fit_transform(X_dataset[i])
  new_dataset[i] = a.flatten()

# Create an SVM classifier
clf = svm.SVC(kernel='linear', probability=True)

# Train the classifier
clf.fit(new_dataset, Y_dataset)

"""
Model testing
"""
with open(basedir_data+'X_test_S.npy', 'rb') as f:
    X_test_dataset = np.load(f)
with open(basedir_data+'Y_test.npy', 'rb') as f:
    Y_test_dataset = np.load(f)

new_test_dataset = np.zeros((X_test_dataset.shape[0], X_test_dataset.shape[1]*X_test_dataset.shape[2]))
for i in range(len(new_test_dataset)):
  a = scaler.transform(X_test_dataset[i])
  new_test_dataset[i] = a.flatten()
Y_test_dataset = Y_test_dataset.astype(int)

# Make predictions on the test set
y_pred = clf.predict(new_test_dataset)

y_prob = clf.predict_proba(new_test_dataset)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test_dataset, y_prob)
roc_auc = auc(fpr, tpr)

np.save(basedir_data+'SVM_ROC_data.npy', {'fpr': fpr, 'tpr':tpr})

"""
Plotting ROC curve
"""
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for SVM Classifier')
plt.legend(loc='lower right')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('Plots/SVM_ROC.pdf', format='pdf')
plt.show()

"""
Plotting Confusion Matrix
"""
conf_mat = confusion_matrix(Y_test_dataset, y_pred)
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No-bird Call', 'Bird Call'], yticklabels=['No-bird Call', 'Bird Call'])
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.title('CNN ClassiConfusion Matrix')
plt.savefig(basedir_data+'SVM_confusion_matrix.pdf', format='pdf')
plt.show()

"""
Performances metrics
"""
# Evaluate the accuracy
accuracy = accuracy_score(Y_test_dataset, y_pred)

# Calculate Precision
precision = precision_score(Y_test_dataset, y_pred)

# Calculate Recall
recall = recall_score(Y_test_dataset, y_pred)

# Calculate F1 Score
f1 = f1_score(Y_test_dataset, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')