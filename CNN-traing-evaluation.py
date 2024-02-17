import librosa
import numpy as np
import random
import tensorflow as tf
from IPython.display import Audio
import matplotlib.pyplot as plt
from Functions import *
from Contants import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D, Input
from tensorflow.keras.models import Sequential
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc



print("Data loading...........")
with open(basedir_data+'x_dataset.npy', 'rb') as f:
    X_dataset = np.load(f)
with open(basedir_data+'y_dataset.npy', 'rb') as f:
    Y_dataset = np.load(f)

X_dataset = np.expand_dims(X_dataset, axis=-1)

print('X_dataset.shape: ', str(X_dataset.shape))
print('Y_dataset.shape: ',str(Y_dataset.shape))

# Converting categorical string labels ('gibbons' and 'no-gibbon) to 0s and 1s
for index, call_type in enumerate(call_order):
    Y_dataset = np.where(Y_dataset == call_type, index, Y_dataset)

Y_dataset = to_categorical(Y_dataset,
                             num_classes = 2)

print('X_dataset.shape: ', str(X_dataset.shape))
print('Y_dataset.shape: ',str(Y_dataset.shape))

# Model specification
INPUT_SHAPE = (X_dataset.shape[1], X_dataset.shape[2], X_dataset.shape[3])
inputs = Input(shape=INPUT_SHAPE)
x = inputs
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(2, activation='softmax')(x)
outputs = x
model1 = Model(inputs, outputs)
print(model1.summary())

print("Train - Validation set splitting ...............")
X_train, X_val, y_train, y_val = train_test_split(X_dataset, Y_dataset, test_size=0.2, random_state=42)

print('Training ..............')
model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history1 = model1.fit(X_train,
          y_train ,
          epochs=10, validation_data=(X_val, y_val),
          batch_size=32)

np.save(basedir_data+"history_CNN_good.npy", history1.history)

"""
plotting the loss and accuracy on train and validation set
"""
fig, axes = plt.subplots(1,2, figsize=(18,6))

axes[0].plot(history1.history['accuracy'])
axes[0].plot(history1.history['val_accuracy'])
axes[0].set_title('Accuracy for CNN model')
axes[0].set_ylabel('accuracy')
axes[0].set_xlabel('epoch')
axes[0].legend(['train', 'test'], loc='upper left')
axes[0].grid(True)
# summarize history for loss
axes[1].plot(history1.history['loss'])
axes[1].plot(history1.history['val_loss'])
axes[1].set_title('Loss for CNN model')
axes[1].set_ylabel('loss')
axes[1].set_xlabel('epoch')
axes[1].legend(['train', 'test'], loc='upper right')
axes[1].grid(True)
plt.savefig("Plots/CNN_train_results.pdf", format='pdf')
plt.show()

"""
Testing the model
"""
print("Testing .................")
with open(basedir_data+'X_test_S.npy', 'rb') as f:
    X_test_S = np.load(f)
with open(basedir_data+'Y_test.npy', 'rb') as f:
    Y_test = np.load(f)
    
X_test_dataset = np.expand_dims(X_test_S, axis=-1)
for index, call_type in enumerate(call_order):
    Y_test_dataset = np.where(Y_test == call_type, index, Y_test)
Y_test_dataset = Y_test_dataset.astype(int)

print(X_test_dataset.shape)
print(Y_test_dataset.shape)
print("|________prediction.....")
y_pred_probs = model1.predict(X_test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

"""
Confusion matrix plotting
"""
conf_mat = confusion_matrix(Y_test_dataset, y_pred)
# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No-bird Call', 'Bird Call'], yticklabels=['No-bird Call', 'Bird Call'])
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.title('CNN ClassiConfusion Matrix')
plt.savefig('Plots/CNN_confusion_matrix.pdf', format='pdf')
plt.show()

"""
ROC curve plotting
"""
fpr, tpr, thresholds = roc_curve(Y_test_dataset, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)
np.save(basedir_data+'CNN_ROC_data.npy', {'fpr': fpr, 'tpr':tpr})
# Plot the ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
#plt.title('CNN Classifier ROC Curve')
plt.legend(loc='lower right')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('Plots/CNN_ROC.pdf', format='pdf')
plt.show()


"""
Performance metrics
"""
Y = to_categorical(Y_test_dataset, num_classes = 2)
test_loss, test_acc = model1.evaluate(X_test_dataset,  Y, verbose=2)


# Calculate Precision
precision = precision_score(Y_test_dataset, y_pred)

# Calculate Recall
recall = recall_score(Y_test_dataset, y_pred)

# Calculate F1 Score
f1 = f1_score(Y_test_dataset, y_pred)

print(f'Test accuracy: {test_acc:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')