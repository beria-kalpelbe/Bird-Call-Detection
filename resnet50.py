import numpy as np
import matplotlib.pyplot as plt
from Functions import *
from Contants import *
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

with open(basedir_data +'x_dataset.npy', 'rb') as f:
    X_dataset = np.load(f)
with open(basedir_data +'y_dataset.npy', 'rb') as f:
    Y_dataset = np.load(f)
    
X_dataset = np.expand_dims(X_dataset, axis=-1)

print(X_dataset.shape)
print(Y_dataset.shape)

"""
Model Specification
"""
INPUT_SHAPE = (X_dataset.shape[1], X_dataset.shape[2], 3)
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
base_model.trainable = False
inputs = Input(shape=INPUT_SHAPE)

x = base_model(inputs, training=False)

x = Flatten()(x) # flatten everything from the current model
outputs = Dense(2, activation='softmax')(x) # Add a softmax layer
model3 = Model(inputs, outputs) # bring everything together

model3.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-5),
              metrics=['accuracy'])
model3.summary()

# Convert 1D image in to 3D image
X_dataset = np.concatenate((0.3*X_dataset, 0.59*X_dataset, 0.11*X_dataset), axis=-1)
print(X_dataset.shape)

for index, call_type in enumerate(call_order):
    Y_dataset = np.where(Y_dataset == call_type, index, Y_dataset)
Y_dataset = to_categorical(Y_dataset,
                             num_classes = 2)


X_train, X_val, y_train, y_val = train_test_split(X_dataset, Y_dataset, test_size=0.2, random_state=42)

del X_dataset

"""
Model Training
"""
history3 = model3.fit(X_train,
          y_train ,
          epochs=5,
          batch_size=15, validation_data=(X_val, y_val))

np.save(basedir_data+"history_ResNet_good.npy", history3.history)

"""
Plotting loss and accuracy on train and test set
"""
fig, axes = plt.subplots(1,2, figsize=(18,6))
axes[0].plot(history3.history['accuracy'])
axes[0].plot(history3.history['val_accuracy'])
axes[0].set_title('Accuracy for Transfer learning model')
axes[0].set_ylabel('accuracy')
axes[0].set_xlabel('epoch')
axes[0].legend(['train', 'test'], loc='upper left')
axes[0].grid(True)
# summarize history for loss
axes[1].plot(history3.history['loss'])
axes[1].plot(history3.history['val_loss'])
axes[1].set_title('Loss for Transfer learning model')
axes[1].set_ylabel('loss')
axes[1].set_xlabel('epoch')
axes[1].legend(['train', 'test'], loc='upper right')
axes[1].grid(True)
plt.savefig("Plots/ResNet_train_results.pdf", format='pdf')
plt.show()

"""
Model testing
"""
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

X_test_dataset = np.concatenate((0.3*X_test_dataset, 0.59*X_test_dataset, 0.11*X_test_dataset), axis=-1)
print(X_test_dataset.shape)

y_pred_probs = model3.predict(X_test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
np.save(basedir_data+'y_pred_ResNet50.npy', y_pred)
"""
Confusion matrix plotting
"""
conf_mat = confusion_matrix(Y_test_dataset, y_pred)
# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['No-bird Call', 'Bird Call'], yticklabels=['No-bird Call', 'Bird Call'])
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.title('CNN ClassiConfusion Matrix')
plt.savefig('Plots/ResNet_confusion_matrix.pdf', format='pdf')
plt.show()

"""
Plotting ROC curve
"""
fpr, tpr, thresholds = roc_curve(Y_test_dataset, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)
np.save(basedir_data+'ResNet_ROC_data.npy', {'fpr': fpr, 'tpr':tpr})
# Plot the ROC curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve of Transfer learning Classifier')
plt.legend(loc='lower right')
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('Plots/ResNet_ROC.pdf', format='pdf')
plt.show()


"""
Performance metrics
"""
Y = to_categorical(Y_test_dataset, num_classes = 2)
test_loss, test_acc = model3.evaluate(X_test_dataset,  Y, verbose=2)


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