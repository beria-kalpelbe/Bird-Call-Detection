from Preprocessing import *
import librosa
import numpy as np
import random
import tensorflow as tf
from IPython.display import Audio
import resampy
from Functions import *
from Contants import *
import zipfile

if colab:
    from google.colab import drive
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # Google Authentication
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

    # Download data files
    downloaded = drive.CreateFile({'id':"1seDpWl9c28V-kCQxjhkuaws7i8d-_R31"})
    downloaded.GetContentFile('Data.zip')

print("Data extraction..................")
# Extract files to temporary location in Google Drive
with zipfile.ZipFile('Data.zip', 'r') as zip_file:
    zip_file.extractall()


print('Preprocessing..............')
# This takes about 1 minute
pre_pro = Preprocessing(species_folder, lowpass_cutoff,
                downsample_rate, nyquist_rate,
                segment_duration,
                positive_class, negative_class,n_fft,
                hop_length, n_mels, f_min, f_max, file_type,
                audio_extension)

X, Y = pre_pro.create_dataset(False)

x_file = basedir_data+'x.npy'
y_file = basedir_data+'y.npy'

print('\n\n Shapes of data pre-processed\n')

print('Shape of X: ', str(X.shape))
print('Shape of Y: ', str(Y.shape))


print("Converting all audio sequences in images......")
X_S = convert_all_to_image(X, n_fft, hop_length, n_mels)
np.save(basedir_data+'x_s.npy', X_S)

unique, counts = np.unique(Y, return_counts=True)
original_distribution = dict(zip(unique, counts))
print('Data distribution:',original_distribution)


print('Data augmentation is running............')
# Augmenting presences
new_presence, new_targets = generate_new_presence_spectrograms(X_S, Y, 500)
print('New presence shape: ',str(new_presence.shape))
print('New target shape: ',str(new_targets.shape))

X_positive = np.concatenate([X_S[np.where(Y =='1')], new_presence])
X_positive = np.asarray(X_positive)
print('X_positive shape: ',str(X_positive.shape))

Y_positive = np.concatenate([Y[np.where(Y =='1')], new_targets])
Y_positive = np.asarray(Y_positive)
print('Y_positive shape: ',str(Y_positive.shape))

# Augmenting absences
new_absence, new_targets = generate_new_absence_spectrograms(X_S, Y,500)
print('New absence shape: ',str(new_absence.shape))
print('New target shape: ',str(new_targets.shape))

X_negatives = np.concatenate([X_S[np.where(Y =='0')], new_absence])
X_negatives = np.asarray(X_negatives)
print('X_negative shape: ',str(X_negatives.shape))

Y_negatives = np.concatenate([Y[np.where(Y =='0')], new_targets])
Y_negatives = np.asarray(Y_negatives)
print('Y_negative shape: ',str(Y_negatives.shape))


X_dataset = np.concatenate([X_positive, X_negatives])
Y_dataset = np.concatenate([Y_positive, Y_negatives])

print('Shape of X dataset: ',str(X_dataset.shape))
print('Shape of Y dataset: ',str(Y_dataset.shape))

np.save(basedir_data+'x_dataset.npy', X_dataset)
np.save(basedir_data+'y_dataset.npy', Y_dataset)

unique, counts = np.unique(Y_dataset, return_counts=True)
original_distribution = dict(zip(unique, counts))
print('Data distribution:',original_distribution)


print('Preprocessing done........!!!')