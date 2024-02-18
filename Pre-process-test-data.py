from Preprocessing import *
import librosa
import numpy as np
import random
import tensorflow as tf
from IPython.display import Audio
import resampy
from Functions import *
from Contants import *

pre_pro_test = Preprocessing(species_folder, lowpass_cutoff,
                downsample_rate, nyquist_rate,
                segment_duration,
                positive_class, negative_class,n_fft,
                hop_length, n_mels, f_min, f_max, file_type,
                audio_extension)
pre_pro_test.training_files = './DataFiles/TestingFiles.txt'
X_test, Y_test = pre_pro_test.create_dataset(False)


# indexes = range(X_test.shape[0])
# chosen_indexes = np.random.choice(indexes, size=500, replace=False)
# X_test = X_test[chosen_indexes,:]
# Y_test = Y_test[chosen_indexes]
print(X_test.shape)
print(Y_test.shape)

X_test_S = convert_all_to_image(X_test, n_fft, hop_length, n_mels)
del X_test
np.save(basedir_data+'X_test_S.npy', X_test_S)
del X_test_S
np.save(basedir_data+'Y_test.npy', Y_test)

unique, counts = np.unique(Y_test, return_counts=True)
original_distribution = dict(zip(unique, counts))
print('Data distribution:',original_distribution)

print('Preprocessing done ............!!!')

