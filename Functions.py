from Preprocessing import *
import librosa
import numpy as np
import random


def audio_to_spectrogram(audio, n_fft, hop_length, n_mels):
  """
  Converts an audio file to a spectrogram.

  Args:
    audio: The audio file to convert.
    n_fft: The number of samples per frame.
    hop_length: The number of samples between frames.
    n_mels: The number of mel bands.

  Returns:
    A spectrogram of the audio file.
  """
  S = librosa.feature.melspectrogram(y=audio, n_fft=n_fft,
                                     hop_length=hop_length, n_mels=n_mels)
  image = librosa.power_to_db(S, ref=np.max)
  mean = image.flatten().mean()
  std = image.flatten().std()
  eps=1e-8
  spec_norm = (image - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
  return spec_scaled

def convert_all_to_image(segments, n_fft, hop_length, n_mels):
  """
  Converts all audio segments in a list to spectrograms.

  Args:
    segments: The list of audio segments to convert.
    n_fft: The number of samples per frame.
    hop_length: The number of samples between frames.
    n_mels: The number of mel bands.

  Returns:
    A list of spectrograms of the audio segments.
  """
  spectrograms = []
  for segment in segments:
      spectrograms.append(audio_to_spectrogram(segment, n_fft, hop_length, n_mels))
  return np.array(spectrograms)

def augment_one_spectrogram(spectrogram, true_target, time_mask_length = 2, frequency_mask_width = 2):
  """
  Augments a spectrogram by randomly masking out time and frequency regions.

  Args:
    spectrogram: The spectrogram to augment.
    true_target: The true target of the spectrogram.
    time_mask_length: The length of the time mask.
    frequency_mask_width: The width of the frequency mask.

  Returns:
    The augmented spectrogram and the true target.
  """
  ts = np.random.randint(0, spectrogram.shape[1] - time_mask_length, size=3)
  new_spectrogram = np.copy(spectrogram)
  for t in ts:
    new_spectrogram[:, t:(t + time_mask_length)] = 0
  fs = np.random.randint(0, new_spectrogram.shape[0] - frequency_mask_width, size=3)
  for f in fs:
    new_spectrogram[f:(f + frequency_mask_width), :] = 0
  return new_spectrogram, true_target

def randomly_select_presence(all_spectrograms, targets):
  """
  Randomly selects a presence spectrogram from the dataset.

  Args:
    all_spectrograms: The list of all spectrograms in the dataset.
    targets: The list of all targets in the dataset.

  Returns:
    A randomly selected presence spectrogram.
  """
  presence_indices = np.where(targets =='1')[0]
  random_index = random.randint(0,len(presence_indices)-1)
  return all_spectrograms[presence_indices[random_index]]

def randomly_select_absence(all_spectrograms, targets):
  """
  Randomly selects an absence spectrogram from the dataset.

  Args:
    all_spectrograms: The list of all spectrograms in the dataset.
    targets: The list of all targets in the dataset.

  Returns:
    A randomly selected absence spectrogram.
  """
  absence_indices = np.where(targets =='0')[0]
  random_index = random.randint(0,len(absence_indices)-1)
  return all_spectrograms[absence_indices[random_index]]

def generate_new_presence_spectrograms(all_spectrograms, all_targets, quantity):
  """
  Generates new presence spectrograms by augmenting existing presence spectrograms.

  Args:
    all_spectrograms: The list of all spectrograms in the dataset.
    all_targets: The list of all targets in the dataset.
    quantity: The number of new spectrograms to generate.

  Returns:
    A list of new presence spectrograms.
  """
  new_spectrograms = []
  new_targets = []
  for i in range (0, quantity):
    presence_spectrogram = randomly_select_presence(all_spectrograms, all_targets)
    augmented_spectrogram, augmented_target = augment_one_spectrogram(presence_spectrogram,'1')
    new_spectrograms.append(augmented_spectrogram)
    new_targets.append(augmented_target)
  return np.asarray(new_spectrograms), np.asarray(new_targets)

def generate_new_absence_spectrograms(all_spectrograms, all_targets, quantity):
  """
  Generates new absence spectrograms by augmenting existing absence spectrograms.

  Args:
    all_spectrograms: The list of all spectrograms in the dataset.
    all_targets: The list of all targets in the dataset.
    quantity: The number of new spectrograms to generate.

  Returns:
    A list of new absence spectrograms.
  """
  new_spectrograms = []
  new_targets = []
  for i in range (0, quantity):
    absence_spectrogram = randomly_select_absence(all_spectrograms, all_targets)
    augmented_spectrogram, augmented_target = augment_one_spectrogram(absence_spectrogram,'0')
    new_spectrograms.append(augmented_spectrogram)
    new_targets.append(augmented_target)
  return np.asarray(new_spectrograms), np.asarray(new_targets)


def undersample(X, Y):
    """
    Undersample the data to balance the classes.

    Args:
        X (np.ndarray): The data.
        Y (np.ndarray): The labels.

    Returns:
        X_resampled (np.ndarray): The undersampled data.
        Y_resampled (np.ndarray): The undersampled labels.
    """
  
    np.random.seed(42)

    indexes_0 = np.where(Y == '0')[0]
    indexes_1 = np.where(Y == '1')[0]

    minority_len = min(len(indexes_0), len(indexes_1))
    majority_len = max(len(indexes_0), len(indexes_1))
    maj_class = '0' if len(indexes_0) > len(indexes_1) else '1'
    
    low_bound = int(minority_len * 0.9)
    high_bound = min(int(minority_len * 1.3), majority_len)
    new_majority_len = np.random.randint(low=low_bound, high=high_bound, size=1)[0]

    new_maj_indexes = np.random.choice(
        indexes_1 if maj_class == "1" else indexes_0, 
        size=new_majority_len, 
        replace=False
    )
    min_indexes = indexes_1 if maj_class == "0" else indexes_0
    new_index_sampled = np.concatenate((min_indexes, new_maj_indexes))
    np.random.shuffle(new_index_sampled)

    X_resampled = X[new_index_sampled, :, :]
    Y_resampled = Y[new_index_sampled]
    
    return X_resampled, Y_resampled
