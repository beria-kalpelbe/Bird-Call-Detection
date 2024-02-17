positive_class = ['1'] # which labels should be bundled together for the positive  class
negative_class = ['0'] # which labels should be bundled together for the negative  class

# Data hyper-parameters
lowpass_cutoff = 2000 # Cutt off for low pass filter
downsample_rate = 16000 # Frequency to downsample to
nyquist_rate = 8000 # Nyquist rate (half of sampling rate)
segment_duration = 4 # how long should a segment be


# Spectrogram hyper-parameters
n_fft = 1024 # Hann window length
hop_length = 256 # Sepctrogram hop size
n_mels = 128 # Spectrogram number of mells
f_min = 2000 # Spectrogram, minimum frequency for call
f_max = 7000 # Spectrogram, maximum frequency for call

# Don't change these
species_folder = '.' # Should contain /Audio and /Annotations, don't change this
file_type = 'svl' # don't change this
audio_extension = '.wav' # don't change this

basedir_data = 'Saved data/'

call_order = ['0','1']

colab = False