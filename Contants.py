positive_class = ['1'] # which labels should be bundled together for the positive  class
negative_class = ['0'] # which labels should be bundled together for the negative  class

# Data hyper-parameters
lowpass_cutoff = 9000 # Cutt off for low pass filter
downsample_rate = 22000 # Frequency to downsample to
nyquist_rate = 11000 # Nyquist rate (half of sampling rate)
segment_duration = 4 # how long should a segment be


# Spectrogram hyper-parameters
n_fft = 1024 # Hann window length
hop_length = 256 # Sepctrogram hop size
n_mels = 128 # Spectrogram number of mells
f_min = 2500 # Spectrogram, minimum frequency for call
f_max = 9000 # Spectrogram, maximum frequency for call

species_folder = '.'
file_type = 'svl'
audio_extension = '.wav'

basedir_data = 'Saved_data/'

call_order = ['0','1']
