import pandas as pd
from trim_n_filter import bp_filter, notch_filter, plot_signal, clipdata

# Load data from Excel file
signal_path = 'data.xlsx'
emg_signal = pd.read_excel(signal_path, usecols = [2]).values
channel_name = 'Raw EMG Data'

# Sampling Frequency of 1000 (1000 Samples per second)
sampling_frequency = 1e3


# Plot raw sEMG signal
plot_signal(emg_signal, sampling_frequency, channel_name)

emg_signal = emg_signal.reshape((emg_signal.size,))

# plot trimmed one (from original data)
clipdata = clipdata(emg_signal, sampling_frequency, True) 

# Band Stop Filter (BSF)
notched_signal = notch_filter(clipdata, sampling_frequency, True)

# Band Pass Filter (BPF)
# 60hz notch 50 HPF 150 LPF
filtered_signal_50_150 = bp_filter(notched_signal,emg_signal, 50, 150, sampling_frequency, True)


# export result data in excel
_to_excel = pd.DataFrame(filtered_signal_50_150)
_to_excel.to_excel("-100mv&filters.xlsx", index = False)

