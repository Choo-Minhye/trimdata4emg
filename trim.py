from pickle import NONE
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clipdata(x, samplerate, plot=False):
    range_ = 2000
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    x_clip = [100 if i > 60 else i for i in x]
    print(len(x_clip))
    i = 0
    while i < len(x_clip) :
        j = 0
        if x_clip[i] == 100 :
            if i < int(range_/2) + 1 : 
                for j in range(i, i + range_) : 
                    x_clip[j] = -100 #mv
                fs =i/1000
                ss = (i + range_) /1000 
            else : 
                for j in range(i-int(range_/2),i+int(range_/2) ) :
                    x_clip[j] = -100 #mv
        
                fs =(i-range_/2) / 1000 
                ss = (i + range_/2) /1000 
        
            print("no usual 구간 : " + str(fs) + "초 - " + str(ss) + "초" )
            
        i = i+1



    if plot:
        plt.plot(t, x)
        plt.plot(t, x_clip, 'k')
        plt.autoscale(tight=True)
        plt.title('clip(60mv)')
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_clip


def notch_filter(x, samplerate, plot=False):
    x = x - np.mean(x)

    high_cutoff_notch = 59 / (samplerate /2) # nyquist frequency = sampling rate / 2 
    low_cutoff_notch = 61 / (samplerate /2)

    # Band Stop Filter (BSF) or Band Reject Filter
    [b, a] = signal.butter(4, [high_cutoff_notch, low_cutoff_notch], btype='stop')

    x_filt = signal.filtfilt(b, a, x.T)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt.T, 'k')
        plt.autoscale(tight=True)
        plt.title('notch filter(60mv)')
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt

# 60hz notch 50 HPF 150 LPF
def bp_filter(x,y, high_pass, low_pass, samplerate, plot=False):
    # x = x - np.mean(x)

    low_cutoff_bp = high_pass / (samplerate)
    high_cutoff_bp = low_pass / (samplerate)

    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')

    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, y)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.title('Band pass Filter')
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def plot_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()
    
    
    
    
    
    
# import logging
# import numpy as np
# import pandas as pd

# logging.basicConfig(datefmt='%H:%M:%S',
#                     stream=sys.stdout, level=logging.DEBUG,
#                     format='%(asctime)s %(message)s')

# # Distance away from the FBEWMA that data should be removed.
# DELTA = 1

# # clip data above this value:
# HIGH_CLIP = 91

# # clip data below this value:
# LOW_CLIP = -91

# # random values above this trigger a spike:
# RAND_HIGH = 0.98

# # random values below this trigger a negative spike:
# RAND_LOW = 0.02

# # How many samples to run the FBEWMA over.
# SPAN = 1000

# # spike amplitude
# SPIKE = 90


# def clip_data(unclipped, high_clip, low_clip):
#     ''' Clip unclipped between high_clip and low_clip. 
#     unclipped contains a single column of unclipped data.'''
    
#     # convert to np.array to access the np.where method
#     np_unclipped = np.array(unclipped)
#     # clip data above HIGH_CLIP or below LOW_CLIP
#     cond_high_clip = (np_unclipped > HIGH_CLIP) | (np_unclipped < LOW_CLIP)
#     np_clipped = np.where(cond_high_clip, np.nan, np_unclipped)
#     return np_clipped.tolist()


# def create_sample_data():
#     ''' Create sine wave, amplitude +/-2 with random spikes. '''
#     x = np.linspace(0, 2*np.pi, 1000)
#     y = 2 * np.sin(x)
#     df = pd.DataFrame(list(zip(x,y)), columns=['x', 'y'])
#     df['rand'] = np.random.random_sample(len(x),)
#     # create random positive and negative spikes
#     cond_spike_high = (df['rand'] > RAND_HIGH)
#     df['spike_high'] = np.where(cond_spike_high, SPIKE, 0)
#     cond_spike_low = (df['rand'] < RAND_LOW)
#     df['spike_low'] = np.where(cond_spike_low, -SPIKE, 0)
#     df['y_spikey'] = df['y'] + df['spike_high'] + df['spike_low']
#     return df


# # def ewma_fb(df_column, span):
# #     ''' Apply forwards, backwards exponential weighted moving average (EWMA) to df_column. '''
# #     # Forwards EWMA.
# #     fwd = pd.Series.ewm(df_column, span=span).mean()
#     # Backwards EWMA.
#     bwd = pd.Series.ewm(df_column[::-1],span=10).mean()
#     # Add and take the mean of the forwards and backwards EWMA.
#     stacked_ewma = np.vstack(( fwd, bwd[::-1] ))
#     fb_ewma = np.mean(stacked_ewma, axis=0)
#     return fb_ewma
    
    
# def remove_outliers(spikey, fbewma, delta):
#     ''' Remove data from df_spikey that is > delta from fbewma. '''
#     np_spikey = np.array(spikey)
#     np_fbewma = np.array(fbewma)
#     cond_delta = (np.abs(np_spikey-np_fbewma) > delta)
#     np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
#     return np_remove_outliers

    
# def main():
#     df = create_sample_data()

#     df['y_clipped'] = clip_data(df['y_spikey'].tolist(), HIGH_CLIP, LOW_CLIP)
#     df['y_ewma_fb'] = ewma_fb(df['y_clipped'], SPAN)
#     df['y_remove_outliers'] = remove_outliers(df['y_clipped'].tolist(), df['y_ewma_fb'].tolist(), DELTA)
#     df['y_interpolated'] = df['y_remove_outliers'].interpolate()
    
#     ax = df.plot(x='x', y='y_spikey', color='blue', alpha=0.5)
#     ax2 = df.plot(x='x', y='y_interpolated', color='black', ax=ax)
    
# main()
