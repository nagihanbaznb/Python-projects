# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 01:17:03 2022

@author: 181805052-Elize Hamitoğlu
171805024-Nagihan Baz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import pandas as pd
#from scipy import fftpack
import scipy
from scipy.io import loadmat

rawsignals = loadmat(r'C:\Users\USER\Desktop\s1w.mat')
#s1b = loadmat(r'C:\Users\elx13\Desktop\s1b.mat')
#s1w = loadmat(r'C:\Users\elx13\Desktop\s1w.mat')
#ecg = rawsignals["PPG"].to_numpy()
# EEG signal for channel 0
eeg = rawsignals["data"]["EEG"][0,0][0:,0]
eog = rawsignals["data"]["EOG"][0,0][0:,0]
# sampling frequency 360Hz
fs = 125
# generating time axis values
lower = 17000
upper = 18000

time = np.arange(eeg.size) / fs
time1= np.arange(eog.size) / fs

winsize = fs * 5
winhop = fs
i = 0


def on_press(event):
    global i
    print('press', event.key)
    sys.stdout.flush()

    lower = i
    upper = i + winsize

    ax1.cla()
    ax1.plot(time, eeg, 'g')
    #ax1.plot(time, winhighlight, 'r')
    ax1.plot(time[lower:upper], eeg[lower:upper], 'r')
    ax1.grid()
    ax1.set_title('Raw Signal of EEG')

    ###
    # first
    x = eeg[lower:upper]
    peaks, properties = find_peaks(x, height=(None, 0.6), prominence=(-1))
    ax2.cla()
    ax2.plot(x, 'black')
    ax2.plot(peaks, x[peaks], 'o')
    ax2.set_title('Pulse Wave Systolic Peak of EEG')
    

    # compute the power spectrum of the filter kernel
    #filtpow = np.abs(scipy.fftpack.fft(eeg))**2
    # compute the frequencies vector and remove negative frequencies
    #hz      = np.linspace(0,fs/2,int(np.floor(len(x)/2)+1))
    #filtpow = filtpow[0:len(hz)]
    """
    ax3.cla()
    ax3.plot(hz,filtpow,'ks-',label='Actual')
    ax3.grid()
    """
    
    ax3.cla()
    ax3.plot(time1, eog, 'g')
    # ax1.plot(time, winhighlight, 'r')
    ax3.plot(time1[lower:upper], eog[lower:upper], 'r')
    ax3.grid()
    ax3.set_title('Raw Signal of EOG')
    
    x1 = eog[lower:upper]
    peaks, properties = find_peaks(x1, height=(None, 0.4), prominence=[0.1])
    ax4.cla()
    ax4.plot(x1, 'black')
    ax4.plot(peaks, x1[peaks], 'o')
    ax4.set_title('Pulse Wave Systolic Peak of EOG')
    
    
    # compute the power spectrum of the filter kernel
    #filtpow = np.abs(scipy.fftpack.fft(x))**2
    # compute the frequencies vector and remove negative frequencies
    #hz      = np.linspace(0,fs/2,int(np.floor(len(x)/2)+1))
    #filtpow = filtpow[0:len(hz)]
    """
    ax4.cla()
    ax4.plot(hz,filtpow,'ks-',label='Actual')
    ax4.grid()

    frex,tf,pwr = scipy.signal.spectrogram(eeg,fs)
    ax5.pcolormesh(tf,frex,pwr,shading='auto')
    #ax4.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
    
    frex,tf,pwr = scipy.signal.spectrogram(x,fs)
    ax6.pcolormesh(tf,frex,pwr,shading='auto')
    #ax5.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
   """ 


    if event.key == 'right':
        i = i + winhop
        fig.canvas.draw()
    elif event.key == 'left':
        i = i - winhop
        fig.canvas.draw()


fig = plt.figure()

ax1 = fig.add_subplot(321)
ax1.plot(time, eeg, 'g')
ax1.grid()
ax1.set_title('Raw Signal of EEG')

ax2 = fig.add_subplot(322)
ax2.grid()
ax2.set_title('Pulse Wave Systolic Peak of EEG')


ax3 = fig.add_subplot(323)
ax3.plot(time1, eog, 'g')
ax3.grid()
ax3.set_title('Raw Signal of EOG')

ax4 = fig.add_subplot(324)
ax4.grid()
ax4.set_title('Pulse Wave Systolic Peak of EOG')
"""
ax5 = fig.add_subplot(325)
ax5.grid()

ax6 = fig.add_subplot(326)
ax6.grid()
"""
fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()