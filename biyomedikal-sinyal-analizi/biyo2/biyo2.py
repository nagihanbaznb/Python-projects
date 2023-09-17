# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 00:53:46 2022

@author:181805052-Elize Hamitoğlu
171805024-Nagihan Baz
"""
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import numpy as np

s1b = loadmat(r'C:\Users\USER\Desktop\s1b.mat')
s1w = loadmat(r'C:\Users\USER\Desktop\s1w.mat')

s1w_eeg_channel_label=s1w["data"]["channel_labels"][0,0][0,0]
fs = 125

#time = np.arange(s1w_eeg_channel_label.size) / fs
winsize = fs * 5
winhop = fs
i = 0

s1w_eeg_ch0=s1w["data"]["EEG"][0,0][0:,0]
s1w_eeg_ch1=s1w["data"]["EEG"][0,0][0:,1]
s1w_eeg_ch2=s1w["data"]["EEG"][0,0][0:,2]
s1w_eeg_ch3=s1w["data"]["EEG"][0,0][0:,3]

s1w_eog_ch0=s1w["data"]["EOG"][0,0][0:,0]
s1w_eog_ch1=s1w["data"]["EOG"][0,0][0:,1]
s1w_eog_ch2=s1w["data"]["EOG"][0,0][0:,2]

def on_press(event):
    global i
    print('press', event.key)
    sys.stdout.flush()

    lower = i
    upper = i + winsize
    
    ax2.cla()
    ax2.plot(s1w_eeg_ch0[lower:upper], alpha=0.8,label="ch0")
    ax2.plot(s1w_eeg_ch1[lower:upper], alpha=0.8,label="ch1")
    ax2.plot(s1w_eeg_ch2[lower:upper], alpha=0.8, label="ch2")
    ax2.plot(s1w_eeg_ch3[lower:upper], alpha=0.8,label="ch3")
    ax2.legend(loc='upper right', shadow=True)
    
    ax3.cla()
    ax3.plot(s1w_eog_ch0[lower:upper], alpha=0.8, label="ch0")
    ax3.plot(s1w_eog_ch1[lower:upper], alpha=0.8,label="ch1")
    ax3.plot(s1w_eog_ch2[lower:upper], alpha=0.8, label="ch2")
    ax3.legend(loc='upper right',shadow=True)
    
    if event.key == 'right':
        i = i + winhop
        fig.canvas.draw()
    elif event.key == 'left':
        i = i - winhop
        fig.canvas.draw()


fig = plt.figure()
ax0=fig.add_subplot(221)
ax0.plot(s1w_eeg_ch0, alpha=0.8, label="ch0")
ax0.plot(s1w_eeg_ch1, alpha=0.8,label="ch1")
ax0.plot(s1w_eeg_ch2, alpha=0.8, label="ch2")
ax0.plot(s1w_eeg_ch3, alpha=0.8,label="ch3")
ax0.legend( loc='upper right', shadow=True)
ax0.set_title(' EEG ')

ax1 = fig.add_subplot(222)
ax1.plot(s1w_eog_ch0, alpha=0.8, label="ch0")
ax1.plot(s1w_eog_ch1, alpha=0.8,label="ch1")
ax1.plot(s1w_eog_ch2, alpha=0.8, label="ch2")
ax1.legend( loc='upper right', shadow=True)
ax1.set_title(' EOG ')

ax2 = fig.add_subplot(223)
ax2.grid()

ax3 = fig.add_subplot(224)
ax3.grid()

fig.canvas.mpl_connect('key_press_event',on_press)

plt.show()

s1w_eeg_trig=s1w["data"]["trigger"][0,0][0:,0]
s1w_eeg_head=s1w["data"]["header"][0,0][0:,0]
s1w_eeg_FIR_resample=s1w["data"]["FIR_resample"][0,0][0]
s1w_eeg_subjective_report=s1w["data"]["subjective_report"][0,0]

