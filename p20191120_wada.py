#MNE tutorial

#Import modules
import os
import numpy as np
import mne
import re
import complexity_entropy as ce

#Import specific smodules for filtering
from numpy.fft import fft, fftfreq
from scipy import signal
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter
import matplotlib.pyplot as plt

### PUT ALL PARAMETERS HERE ###


### ### ### ### ### ### ### ###


### PUT FUNCTIONS HERE OR BETTER, IN SEPARATE FILE ###

### ### ### ### ### ### ### ### ### ### ### ### ### ###


#Path(s) to data #UPDATE TO READ ALL SUBFOLDERS IN A FOLDER
data_folder = 'Y:\Data\Wada Data Swiss\Visit_JFS_BJE\Originals'
data_raw_file = os.path.join(data_folder, 
                                    'wadatest_14_06_19.edf')

### LOOP OVER ALL SUBJECTS FOR PREPROCESSING ###
### consider putting pre-processing ###

#Read data
raw = mne.io.read_raw_edf(data_raw_file, misc=['ECG EKG-REF'], 
                          stim_channel='Event EVENT-REF', preload=True)

#Convenience function to trim channel names
def ch_rename(oldname): 
    return re.findall(r"\s.+-", oldname)[0][1:-1]

#Trim channel names
raw.rename_channels(ch_rename)

#Print overall and detailed info about raw dataset
print(raw)
print(raw.info)

#Read montage
montage = mne.channels.make_standard_montage('standard_postfixed')

#Set montage
raw.set_montage(montage)

#Plot sensor locations
#raw.plot_sensors(show_names=True)


#Temporarily add dummy annotation to spare user from adding new label
raw.annotations.append(onset=raw.times[0]-1.0, duration=0.0, description='Slow EEG')

#Plot raw EEG traces. Mark onset of slow EEG
raw.plot(start=0, duration=15, n_channels=26, 
         scalings=dict(eeg=1e-4, misc=1e-3, stim=1),
         remove_dc=True, title='Mark onset of slow EEG')

#Crop data around the newly inserted marker
seg_length = 300 #seconds
times_slow = [a['onset'] for a in raw.annotations if 'Slow' in a['description']]
tmin = times_slow[1]-seg_length
tmax = times_slow[1]+seg_length
raw = raw.crop(tmin=tmin,tmax=tmax)

#Temporarily add dummy annotation to spare user from adding new label
raw.annotations.append(onset=raw.times[0]-1.0, duration=0.0, description='BAD_segments')

#Plot raw EEG traces. Reject obviously bad channels and mark bad segments
raw.plot(start=0, duration=15, n_channels=26, 
         scalings=dict(eeg=1e-4, misc=1e-3, stim=1),
         remove_dc=True, title='Reject obviously bad channels and bad segments')

# Making and inserting events for epoching data
epoch_length = 10.0 # sec
overlap = 9.0 # sec
event_id = 1
t_min = 0.0
events = mne.make_fixed_length_events(raw, id=event_id, start=t_min,
                                      stop=None, duration=epoch_length, 
                                      first_samp=True, overlap=overlap)
raw.add_events(events, stim_channel='EVENT', replace=False)

# Check that events are in the right place
raw.plot(start=0, duration=15, n_channels=26, 
         scalings=dict(eeg=1e-4, misc=1e-3, stim=1),
         remove_dc=True, title='Check position of events', events=events)

# Read epochs
rawepochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=t_min, 
                    tmax=epoch_length, baseline=(None, None), picks='eeg', 
                    preload=True, reject=None, proj=False)

#Plot epoched data
rawepochs.plot(n_epochs=10, n_channels=22, scalings=dict(eeg=1e-4, misc=1e-3, stim=100))

#Plot power spectrum
rawepochs.plot_psd(fmax=180)

#Filter the data from 1-100 Hz using the default options
#NOTE: Usually you should apply high-pass and low-pass filter separately, but 
#this is done 'behind the scenes' in this case
epochs = rawepochs.copy().filter(1, 80, picks='eeg', filter_length='auto', 
                 l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
                 method='fir', phase='zero', fir_window='hamming', 
                 fir_design='firwin')

#Plot power spectra
epochs.plot_psd(fmax=180)

#Plot epoched EEG traces. Reject obviously bad channels and mark bad segments
epochs.plot(n_epochs=10, n_channels=22, scalings=dict(eeg=3e-4, misc=1e-3, stim=100), 
            title='Reject obviously bad channels and bad segments')


#Set up and fit the ICA
ica = mne.preprocessing.ICA(method = 'infomax', fit_params=dict(extended=True),
                            random_state=0, max_iter=1000)

ica.fit(epochs, picks='eeg')

#Quick look at components
ica.plot_components(inst=epochs, plot_std=True,
                    picks='eeg',
                    psd_args=dict(fmax=85))

#Plot time course of ICs
ica.plot_sources(epochs)

# =============================================================================
# #Check components one by one and mark bad ones
# n_comps = ica.get_components().shape[1]
# is_brain = [True for i in range(0,n_comps)]
# print('Press a keyboard key for brain, and a mouse button for non-brain')
# for i in range(0,n_comps) :
#     ica.plot_properties(prep, picks=i, psd_args=dict(fmin=0, fmax=110))
#     is_brain[i] = plt.waitforbuttonpress()
#     plt.close()
# idx_bad = [i for i, x in enumerate(is_brain) if not(x)]   
# ica.exclude = idx_bad
# =============================================================================

ica.apply(epochs)

#Plot cleaned data
epochs.plot(scalings=dict(eeg=3e-4, misc=1e-3, stim=1),n_epochs=5)

#Compare power spectra
epochs.plot_psd(fmax=90)

#Set bipolar (double banana) reference
anodes = ['Fp2', 'F8', 'T4', 'T6', 'Fp1', 'F7', 'T3', 'T5', 
          'Fp2', 'F4', 'C4', 'P4', 'Fp1', 'F3', 'C3', 'P3',
          'Fz', 'Cz',
          'T6', 'T5',
          'T4', 'T3']
cathodes = ['F8', 'T4', 'T6', 'O2', 'F7', 'T3', 'T5', 'O1', 
            'F4', 'C4', 'P4', 'O2', 'F3', 'C3', 'P3', 'O1',
            'Cz', 'Pz',
            'A2', 'A1',
            'T2', 'T1']

#Read montage
montage = mne.channels.make_standard_montage('standard_postfixed')

#Set montage
epochs.set_montage(montage)


epochs_bipolar = mne.set_bipolar_reference(epochs, anodes, cathodes, 
                                           drop_refs=False)


#Print info for bipolar (double banana) reference raw data
print(prep_bi)
print(prep_bi.info['ch_names'])
#WARNING: Plotting of sensor locations does not work, set locations first
#Plot sensor locations for bipolar (double banana) reference raw data
#raw_bi.plot_sensors(show_names=True)

# =============================================================================
# order=np.array([0, 2, 4, 6, 21, 8, 22, 23, 10, 12,
#                 14, 15,
#                 1, 3, 5, 7, 18, 9, 19, 20, 11, 13, 
#                 16, 17])
# =============================================================================

ch_names = ['T3-T1', 'T5-A1', 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp1-F3', 
            'F3-C3', 'C3-P3', 'P3-O1', 'Fz-Cz', 'Cz-Pz', 'Fp2-F4', 'F4-C4', 
            'C4-P4', 'P4-O2', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'T4-T2',
             'T6-A2', 'EKG', 'EVENT']

# =============================================================================
# ch_names = ['T1-A1','F7-A1','T3-A1','T5-A1','Fp1-A1','F3-A1','C3-A1','P3-A1','O1-A1',
#             'Fz-Cz','Pz-Cz',
#             'O2-A2','P4-A2','C4-A2','F4-A2','Fp2-A2','T6-A2','T4-A2','F8-A2','T2-A2',
#             'EKG','EVENT']
# =============================================================================

prep_bi.reorder_channels(ch_names)

#Plot re-referenced data (bipolar double banana reference)
prep_bi.plot(start=0, duration=15, n_channels=24, 
          scalings=dict(eeg=1e-4, misc=1e-3, stim=100),
          remove_dc=False)

#Compare power spectra
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
ax.set_xlim(0, 110)
ax.set_ylim(-70, 50)
#raw.plot_psd(fmax=110, ax=ax)
prep_bi.plot_psd(fmax=110, ax=ax)

prep_short = prep_bi.copy()

# =============================================================================
# # Filter again
# prep_short = prep_short.filter(1, 80, picks='eeg', filter_length='auto', 
#                  l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
#                  method='fir', phase='zero', fir_window='hamming', 
#                  fir_design='firwin')
# #Compare power spectra
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
# ax.set_xlim(0, 100)
# ax.set_ylim(-70, 50)
# prep_short.plot_psd(fmax=100, ax=ax)
# =============================================================================

#prep_short = prep_short.crop(tmin=3840,tmax=4740)

#Plot cropped data
prep_short.plot(start=0, duration=15, n_channels=24, 
          scalings=dict(eeg=1e-4, misc=1e-3, stim=100),
          remove_dc=False)

#Get start of infusion. 
#WARNING: Hard coded index + not equal to start of slowing of EEG
#time_ipsi_slow = prep_short.annotations[0]['onset']-prep_short._first_time
time_ipsi_slow = prep_short.annotations[1]['onset']-prep_short._first_time #!!! Horrible hack! Manually inserted annotation
epoch_length = 16
time_first_event = time_ipsi_slow - epoch_length*(time_ipsi_slow//epoch_length)
events = mne.make_fixed_length_events(prep_short, id=1, start=time_first_event,
                                      stop=None, duration=epoch_length, 
                                      first_samp=True, overlap=0.0)
prep_short.add_events(events, stim_channel='EVENT', replace=False)

#Plot data with added events
prep_short.plot(start=0, duration=15, n_channels=24, 
          scalings=dict(eeg=1e-4, misc=1e-3, stim=100),
          remove_dc=False, events=events)

# Read epochs
epochs = mne.Epochs(prep_short, events=events, event_id=1, tmin=0.0, 
                    tmax=epoch_length, baseline=(None, None), picks='eeg', 
                    preload=True, reject=None, proj=False)

#Plot epoched data
epochs.plot(n_epochs=3, n_channels=22, scalings=dict(eeg=1e-4, misc=1e-3, stim=100))

#Get the 3D matrix of epoched EEG-data
data = epochs.get_data(picks='eeg')
idx_left = [2,3,4,5,6,7,8,9] #[3,4,7,8] #[2,3,4,5,7,8]
idx_right = [12,13,14,15,16,17,18,19] #[13,14,17,18] #[13,14,16,17,18,19]
idx_all =  idx_left+idx_right #[3,4,7,8,13,14,17,18]

#Calculate Lempel-Ziv complexity
LZC = np.zeros(data.shape[0])
LZCcontra = np.zeros(data.shape[0])
LZCipsi = np.zeros(data.shape[0])
for i in range(0,data.shape[0]) :
    LZC[i] = ce.LZc(np.transpose(data[i,idx_all,:]))
    LZCcontra[i] = ce.LZc(np.transpose(data[i,idx_left,:]))
    LZCipsi[i] = ce.LZc(np.transpose(data[i,idx_right,:]))
#Plot LZC vs epoch number
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
#plt.plot(range(1,data.shape[0]+1), LZC/LZC[50:60].mean())
#plt.plot(range(1,data.shape[0]+1), LZCcontra/LZCcontra[50:60].mean())
#plt.plot(range(1,data.shape[0]+1), LZCipsi/LZCipsi[50:60].mean())
#plt.step(range(1,data.shape[0]+1), LZC/LZC[50:60].mean(),where='mid')
plt.step(range(1,data.shape[0]+1), LZCcontra/LZCcontra[50:60].mean(),where='mid')
plt.step(range(1,data.shape[0]+1), LZCipsi/LZCipsi[50:60].mean(),where='mid')
ylim = ax.get_ylim()
plt.plot([59.5, 59.5],ylim,'k:')
plt.text(59.5, ylim[1]+0.02*(ylim[1]-ylim[0]),'Start Etomidtae',horizontalalignment='center')
plt.plot([50, 113],[1, 1],'k:')
ax.set_xlim(50, 113)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('LZC/LZC_baseline')
plt.legend(('tLZCcontra', 'tLZCipsi'))
plt.title('Lempel-Ziv complexity - 16s epochs - 8 bipolar channels - 1-30 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#Calculate amplitude coalition entropy
ACE = np.zeros(data.shape[0])
ACEcontra = np.zeros(data.shape[0])
ACEipsi = np.zeros(data.shape[0])
for i in range(0,data.shape[0]) :
    ACE[i] = ce.ACE(data[i,idx_all,:])
    ACEcontra[i] = ce.ACE(data[i,idx_left,:])
    ACEipsi[i] = ce.ACE(data[i,idx_right,:])
#Plot ACE vs epoch number
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#plt.plot(range(1,data.shape[0]+1), ACE/ACE[50:60].mean())
#plt.plot(range(1,data.shape[0]+1), ACEcontra/ACEcontra[50:60].mean())
#plt.plot(range(1,data.shape[0]+1), ACEipsi/ACEipsi[50:60].mean())
#plt.step(range(1,data.shape[0]+1), ACE/ACE[50:60].mean(),where='mid')
plt.step(range(1,data.shape[0]+1), ACEcontra/ACEcontra[50:60].mean(),where='mid')
plt.step(range(1,data.shape[0]+1), ACEipsi/ACEipsi[50:60].mean(),where='mid')
ylim = ax.get_ylim()
plt.plot([59.5, 59.5],ylim,'k:')
plt.text(59.5, ylim[1]+0.02*(ylim[1]-ylim[0]),'Start Etomidtae',horizontalalignment='center')
plt.plot([50, 113],[1, 1],'k:')
ax.set_xlim(50, 113)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('ACE/ACE_baseline')
plt.legend(('ACEcontra', 'ACEipsi'))
plt.title('Amplitude coalition entropy - 16s epochs - 8 bipolar channels - 1-35 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#Calculate synchrony coalition entropy
SCE = np.zeros(data.shape[0])
SCEcontra = np.zeros(data.shape[0])
SCEipsi = np.zeros(data.shape[0])
for i in range(0,data.shape[0]) :
    SCE[i] = ce.SCE(data[i,idx_all,:])
    SCEcontra[i] = ce.SCE(data[i,idx_left,:])
    SCEipsi[i] = ce.SCE(data[i,idx_right,:])
#Plot SCE vs epoch number
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
#plt.plot(range(1,data.shape[0]+1), SCE/SCE[50:60].mean())
#plt.plot(range(1,data.shape[0]+1), SCEcontra/SCEcontra[50:60].mean())
#plt.plot(range(1,data.shape[0]+1), SCEipsi/SCEipsi[50:60].mean())
#plt.step(range(1,data.shape[0]+1), SCE/SCE[50:60].mean(),where='mid')
plt.step(range(1,data.shape[0]+1), SCEcontra/SCEcontra[50:60].mean(),where='mid')
plt.step(range(1,data.shape[0]+1), SCEipsi/SCEipsi[50:60].mean(),where='mid')
ylim = ax.get_ylim()
plt.plot([59.5, 59.5],ylim,'k:')
plt.text(59.5, ylim[1]+0.02*(ylim[1]-ylim[0]),'Start Etomidtae',horizontalalignment='center')
plt.plot([50, 113],[1, 1],'k:')
ax.set_xlim(50, 113)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('SCE/SCE_baseline')
plt.legend(('SCEcontra', 'SCEipsi'))
plt.title('Synchrony coalition entropy - 16s epochs - 8 bipolar channels - 1-35 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)     
    
## POSSIBLY USEFUL ##
    
# =============================================================================
# #Resample if needed (Warning: looking at PSD there seems to be some passband-ripples?)
# prep = raw.copy().resample(64)
# 
# #Compare power spectra
# raw.plot_psd(fmax=32)
# prep.plot_psd(fmax=32)
# 
# #Compare EEG traces
# raw.plot(start=0, duration=15, n_channels=26, 
#          scalings=dict(eeg=1e-4, misc=1e-3, stim=1),
#          remove_dc=True)
# prep.plot(start=0, duration=15, n_channels=26, 
#          scalings=dict(eeg=1e-4, misc=1e-3, stim=1),
#          remove_dc=True)
# =============================================================================

# =============================================================================
# #Construct and visualize FIR filter (recommended over IIR for most applications)
# sfreq = 1000.
# f_p = 40.
# flim = (1.0, sfreq / 2.0)  # limits for plotting
# nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate
# freq = [0, f_p, f_p, nyq]
# gain = [1, 1, 0, 0]
# 
# third_height = np.array(plt.rcParams['figure.figsize']) * [1, 1.0 / 3.]
# ax = plt.subplots(1, figsize=third_height)[1]
# plot_ideal_filter(freq, gain, ax, title='Ideal %s Hz lowpass' % f_p, flim=flim)
# =============================================================================



## GRAVEYARD ##

# =============================================================================
# stim_data = np.zeros((1, len(prep_short.times)))
# info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
# stim_raw = mne.io.RawArray(stim_data, info)
# raw.add_channels([stim_raw], force_update_info=True)
# 
# =============================================================================

# =============================================================================
# #Set bipolar (double banana) reference
# anodes = ['Fp2', 'F8', 'T4', 'T6', 'Fp1', 'F7', 'T3', 'T5', 
#           'Fp2', 'F4', 'C4', 'P4', 'Fp1', 'F3', 'C3', 'P3',
#           'Fz', 'Cz',
#           'T6', 'T5',
#           'T4', 'T3']
# cathodes = ['F8', 'T4', 'T6', 'O2', 'F7', 'T3', 'T5', 'O1', 
#             'F4', 'C4', 'P4', 'O2', 'F3', 'C3', 'P3', 'O1',
#             'Cz', 'Pz',
#             'A2', 'A1',
#             'T2', 'T1']
# raw_bi = mne.set_bipolar_reference(raw, anodes, cathodes)
# #Print info for bipolar (double banana) reference raw data
# print(raw_bi)
# print(raw_bi.info)
# #WARNING: Plotting of sensor locations does not work, set locations first
# #Plot sensor locations for bipolar (double banana) reference raw data
# #raw_bi.plot_sensors(show_names=True)
# =============================================================================
