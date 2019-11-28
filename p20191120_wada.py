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
data_folder = 'Z:\\Data\\Wada_Data_Swiss'
filenames = ['Pilot 1\\wadatest.edf', 
             'Pilot 2\\NEEDS TO BE EXPORTED AS EDF', 
             'Visit_JFS_BEJ\\Wadatest_14_06_2019_EDF.edf']
savefolder = 'analysis'
savenames = ['pilot1', 
             'pilot2', 
             'pilot3']

# select file to work on
data_num = 2
data_raw_file = os.path.join(data_folder, 
                                    filenames[data_num])

### LOOP OVER ALL SUBJECTS FOR PREPROCESSING ###
### consider putting pre-processing ###

#Read data
raw = mne.io.read_raw_edf(data_raw_file, misc=['ECG EKG-REF'], 
                          stim_channel='Event EVENT-REF', preload=True)

## THIS FUNCTION DOES NOT WORK ON MY COMPUTER!
##Convenience function to trim channel names
#def ch_rename(oldname): 
#    return re.findall(r"\s.+-", oldname)[0][1:-1]
#
##Trim channel names
#raw.rename_channels(ch_rename)

#Print overall and detailed info about raw dataset
print(raw)
print(raw.info)

#Read montage
montage = mne.channels.make_standard_montage('standard_postfixed')

#Set montage
raw.set_montage(montage,raise_if_subset=False)

##Plot sensor locations
#raw.plot_sensors(show_names=True)

#Temporarily add dummy annotation to spare user from adding new label
raw.annotations.append(onset=raw.times[0]-1.0, duration=0.0, description='Slow EEG')

#Plot raw EEG traces. Mark onset of slow EEG and any non-EEG channels
raw.plot(start=0, duration=15, n_channels=26, 
         scalings=dict(eeg=1e-4, misc=1e-3, stim=1),
         remove_dc=True, title='Mark onset of slow EEG and set non-EEG channels as bad')

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
overlap = 5.0 # sec
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
rawepochs.plot_psd(fmax=180,picks='eeg')

#Filter the data from 1-80 Hz using the default options
#NOTE: Usually you should apply high-pass and low-pass filter separately, but 
#this is done 'behind the scenes' in this case
epochs = rawepochs.copy().filter(1, 80, picks='eeg', filter_length='auto', 
                 l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
                 method='fir', phase='zero', fir_window='hamming', 
                 fir_design='firwin')

#Plot power spectra
epochs.plot_psd(fmax=180,picks='eeg')

#Plot epoched EEG traces. Reject obviously bad channels and mark bad segments
epochs.plot(n_epochs=10, n_channels=22, scalings=dict(eeg=3e-4, misc=1e-3, stim=100), 
            title='Reject obviously bad channels and bad segments')

#Set up and fit the ICA
ica = mne.preprocessing.ICA(method = 'infomax', fit_params=dict(extended=True),
                            random_state=0, max_iter=1000,
                            n_components=epochs.info['nchan'])

ica.fit(epochs, picks='eeg')

#Quick look at components
ica.plot_components(inst=epochs, plot_std=True,
                    ch_type='eeg',
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
epochs.plot(scalings=dict(eeg=3e-4),n_epochs=5)

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

# Alternative setup (not requiring reordering, and with no overlap)
anodes = ['T1', 
          'Fp1', 'F7', 'T3', 'T5',
          'F3', 'T3',  
          'Fz', 'Cz',
          'F4', 'T4',  
          'Fp2', 'F8', 'T4', 'T6',
          'T2']
cathodes = ['A1', 
            'F7', 'T3', 'T5', 'O1',
            'T3', 'P3',  
            'Cz', 'Pz',
            'T4', 'P4',  
            'F8', 'T4', 'T6', 'O2',
            'A2']

#Read and set original montage
montage = mne.channels.make_standard_montage('standard_postfixed')
epochs.set_montage(montage,raise_if_subset=False)

# Set bipolar montage
epochs_bipolar = mne.set_bipolar_reference(epochs, anodes, cathodes, 
                                           drop_refs=False)

# Removing old channels (keeping only biploar)
epochs_bipolar.picks = None
epochs_bipolar.drop_channels(epochs.info['ch_names'])

#Print info for bipolar (double banana) reference raw data
print(epochs_bipolar)
print(epochs_bipolar.info['ch_names'])

# reordering bipolar channels (given original setup of channels)
ch_order = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 
            'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',  
            'Fz-Cz', 'Cz-Pz', 
            'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
            'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 
            'T3-T1', 
            'T5-A1', 
            'T4-T2',
            'T6-A2']

epochs_bipolar.reorder_channels(ch_order)

#Plot re-referenced data (bipolar double banana reference)
epochs_bipolar.plot(scalings=dict(eeg=1e-4, misc=1e-3, stim=100),
                    n_epochs=5,title='epoched and cleaned data with double banana reference')

# Plot power spectrum
epochs_bipolar.plot_psd(fmax=110)

# Filter again
preprocessed = epochs_bipolar.filter(1, 30, filter_length='auto', 
                 l_trans_bandwidth='auto', h_trans_bandwidth='auto', 
                 method='fir', phase='zero', fir_window='hamming', 
                 fir_design='firwin')

#Plot cropped data
preprocessed.plot(scalings=dict(eeg=1e-4),title='processed data',n_epochs=1)


### CALCULATING THE MEASURES ###
#Get the 3D matrix of epoched EEG-data
data = preprocessed.get_data(picks='eeg')
idx_left = [1,2,3,4,5,6] #[3,4,7,8] #[2,3,4,5,7,8]
idx_right = [9,10,11,12,13,14] #[13,14,17,18] #[13,14,16,17,18,19]
idx_all =  idx_left+idx_right #[3,4,7,8,13,14,17,18]
idx_drop = [7,8]
half_idx = list(np.random.choice(idx_left,3,replace=False)) + list(np.random.choice(idx_right,3,replace=False)) 

# creating subsampled data
right_eeg = preprocessed.copy().drop_channels([preprocessed.info['ch_names'][i] for i in idx_left+idx_drop])
left_eeg = preprocessed.copy().drop_channels([preprocessed.info['ch_names'][i] for i in idx_right+idx_drop])
all_eeg = preprocessed.copy().drop_channels([preprocessed.info['ch_names'][i] for i in idx_drop])
half_eeg = preprocessed.copy().drop_channels([preprocessed.info['ch_names'][i] for i in half_idx])

# extracting data
all_data = all_eeg.get_data(picks='eeg')
left_data = left_eeg.get_data(picks='eeg')
right_data = right_eeg.get_data(picks='eeg')
half_data = half_eeg.get_data(picks='eeg')

### Calculate Lempel-Ziv complexity (concatinated in time)
LZC = np.zeros(data.shape[0])
LZCcontra = np.zeros(data.shape[0])
LZCipsi = np.zeros(data.shape[0])
LZChalf = np.zeros(data.shape[0])
for i in range(0,data.shape[0]) :
    LZC[i] = ce.LZc(all_data[i,:,:],'time')
    LZCcontra[i] = ce.LZc(left_data[i,:,:],'time')
    LZCipsi[i] = ce.LZc(right_data[i,:,:],'time')
    LZChalf[i] = ce.LZc(half_data[i,:,:],'time')
    
### LZ spatial concatination
LZC_s = np.zeros(all_data.shape[0])
LZCcontra_s = np.zeros(left_data.shape[0])
LZCipsi_s = np.zeros(right_data.shape[0])
LZChalf_s = np.zeros(half_data.shape[0])
for i in range(0,data.shape[0]):
    LZC_s[i] = ce.LZc(all_data[i,:,:])
    LZCcontra_s[i] = ce.LZc(left_data[i,:,:])
    LZCipsi_s[i] = ce.LZc(right_data[i,:,:])
    LZChalf_s[i] = ce.LZc(half_data[i,:,:])

#Calculate amplitude coalition entropy
ACE = np.zeros(all_data.shape[0])
ACEcontra = np.zeros(left_data.shape[0])
ACEipsi = np.zeros(right_data.shape[0])
ACEhalf = np.zeros(half_data.shape[0])
for i in range(0,data.shape[0]) :
    ACE[i] = ce.ACE(all_data[i,:,:])
    ACEcontra[i] = ce.ACE(left_data[i,:,:])
    ACEipsi[i] = ce.ACE(right_data[i,:,:])
    ACEhalf[i] = ce.ACE(half_data[i,:,:])
    
#Calculate synchrony coalition entropy
SCE = np.zeros(all_data.shape[0])
SCEcontra = np.zeros(left_data.shape[0])
SCEipsi = np.zeros(right_data.shape[0])
SCEhalf = np.zeros(half_data.shape[0])
for i in range(0,data.shape[0]) :
    SCE[i] = ce.SCE(all_data[i,:,:])
    SCEcontra[i] = ce.SCE(left_data[i,:,:])
    SCEipsi[i] = ce.SCE(right_data[i,:,:])
    SCEhalf[i] = ce.SCE(half_data[i,:,:])
    
## Calculating Time-Frequency (multitaper)
sfreq = preprocessed.info['sfreq']
freqs = mne.time_frequency.psd_array_multitaper(all_data[0], sfreq, fmin=1, fmax=30, adaptive=True)[1]
# all data
TF = np.transpose(np.array([np.median(mne.time_frequency.psd_array_multitaper(d, sfreq, fmin=1, fmax=30, adaptive=True)[0],0) for d in all_data]))
dB = np.array([tf/np.mean(tf[:int(trials/2-1)]) for tf in TF])
# half data
TFhalf = np.transpose(np.array([np.median(mne.time_frequency.psd_array_multitaper(d, sfreq, fmin=1, fmax=30, adaptive=True)[0],0) for d in half_data]))
dBhalf = np.array([tf/np.mean(tf[:int(trials/2-1)]) for tf in TFhalf])
# ipsi data
TFleft = np.transpose(np.array([np.median(mne.time_frequency.psd_array_multitaper(d, sfreq, fmin=1, fmax=30, adaptive=True)[0],0) for d in left_data]))
dBleft = np.array([tf/np.mean(tf[:int(trials/2-1)]) for tf in TFleft])
# contra data
TFright = np.transpose(np.array([np.median(mne.time_frequency.psd_array_multitaper(d, sfreq, fmin=1, fmax=30, adaptive=True)[0],0) for d in right_data]))
dBright = np.array([tf/np.mean(tf[:int(trials/2-1)]) for tf in TFright])
    
# =============================================================================
#      
# #Plot LZC vs epoch number (normalized)
# trials = data.shape[0]+1
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
# plt.step(range(1,trials), LZCcontra/LZCcontra[:int(trials/2.-1)].mean(),where='mid')
# plt.step(range(1,trials), LZCipsi/LZCipsi[:int(trials/2.-1)].mean(),where='mid')
# plt.step(range(1,trials), LZC/LZC[:int(trials/2.-1)].mean(),where='mid')
# plt.step(range(1,trials), LZChalf/LZChalf[:int(trials/2.-1)].mean(),where='mid')
# ylim = ax.get_ylim()
# plt.plot([trials/2, trials/2],ylim,'k:')
# plt.text(trials/2, ylim[1]+0.02*(ylim[1]-ylim[0]),'Start Etomidate',horizontalalignment='center')
# plt.plot([0, trials],[1, 1],'k:')
# ax.set_xlim(0, trials)
# ax.set_ylim(ylim)
# plt.xlabel('Epoch number')
# plt.ylabel('LZC/LZC_baseline')
# plt.legend(('tLZCcontra', 'tLZCipsi','tLZ all','half'))
# plt.title('Lempel-Ziv complexity - 10s epochs - 6 bipolar channels - 1-30 Hz')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# 
# =============================================================================
#Plot LZC vs epoch number (not normalized)
trials = data.shape[0]+1
fig_LZt = plt.figure()
ax = fig_LZt.add_axes([0.1, 0.1, 0.85, 0.85])
plt.step(range(1,trials), LZCcontra,where='mid')
plt.step(range(1,trials), LZCipsi,where='mid')
plt.step(range(1,trials), LZC,where='mid')
plt.step(range(1,trials), LZChalf,where='mid')
ylim = ax.get_ylim()
plt.plot([trials/2, trials/2],ylim,'k:')
plt.text(trials/2, ylim[1]-0.05*(ylim[1]-ylim[0]),'Start Etomidate',horizontalalignment='center')
plt.plot([0, trials],[1, 1],'k:')
ax.set_xlim(0, trials)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('LZC')
plt.legend(('tLZCcontra', 'tLZCipsi','tLZ all','half'))
plt.title('Lempel-Ziv complexity (time) - 10s epochs - 6 bipolar channels - 1-30 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

   
#Plot LZC vs epoch number (space,not normalized)
trials = data.shape[0]+1
fig_LZs = plt.figure()
ax = fig_LZs.add_axes([0.1, 0.1, 0.85, 0.85])
plt.step(range(1,trials), LZCcontra_s,where='mid')
plt.step(range(1,trials), LZCipsi_s,where='mid')
plt.step(range(1,trials), LZC_s,where='mid')
plt.step(range(1,trials), LZChalf_s,where='mid')
ylim = ax.get_ylim()
plt.plot([trials/2, trials/2],ylim,'k:')
plt.text(trials/2, ylim[1]-0.05*(ylim[1]-ylim[0]),'Start Etomidate',horizontalalignment='center')
plt.plot([0, trials],[1, 1],'k:')
ax.set_xlim(0, trials)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('LZC')
plt.legend(('tLZCcontra', 'tLZCipsi','tLZ all','half'))
plt.title('Lempel Ziv Complexity (space)   - 10s epochs - 6 bipolar channels - 1-30 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#Plot ACE vs epoch number
fig_ACE = plt.figure()
ax = fig_ACE.add_axes([0.1, 0.1, 0.8, 0.8])
plt.step(range(1,data.shape[0]+1), ACEcontra,where='mid')
plt.step(range(1,data.shape[0]+1), ACEipsi,where='mid')
plt.step(range(1,data.shape[0]+1), ACE,where='mid')
plt.step(range(1,data.shape[0]+1), ACEhalf,where='mid')
ylim = ax.get_ylim()
plt.plot([trials/2, trials/2],ylim,'k:')
plt.text(trials/2, ylim[1]-0.05*(ylim[1]-ylim[0]),'Start Etomidate',horizontalalignment='center')
plt.plot([0, trials],[1, 1],'k:')
ax.set_xlim(0, trials)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('ACE')
plt.legend(('tACEcontra', 'tACEipsi','tACE all','ACEhalf'))
plt.title('Amplitude Coalition Entropy - 10s epochs - 6 bipolar channels - 1-30 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#Plot ACE vs epoch number
fig_SCE = plt.figure()
ax = fig_SCE.add_axes([0.1, 0.1, 0.8, 0.8])
plt.step(range(1,data.shape[0]+1), SCEcontra,where='mid')
plt.step(range(1,data.shape[0]+1), SCEipsi,where='mid')
plt.step(range(1,data.shape[0]+1), SCE,where='mid')
plt.step(range(1,data.shape[0]+1), SCEhalf,where='mid')
ylim = ax.get_ylim()
plt.plot([trials/2, trials/2],ylim,'k:')
plt.text(trials/2, ylim[1]-0.05*(ylim[1]-ylim[0]),'Start Etomidate',horizontalalignment='center')
plt.plot([0, trials],[1, 1],'k:')
ax.set_xlim(0, trials)
ax.set_ylim(ylim)
plt.xlabel('Epoch number')
plt.ylabel('SCE')
plt.legend(('tSCEcontra', 'tSCEipsi','tSCE all','SCEhalf'))
plt.title('Synchrony coalition entropy  - 10s epochs - 6 bipolar channels - 1-30 Hz')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# plotting time frequency
vmin = -4
vmax = 4
fig_TF = plt.figure()
plt.subplot(141)
plt.imshow(np.log(dB),vmin=vmin,vmax=vmax)
plt.xticks([], [])
plt.yticks(np.arange(0,len(freqs),20),np.arange(1,30,2))
plt.ylabel('Multitaper time-frequency plots \nFrequencies (Hz)')
plt.xlabel('Time/Epochs \n(centered on induction)')
plt.title('all channels')
plt.subplot(142)
plt.imshow(np.log(dBhalf),vmin=vmin,vmax=vmax)
plt.xticks([], [])
plt.yticks([], [])
plt.title('half channels')
plt.subplot(143)
plt.imshow(np.log(dBleft),vmin=vmin,vmax=vmax)
plt.xticks([], [])
plt.yticks([], [])
plt.title('left channels \n(contra)')
plt.subplot(144)
plt.imshow(np.log(dBright),vmin=vmin,vmax=vmax)
plt.xticks([], [])
plt.yticks([], [])
plt.title('right channels \n(ipsi)')
plt.colorbar(fraction=.03)
plt.autoscale(False)


# saving data and figures
save_file = os.path.join(savefolder, 
                                    savenames[data_num])

savedata = {
        'data' : {'all':all_data,
                  'half':half_data,
                  'ipsi':right_data,
                  'contra':left_data,
                  },
        'time-frequency' : {'all': dB,
                  'half': dBhalf,
                  'ipsi': dBright,
                  'contra': dBleft,
                  },
        'LZ space' : {'all': LZC_s,
                  'half': LZChalf_s,
                  'ipsi':LZCipsi_s,
                  'contra':LZCcontra_s,
                  },
        'LZ time' : {'all': LZC,
                  'half': LZChalf,
                  'ipsi':LZCipsi,
                  'contra':LZCcontra,
                  },
        'ACE' : {'all': ACE,
                  'half': ACEhalf,
                  'ipsi':ACEipsi,
                  'contra':ACEcontra,
                  },
        'SCE' : {'all': SCE,
                  'half': SCEhalf,
                  'ipsi':SCEipsi,
                  'contra':SCEcontra,
                  }
        }


os.mkdir(save_file)
fig_TF.savefig(save_file+'\\timefreq.png')
fig_LZs.savefig(save_file+'\\LZ_s.png')
fig_LZt.savefig(save_file+'\\LZ_t.png')
fig_ACE.savefig(save_file+'\\ACE.png')
fig_SCE.savefig(save_file+'\\SCE.png')

import pickle
f = open(save_file+'\\data.pckl','wb')
pickle.dump(savedata,f)