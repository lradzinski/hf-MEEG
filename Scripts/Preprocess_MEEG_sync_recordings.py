# %%
"""
Preprocess_MEEG_sync_recordings.py
Lukasz Radzinski
Charite Neurophysics Group, Berlin
Script for preprocessing combined
MEEG recordings 
"""

# %%
import os
import meet
import scipy
import tables
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import signal as sig
import helper_scripts.helper_functions as helper_functions

# set global parameters for plots
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['savefig.dpi'] = 600

# %%
# select subject [S1-S7]

subject = 'S1'
date = '2018-03-19'

data_input_folder = '../Data/raw_data'
data_output_folder = '../Data/cleaned_data'
plots_output_folder = '../Results'

additional_plot_title = ''

use_brown_noise = False

subject_name = subject+'_MEEG_sync_stim'
data_input_folder = os.path.join(data_input_folder, date, subject)
data_output_folder = os.path.join(data_output_folder, date, subject)
plots_output_folder = os.path.join(plots_output_folder, subject, 'preprocessing', date+'_'+subject_name)

meg_srate = 20000
meg_unit = 'B [fT]'
meg_asd_unit = '[fT/√HZ]'

eeg_srate = 10000
eeg_unit = 'U [μV]'
eeg_asd_unit = '[μV/√HZ]'

# %%
# load the data

filename = '%s_MEG_comb_sc_stim.dat' % subject
raw_meg_data_stim = helper_functions.readMEG(os.path.join(data_input_folder, filename), s_rate=meg_srate, num_chans=2)
filename = '%s_MEG_comb_sc_rest.dat' % subject
raw_meg_data_rest = helper_functions.readMEG(os.path.join(data_input_folder, filename), s_rate=meg_srate, num_chans=2)

filename = '%s_EEG_comb_mc_stim.hdf5' % subject
h5file_eeg_stim = tables.open_file(os.path.join(data_input_folder, filename), mode="r", title="%s_eeg_stim" % subject)
raw_eeg_data_stim = np.array(h5file_eeg_stim.root.EEG_data).T
filename = '%s_EEG_comb_mc_rest.hdf5' % subject
h5file_eeg_rest = tables.open_file(os.path.join(data_input_folder, filename), mode="r", title="%s_eeg_rest" % subject)
raw_eeg_data_rest = np.array(h5file_eeg_rest.root.EEG_data).T

# %%
# extract the signals

MEG_data = -raw_meg_data_stim[:-1]
MEG_stimulus_data = raw_meg_data_stim[-1]
MEG_rest_data = -raw_meg_data_rest[:-1]

EEG_data = raw_eeg_data_stim[:-1, 40:]
EEG_stimulus_data = raw_eeg_data_stim[-1][:-40]
EEG_rest_data = raw_eeg_data_rest[:-1, 40:]

# %%
# add header to the plot function
def plt_header(main_title='', use_suptitle = False, fontsize=12):

    title = subject_name+', '+date
    title += '\n'+main_title+additional_plot_title

    if(use_suptitle):
        plt.suptitle(title, fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)

# %%
# show and save the plot function
def plt_show_save_fig(fig_name=None):

    if(fig_name):
        fig_name += '.png'
    else:
        plt_show_save_fig.counter += 1
        fig_name = 'Fig%02d.png' % plt_show_save_fig.counter

    print('--------------------\n'+fig_name)
    os.makedirs(plots_output_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_output_folder, fig_name), bbox_inches='tight')
    plt.show()

plt_show_save_fig.counter = 0

# %%
# normalize signal to 0-1 range
def min_max_normalize_sig(signal):

    signal_normalized = signal - np.min(signal)
    signal_normalized /= np.max(signal_normalized)

    return signal_normalized

# %%
# get the stimuli positions
all_stimuli_meg = ((MEG_stimulus_data[1:]>162500) & (MEG_stimulus_data[:-1]<162500)).nonzero()[0]
all_stimuli_eeg = ((EEG_stimulus_data[1:]>0.5) & (EEG_stimulus_data[:-1]<0.5)).nonzero()[0]

# get the marker, omit the first and last (avoid edge effects)
marker_meg = all_stimuli_meg[10:-10]
marker_eeg = all_stimuli_eeg[10:-10]
marker_eeg = marker_eeg[:len(marker_meg)]

# %%
# plot MEG and stimulus signals

start_time = np.round(marker_meg[0]/meg_srate - 0.001, 5)
end_time = np.round(marker_meg[0]/meg_srate + 0.001, 5)

data = min_max_normalize_sig(MEG_data[0][int(start_time*meg_srate):int(end_time*meg_srate)])
data_x = np.linspace(start_time*1000, end_time*1000, len(data)+1)[:-1]
data_y = data
plt.plot(data_x, data_y, label='raw MEG ch0 signal')

data = min_max_normalize_sig(MEG_stimulus_data[int(start_time*meg_srate):int(end_time*meg_srate)])
data_x = np.linspace(start_time*1000, end_time*1000, len(data)+1)[:-1]
data_y = data
plt.plot(data_x, data_y, label='stimulus signal')

plt.axvline(1000*(marker_meg[0])/meg_srate, color='darkgrey', label='detected position')

plt_header('Detecting MEG stimulus position')
plt.xlabel('t [ms]')
plt.ylabel('normalized units')
plt.legend()
plt_show_save_fig()

# %%
# plot EEG and stimulus signals

start_time = np.round(marker_eeg[0]/eeg_srate - 0.001, 5)
end_time = np.round(marker_eeg[0]/eeg_srate + 0.001, 5)

data = min_max_normalize_sig(EEG_data[0][int(start_time*eeg_srate):int(end_time*eeg_srate)])
data_x = np.linspace(start_time*1000, end_time*1000, len(data)+1)[:-1]
data_y = data
plt.plot(data_x, data_y, label='raw EEG ch0 signal')

data = min_max_normalize_sig(EEG_stimulus_data[int(start_time*eeg_srate):int(end_time*eeg_srate)])
data_x = np.linspace(start_time*1000, end_time*1000, len(data)+1)[:-1]
data_y = data
plt.plot(data_x, data_y, label='stimulus signal')

plt.axvline(1000*(marker_eeg[0])/eeg_srate, color='darkgrey', label='detected position')

plt_header('Detecting EEG stimulus position')
plt.xlabel('t [ms]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

# %%
# remove the stimuli

interpolate_win_ms_meg = [-2, 3]
interpolate_win_meg = np.round(np.array(interpolate_win_ms_meg) / 1000. * meg_srate).astype(int)
MEG_stimuli_removed_data = meet.interpolateEEG(MEG_data.copy(), all_stimuli_meg, interpolate_win_meg)

interpolate_win_ms_eeg = [-5, 8]
interpolate_win_eeg = np.round(np.array(interpolate_win_ms_eeg) / 1000. * eeg_srate).astype(int)
EEG_stimuli_removed_data = meet.interpolateEEG(EEG_data.copy(), all_stimuli_eeg, interpolate_win_eeg)

# %%
# plot MEG signal without stimuli

data = MEG_data[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = MEG_stimuli_removed_data[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from MEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

data = MEG_data[0][:meg_srate]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = MEG_stimuli_removed_data[0][:meg_srate]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from MEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

start_time = marker_meg[0]/meg_srate - 0.01 
end_time = marker_meg[0]/meg_srate + 0.01

data = MEG_data[0][int(start_time*meg_srate):int(end_time*meg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = MEG_stimuli_removed_data[0][int(start_time*meg_srate):int(end_time*meg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from MEG ch0 data')
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

# %%
data = EEG_data[0]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = EEG_stimuli_removed_data[0]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from EEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

data = EEG_data[0][:eeg_srate]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = EEG_stimuli_removed_data[0][:eeg_srate]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from EEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

start_time = marker_eeg[0]/eeg_srate - 0.01 
end_time = marker_eeg[0]/eeg_srate + 0.01

data = EEG_data[0][int(start_time*eeg_srate):int(end_time*eeg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='with stimulation artifacts')

data = EEG_stimuli_removed_data[0][int(start_time*eeg_srate):int(end_time*eeg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='without stimulation artifacts')

plt_header('Removing stimuli artifacts from EEG ch0 data')
plt.xlabel('t [ms]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

# %%
# apply fc=1Hz hp filter to remove DC component
sos = sig.butter(2, 1, 'highpass', fs=meg_srate, output='sos')
MEG_data = sig.sosfiltfilt(sos, MEG_stimuli_removed_data)

# apply fc=1Hz hp filter to remove DC component
sos = sig.butter(2, 1, 'highpass', fs=eeg_srate, output='sos')
EEG_data = sig.sosfiltfilt(sos, EEG_stimuli_removed_data)

# %%
# amplitude spectral density function
def asd(data, nfft, srate):

    yf, xf = mlab.psd(x=data, NFFT=nfft, Fs=srate, window=sig.windows.hann(nfft), noverlap=nfft//2)
    
    return xf, np.sqrt(yf)

# %%
# calculate and plot amplitude spectral density of MEG
# detect peaks in spectrum to remove

plt_header('ASD of preprocessed MEG signal')
peaks_to_remove = [] # freqs [Hz]

nfft = 2**(int(np.log2(meg_srate))-2)

for i in range(len(MEG_data)):
    data = MEG_data[i]
    xf, yf = asd(data, nfft, meg_srate)
    plt.plot(xf, yf, label='ch%d' % i)

    yf_norm = np.log10(yf)
    yf_norm = yf_norm - np.mean(yf_norm)
    yf_norm = yf_norm / np.std(yf_norm)
    asd_peaks = sig.find_peaks(yf_norm, prominence=2)[0]
    peaks_to_remove.append(xf[asd_peaks])
    plt.scatter(xf[asd_peaks], yf[asd_peaks], marker='x', color='red', label='peaks to remove')

plt.xscale('log')
plt.yscale('log')
plt.xlim((1,meg_srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % meg_asd_unit)
plt.legend()
plt.grid(True)
plt_show_save_fig()

# %%
# remove peaks from the MEG signal

MEG_data_freq_peaks_removed = MEG_data.copy()

for n in range(len(peaks_to_remove)):

    if(len(peaks_to_remove[n]) > 0):

        import mne
        info = mne.create_info(ch_names=['ch1'], sfreq=meg_srate, ch_types=['mag'])
        raw = mne.io.RawArray([MEG_data_freq_peaks_removed[n]], info)

        # second filtering round
        raw = raw.notch_filter(freqs=peaks_to_remove[n], method="spectrum_fit")
        MEG_data_freq_peaks_removed[n] = raw['ch1'][0][0]

        # reinterpolate the stimulus to remove filtering artifacts
        MEG_data_freq_peaks_removed[n] = meet.interpolateEEG(MEG_data_freq_peaks_removed[n], marker_meg, interpolate_win_meg)

        # apply low-pass filter to remove potential artifact near Nyquist freqency
        sos = sig.butter(2, meg_srate//2-100, 'lowpass', fs=meg_srate, output='sos')
        MEG_data_freq_peaks_removed[n] = sig.sosfiltfilt(sos, MEG_data_freq_peaks_removed[n])


# %%
# calculate and plot amplitude spectral density of MEG

plt_header('ASD of preprocessed MEG signal, removing peaks artifacts')
peaks_to_remove = [] # freqs [Hz]

nfft = 2**(int(np.log2(meg_srate))-2)

for i in range(len(MEG_data)):
    data = MEG_data[i]
    xf, yf = asd(data, nfft, meg_srate)
    plt.plot(xf, yf, label='ch%d, with peaks' % i)

    data = MEG_data_freq_peaks_removed[i]
    xf, yf = asd(data, nfft, meg_srate)
    plt.plot(xf, yf, label='ch%d, peaks removed' % i)

plt.xscale('log')
plt.yscale('log')
plt.xlim((1,meg_srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % meg_asd_unit)
plt.legend()
plt.grid(True)
plt_show_save_fig()

MEG_data = MEG_data_freq_peaks_removed

# %%
# calculate and plot amplitude spectral density of EEG

plt_header('ASD of preprocessed EEG signal')

nfft = 2**(int(np.log2(eeg_srate))-2)

for i in range(len(EEG_data)):
    data = EEG_data[i]
    xf, yf = asd(data, nfft, eeg_srate)
    plt.plot(xf, yf, label='ch%d' % i)

plt.xscale('log')
plt.yscale('log')
plt.xlim((1,eeg_srate//2))
plt.ylim((0.001, 10))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % eeg_asd_unit)
plt.legend()
plt.grid(True)
plt_show_save_fig()

# %%
# apply 450Hz-850Hz band-pass filter to extract
# high-frequency band (sigma band) and 
# high-frequency somatosensory evoked response (sigma burst)

lfreq_sigma = 450
rfreq_sigma = 850
sigma_freq_range_str = '%sHz-%sHz' % (lfreq_sigma, rfreq_sigma)

meg_sigma_band_data = meet.iir.butterworth(MEG_data, fs=(lfreq_sigma-50, rfreq_sigma+50),
                                        fp=(lfreq_sigma, rfreq_sigma), s_rate=meg_srate)

eeg_sigma_band_data = meet.iir.butterworth(EEG_data, fs=(lfreq_sigma-50, rfreq_sigma+50),
                                        fp=(lfreq_sigma, rfreq_sigma), s_rate=eeg_srate)

# %%
# extract MEG trials to remove outliers

whole_trial_len = int(np.round(np.mean(np.diff(all_stimuli_meg))))
whole_trial_win_samples = [0,whole_trial_len]
whole_trial_t = (np.arange(whole_trial_win_samples[0], whole_trial_win_samples[1], 1)/float(meg_srate)*1000)

MEG_whole_trials = meet.epochEEG(MEG_data, marker_meg, whole_trial_win_samples)
MEG_sigma_band_whole_trials = meet.epochEEG(meg_sigma_band_data, marker_meg, whole_trial_win_samples)

sigma_win_ms = [10, 35]
sigma_win_samples = np.round(np.array(sigma_win_ms)/1000.*meg_srate).astype(int)
MEG_sigma_burst_trials = meet.epochEEG(meg_sigma_band_data, marker_meg, sigma_win_samples)

# %%
# calculate MEG sigma band and sigma burst trials rms and percentiles to remove outliers

not_outliers_meg = []
thresholds_sigma_band_rms_meg = []
thresholds_sigma_burst_rms_meg = []

for i in range(len(MEG_sigma_band_whole_trials)):

    sigma_band_trials_rms = np.sqrt(np.mean(MEG_sigma_band_whole_trials[i]**2, axis=0))

    sigma_band_rms_q25 = scipy.stats.scoreatpercentile(sigma_band_trials_rms, 25)
    sigma_band_rms_q50 = np.median(sigma_band_trials_rms)
    sigma_band_rms_q75 = scipy.stats.scoreatpercentile(sigma_band_trials_rms, 75)
    sigma_band_rms_iqr = sigma_band_rms_q75 - sigma_band_rms_q25

    # set a high threshold to remove only very outliers
    threshold_sigma_band_rms = sigma_band_rms_q75 + 3*sigma_band_rms_iqr
    thresholds_sigma_band_rms_meg.append(threshold_sigma_band_rms)
    not_outliers_sigma_band = sigma_band_trials_rms <= threshold_sigma_band_rms

    sigma_burst_trials_rms = np.sqrt(np.mean(MEG_sigma_burst_trials[i]**2, axis=0))

    sigma_burst_rms_q25 = scipy.stats.scoreatpercentile(sigma_burst_trials_rms, 25)
    sigma_burst_rms_q50 = np.median(sigma_burst_trials_rms)
    sigma_burst_rms_q75 = scipy.stats.scoreatpercentile(sigma_burst_trials_rms, 75)
    sigma_burst_iqr = sigma_burst_rms_q75 - sigma_burst_rms_q25

    # set a high threshold to remove only very outliers
    threshold_sigma_burst_rms = sigma_burst_rms_q75 + 3*sigma_burst_iqr
    thresholds_sigma_burst_rms_meg.append(threshold_sigma_burst_rms)
    not_outliers_sigma_burst = sigma_burst_trials_rms <= threshold_sigma_burst_rms

    not_outliers_meg.append(not_outliers_sigma_band*not_outliers_sigma_burst)

not_outliers_meg = np.mean(not_outliers_meg, axis=0) > 0.5

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i in range(len(MEG_sigma_band_whole_trials)):

    data_y = np.sqrt(np.mean(MEG_sigma_band_whole_trials[i]**2, axis=0))
    outliers_threshold = thresholds_sigma_band_rms_meg[i]
    plt.plot(np.clip(data_y, 0, 60), linewidth=1, label='ch%d, n=%d' % (i, np.sum(data_y > outliers_threshold)), color=colors[i])
    plt.axhline(outliers_threshold, linewidth=1, alpha=0.5, color=colors[i])

plt_header('MEG sigma band whole trials rms, outliers rejection candidates')
plt.xlabel('trial number')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i in range(len(MEG_sigma_burst_trials)):

    data_y = np.sqrt(np.mean(MEG_sigma_burst_trials[i]**2, axis=0))
    outliers_threshold = thresholds_sigma_burst_rms_meg[i]
    plt.plot(np.clip(data_y, 0, 60), linewidth=1, label='ch%d, n=%d' % (i, np.sum(data_y > outliers_threshold)), color=colors[i])
    plt.axhline(outliers_threshold, linewidth=1, alpha=0.5, color=colors[i])

plt_header('MEG sigma burst trials rms, outliers rejection candidates')
plt.xlabel('trial number')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

plt_header('MEG outliers rejection')
plt.plot(not_outliers_meg, linewidth=1, label='MEG, n=%d' % np.sum(not_outliers_meg == False))
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt.legend()
plt_show_save_fig()

# %%
# extract EEG trials to remove outliers

whole_trial_len = int(np.round(np.mean(np.diff(all_stimuli_eeg))))
whole_trial_win_samples = [0,whole_trial_len]
whole_trial_t = (np.arange(whole_trial_win_samples[0], whole_trial_win_samples[1], 1)/float(eeg_srate)*1000)

EEG_whole_trials = meet.epochEEG(EEG_data, marker_eeg, whole_trial_win_samples)
EEG_sigma_band_whole_trials = meet.epochEEG(eeg_sigma_band_data, marker_eeg, whole_trial_win_samples)

sigma_win_ms = [10, 35]
sigma_win_samples = np.round(np.array(sigma_win_ms)/1000.*eeg_srate).astype(int)
EEG_sigma_burst_trials = meet.epochEEG(eeg_sigma_band_data, marker_eeg, sigma_win_samples)

# %%
# calculate EEG sigma band and sigma burst trials rms and percentiles to remove outliers

not_outliers_eeg = []
thresholds_sigma_band_rms_eeg = []
thresholds_sigma_burst_rms_eeg = []

# calculate sigma band whole trials rms and percentiles to remove outliers

for i in range(len(EEG_sigma_band_whole_trials)):

    sigma_band_trials_rms = np.sqrt(np.mean(EEG_sigma_band_whole_trials[i]**2, axis=0))

    sigma_band_rms_q25 = scipy.stats.scoreatpercentile(sigma_band_trials_rms, 25)
    sigma_band_rms_q50 = np.median(sigma_band_trials_rms)
    sigma_band_rms_q75 = scipy.stats.scoreatpercentile(sigma_band_trials_rms, 75)
    sigma_band_rms_iqr = sigma_band_rms_q75 - sigma_band_rms_q25

    # set a high threshold to remove only very outliers
    threshold_sigma_band_rms = sigma_band_rms_q75 + 3*sigma_band_rms_iqr
    thresholds_sigma_band_rms_eeg.append(threshold_sigma_band_rms)
    not_outliers_sigma_band = sigma_band_trials_rms <= threshold_sigma_band_rms

    sigma_burst_trials_rms = np.sqrt(np.mean(EEG_sigma_burst_trials[i]**2, axis=0))

    sigma_burst_rms_q25 = scipy.stats.scoreatpercentile(sigma_burst_trials_rms, 25)
    sigma_burst_rms_q50 = np.median(sigma_burst_trials_rms)
    sigma_burst_rms_q75 = scipy.stats.scoreatpercentile(sigma_burst_trials_rms, 75)
    sigma_burst_iqr = sigma_burst_rms_q75 - sigma_burst_rms_q25

    # set a high threshold to remove only very outliers
    threshold_sigma_burst_rms = sigma_burst_rms_q75 + 3*sigma_burst_iqr
    thresholds_sigma_burst_rms_eeg.append(threshold_sigma_burst_rms)
    not_outliers_sigma_burst = sigma_burst_trials_rms <= threshold_sigma_burst_rms

    not_outliers_eeg.append(not_outliers_sigma_band*not_outliers_sigma_burst)

not_outliers_eeg = np.mean(not_outliers_eeg, axis=0) > 0.5

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i in range(len(EEG_sigma_band_whole_trials)):

    data_y = np.sqrt(np.mean(EEG_sigma_band_whole_trials[i]**2, axis=0))
    outliers_threshold = thresholds_sigma_band_rms_eeg[i]
    plt.plot(np.clip(data_y, 0, 3), linewidth=1, label='ch%d, n=%d' % (i, np.sum(data_y > outliers_threshold)), color=colors[i])
    plt.axhline(outliers_threshold, linewidth=1, alpha=0.5, color=colors[i])

plt_header('EEG sigma band whole trials rms, outliers rejection candidates')
plt.xlabel('trial number')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

for i in range(len(EEG_sigma_burst_trials)):

    data_y = np.sqrt(np.mean(EEG_sigma_burst_trials[i]**2, axis=0))
    outliers_threshold = thresholds_sigma_burst_rms_eeg[i]
    plt.plot(np.clip(data_y, 0, 3), linewidth=1, label='ch%d, n=%d' % (i, np.sum(data_y > outliers_threshold)), color=colors[i])
    plt.axhline(outliers_threshold, linewidth=1, alpha=0.5, color=colors[i])

plt_header('EEG sigma burst trials rms, outliers rejection candidates')
plt.xlabel('trial number')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

plt_header('MEG and EEG outliers rejection')
plt.plot(not_outliers_meg, linewidth=1, label='MEG, n=%d' % np.sum(not_outliers_meg == False))
plt.plot(not_outliers_eeg, linewidth=1, label='EEG, n=%d' % np.sum(not_outliers_eeg == False))
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt.legend()
plt_show_save_fig()

# %%
# final outliers estimation

not_outliers = not_outliers_meg*not_outliers_eeg
not_outliers = np.convolve(not_outliers, np.ones(5)/5, mode='same')
not_outliers = (not_outliers > 0.999)

not_outliers_arr = []
temp_arr = []

last_value = False
for i in range(len(not_outliers)):
    if(not_outliers[i] == False and last_value == True):
        not_outliers_arr.append(temp_arr)
        temp_arr = []

    if(not_outliers[i] == True):
        temp_arr.append(i)
    
    last_value = not_outliers[i]

for i in range(len(not_outliers_arr)):
    if(len(not_outliers_arr[i]) < 20):
        not_outliers[not_outliers_arr[i]] = False

plt_header('Final outliers rejection, trials rejected = %d, trials remained = %d' % (np.sum(not_outliers == False), np.sum(not_outliers)))
plt.plot(not_outliers)
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt_show_save_fig()

# %%
# reject outliers

MEG_whole_trials = MEG_whole_trials[:,:,not_outliers]
MEG_sigma_band_whole_trials = MEG_sigma_band_whole_trials[:,:,not_outliers]
MEG_sigma_burst_trials = MEG_sigma_burst_trials[:,:,not_outliers]

EEG_whole_trials = EEG_whole_trials[:,:,not_outliers]
EEG_sigma_band_whole_trials = EEG_sigma_band_whole_trials[:,:,not_outliers]
EEG_sigma_burst_trials = EEG_sigma_burst_trials[:,:,not_outliers]

print(MEG_whole_trials.shape)
print(MEG_sigma_band_whole_trials.shape)
print(MEG_sigma_burst_trials.shape)

print(EEG_whole_trials.shape)
print(EEG_sigma_band_whole_trials.shape)
print(EEG_sigma_burst_trials.shape)

# %%
# concatenate trials without outliers

MEG_data_no_outliers = np.concatenate(MEG_whole_trials.T).T
meg_all_stimuli_no_outliers = (np.arange(MEG_whole_trials.shape[-1])*MEG_whole_trials.shape[-2])
meg_marker_no_outliers = meg_all_stimuli_no_outliers[1:]
MEG_data_no_outliers = meet.interpolateEEG(MEG_data_no_outliers, meg_marker_no_outliers, interpolate_win_meg)

EEG_data_no_outliers = np.concatenate(EEG_whole_trials.T).T
eeg_all_stimuli_no_outliers = (np.arange(EEG_whole_trials.shape[-1])*EEG_whole_trials.shape[-2])
eeg_marker_no_outliers = eeg_all_stimuli_no_outliers[1:]
EEG_data_no_outliers = meet.interpolateEEG(EEG_data_no_outliers, eeg_marker_no_outliers, interpolate_win_eeg)

# %%
# plot asd of MEG data with outliers removed

plt_header('ASD of preprocessed MEG signal')

nfft = 2**(int(np.log2(meg_srate))-2)

data = MEG_data[0]
xf, yf = asd(data, nfft, meg_srate)
plt.plot(xf, yf, label='ch0 with outliers')

data = MEG_data_no_outliers[0]
xf, yf = asd(data, nfft, meg_srate)
plt.plot(xf, yf, label='ch0 outliers removed')

plt.xscale('log')
plt.yscale('log')
plt.xlim((1,meg_srate//2))
plt.ylim((0.1, 1000))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % meg_asd_unit)
plt.legend()
plt.grid(True)
plt_show_save_fig()

# %%
# plot asd of EEG data with outliers removed

plt_header('ASD of preprocessed EEG signal, outliers removed')

nfft = 2**(int(np.log2(eeg_srate))-2)

for i in range(len(EEG_data_no_outliers)):
    data = EEG_data_no_outliers[i]
    xf, yf = asd(data, nfft, eeg_srate)
    plt.plot(xf, yf, label='channel %d' % i)

plt.xscale('log')
plt.yscale('log')
plt.xlim((1,eeg_srate//2))
plt.ylim((0.001, 10))
plt.xlabel('f [Hz]')
plt.ylabel('Amplitude Spectral Density %s' % eeg_asd_unit)
plt.legend()
plt.grid(True)
plt_show_save_fig()

# %%
# use trials without outliers as default

MEG_data = MEG_data_no_outliers
all_stimuli_meg = meg_all_stimuli_no_outliers
marker_meg = meg_marker_no_outliers

EEG_data = EEG_data_no_outliers
all_stimuli_eeg = eeg_all_stimuli_no_outliers
marker_eeg = eeg_marker_no_outliers

# %%
# plot MEG broadband evoked response

MEG_trials_averaged = np.mean(MEG_whole_trials, axis=-1)

data = MEG_trials_averaged[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('MEG ch0 average of broadband trials, n = %d' % len(MEG_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.ylim((-600, 600))
plt_show_save_fig()

data = MEG_trials_averaged[0][:meg_srate//10]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('MEG ch0 average of broadband trials, n = %d' % len(MEG_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.ylim((-600, 600))
plt_show_save_fig()

# %%
# plot MEG sigma burst (high-frequency somatosensory evoked response)

MEG_sigma_band_trials_averaged = np.mean(MEG_sigma_band_whole_trials, axis=-1)

data = MEG_sigma_band_trials_averaged[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('MEG ch0 average of sigma burst trials, n = %d' % len(MEG_sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.ylim((-25, 25))
plt_show_save_fig()

data = MEG_sigma_band_trials_averaged[0][:meg_srate//10]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('MEG ch0 average of sigma burst trials, n = %d' % len(MEG_sigma_band_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.ylim((-25, 25))
plt_show_save_fig()

# %%
# plot sigma burst single trials stack on each other

plt.rcParams['image.cmap'] = 'coolwarm'

plt_header('MEG ch0 sigma burst, all trials')
data = MEG_sigma_band_whole_trials[0][:meg_srate//10,]
data_x = np.linspace(0, 1000*(len(data)-1)/meg_srate, len(data))
plt.pcolormesh(data_x, np.arange(len(data.T)), data.T, vmin=-30, vmax=30)
plt.ylabel('Trial number')
plt.xlabel('Time [s]')
clb = plt.colorbar(extend='both')
clb.set_label(meg_unit)
plt_show_save_fig()

start_trial = 400
end_trial = 600
plt_header('MEG ch0 sigma burst, exemplary trials')
data = MEG_sigma_band_whole_trials[0][:meg_srate//10,start_trial:end_trial+1]
data_x = np.linspace(0, 1000*(len(data)-1)/meg_srate, len(data))

plt.pcolormesh(data_x, np.arange(start_trial, end_trial+1), data.T, vmin=-30, vmax=30)
plt.ylabel('Trial number')
plt.xlabel('Time [s]')
clb = plt.colorbar(extend='both')
clb.set_label(meg_unit)
plt_show_save_fig()

# %%
# plot EEG and MEG broadband evoked response

EEG_trials_averaged = np.mean(EEG_whole_trials, axis=-1)

fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('Average of broadband trials', use_suptitle=True, fontsize=14)
cnt = 0
for i in range(3):
    for j in range(3):

        data_y = EEG_trials_averaged[cnt][:eeg_srate//10,]
        data_x = np.linspace(0, (len(data_y)-1)/eeg_srate, len(data_y))*1000
        axs[i][j].plot(data_x, data_y, linewidth=1, label='EEG ch%d' % cnt)

        ylim = np.ceil(np.max(np.abs(EEG_trials_averaged)[:, :eeg_srate//10,]))
        axs[i][j].set_ylim((-ylim, ylim))

        axs[i][j].set_ylabel(eeg_unit)
        axs[i][j].legend()
        if(i==2):
            axs[i][j].set_xlabel('t [ms]')

        cnt+=1

        if(cnt >= 8):

            data_y = MEG_trials_averaged[0][:meg_srate//10,]
            data_x = np.linspace(0, (len(data_y)-1)/meg_srate, len(data_y))*1000
            axs[2][2].plot(data_x, data_y, linewidth=1, label='MEG ch%d' % 0)

            ylim = (1+np.max(np.abs(MEG_trials_averaged)[:, :meg_srate//10,])//100)*100
            axs[2][2].set_ylim((-ylim, ylim))

            axs[2][2].set_ylabel(meg_unit)
            axs[2][2].legend()
            axs[2][2].set_xlabel('t [ms]')

            break


fig.tight_layout()
plt_show_save_fig()

# %%
# plot EEG and MEG sigma burst (high-frequency somatosensory evoked response)

EEG_sigma_band_trials_averaged = np.mean(EEG_sigma_band_whole_trials, axis=-1)

fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('Average of sigma burst trials', use_suptitle=True, fontsize=14)
cnt = 0
for i in range(3):
    for j in range(3):

        data_y = EEG_sigma_band_trials_averaged[cnt][:eeg_srate//10,]
        data_x = np.linspace(0, (len(data_y)-1)/eeg_srate, len(data_y))*1000
        axs[i][j].plot(data_x, data_y, linewidth=1, label='EEG ch%d' % cnt)

        ylim = (1+np.max(100*np.abs(EEG_sigma_band_trials_averaged)[:, :eeg_srate//10,])//5)/20
        axs[i][j].set_ylim((-ylim, ylim))

        axs[i][j].set_ylabel(eeg_unit)
        axs[i][j].legend()
        if(i==2):
            axs[i][j].set_xlabel('t [ms]')

        cnt+=1

        if(cnt >= 8):

            data_y = MEG_sigma_band_trials_averaged[0][:meg_srate//10,]
            data_x = np.linspace(0, (len(data_y)-1)/meg_srate, len(data_y))*1000
            axs[2][2].plot(data_x, data_y, linewidth=1, label='MEG ch%d' % 0)

            ylim = (1+np.max(np.abs(MEG_sigma_band_trials_averaged)[:, :meg_srate//10,])//5)*5
            axs[2][2].set_ylim((-ylim, ylim))

            axs[2][2].set_ylabel(meg_unit)
            axs[2][2].legend()
            axs[2][2].set_xlabel('t [ms]')

            break


fig.tight_layout()
plt_show_save_fig()

# %%
# interpolate EEG to MEG number of samples

total_time = len(MEG_data[0])/meg_srate
meg_timepoints = np.linspace(0, total_time, len(MEG_data[0]), endpoint=False)
eeg_timepoints = np.linspace(0, total_time, len(EEG_data[0]), endpoint=False)

interp = scipy.interpolate.interp1d(eeg_timepoints, EEG_data, axis=1, fill_value='extrapolate')
EEG_data_interpolated = interp(meg_timepoints)

# %%
# save the preprocessed data

stimulus_new = np.zeros(MEG_data.shape[-1])
stimulus_new[marker_meg] = 1.0
out_data = np.concatenate((EEG_data_interpolated, MEG_data, [stimulus_new]))
np.save(os.path.join(data_output_folder, subject_name+'.npy'), out_data)


