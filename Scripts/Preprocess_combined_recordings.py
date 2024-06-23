# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import meet
import helper_scripts.helper_functions as helper_functions
import matplotlib.mlab as mlab
from scipy import signal as sig
import tables

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams["savefig.dpi"] = 600

# %%
# select subject
# SMa - subject a with MEG data
# SEb - subject b with EEG data

subject = 'S1'
data_folder = '../Data'
results_main_folder = '../Results'

additional_plot_title = ''

# use brownian noise to make a reference model
if(subject.endswith('BN')):
    use_brown_noise = True
else:
    use_brown_noise = False

if(use_brown_noise):
    results_folder = os.path.join(results_main_folder, subject+'_MEEG_comb_brown_noise')
    subject_name = subject+'_MEEG_comb_brown_noise'
else:
    results_folder = os.path.join(results_main_folder, subject+'_MEEG_comb')
    subject_name = subject+'_MEEG_comb'

meg_srate = 20000
meg_unit = 'B [fT]'
meg_asd_unit = '[fT/√HZ]'

eeg_srate = 10000
eeg_unit = 'U [μV]'
eeg_asd_unit = '[μV/√HZ]'

filename = '%s_MEG_comb_stim.dat' % subject
raw_meg_data_stim = helper_functions.readMEG(os.path.join(data_folder, subject, filename), s_rate=meg_srate, num_chans=2)
filename = '%s_MEG_comb_relax.dat' % subject
raw_meg_data_relax = helper_functions.readMEG(os.path.join(data_folder, subject, filename), s_rate=meg_srate, num_chans=2)

filename = '%s_EEG_comb_stim.hdf5' % subject
h5file_eeg_stim = tables.open_file(os.path.join(data_folder, subject, filename), mode="r", title="%s_eeg_stim" % subject)
raw_eeg_data_stim = np.array(h5file_eeg_stim.root.EEG_data).T
filename = '%s_EEG_comb_relax.hdf5' % subject
h5file_eeg_rest = tables.open_file(os.path.join(data_folder, subject, filename), mode="r", title="%s_eeg_rest" % subject)
raw_eeg_data_relax = np.array(h5file_eeg_rest.root.EEG_data).T

#raw_meg_data_emg = helper_functions.readMEG(filepath_emg+'.dat', s_rate=meg_srate, num_chans=2)
#h5file_eeg_emg = tables.open_file(filepath_emg+'.hdf5', mode="r", title="%s_eeg_emg" % subject)
#raw_eeg_data_emg = np.array(h5file_eeg_emg.root.EEG_data).T

results_path = results_folder

# %%
def min_max_normalize_sig(signal):

    signal_normalized = signal - np.min(signal)
    signal_normalized /= np.max(signal_normalized)

    return signal_normalized

# %%
MEG_data = -raw_meg_data_stim[:-1]
MEG_stimulus_data = raw_meg_data_stim[-1]
MEG_relax_data = -raw_meg_data_relax[:-1]
#MEG_emg_data = -raw_meg_data_emg[:-1]

EEG_data = raw_eeg_data_stim[:-1, 40:]
EEG_stimulus_data = raw_eeg_data_stim[-1][:-40]
EEG_relax_data = raw_eeg_data_relax[:-1, 40:]
#EEG_emg_data = raw_eeg_data_emg[:-1, 40:]

# %%
'''
if(use_brown_noise):
    # use brownian noise as a reference model
    import colorednoise as cn
    if(measurement_type == 'EEG'):
        XEG_data = 1000*cn.powerlaw_psd_gaussian(2, len(XEG_data))
    else:
        XEG_data = 10000*cn.powerlaw_psd_gaussian(2, len(XEG_data))
'''

# %%
def plt_header(main_title='', use_suptitle = False, fontsize=12):

    title = 'Subject '+ subject
    title += '\n'+main_title+additional_plot_title

    if(use_suptitle):
        plt.suptitle(title, fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)

# %%
def plt_show_save_fig(fig_name=None):

    if(fig_name):
        fig_name += '.png'
    else:
        plt_show_save_fig.counter += 1
        fig_name = 'Fig%02d.png' % plt_show_save_fig.counter

    print('--------------------\n'+fig_name)
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, fig_name), bbox_inches='tight')
    plt.show()

plt_show_save_fig.counter = 0

# %%
# get the stimuli positions
all_stimuli_meg = ((MEG_stimulus_data[1:]>162500) & (MEG_stimulus_data[:-1]<162500)).nonzero()[0]
all_stimuli_eeg = ((EEG_stimulus_data[1:]>0.5) & (EEG_stimulus_data[:-1]<0.5)).nonzero()[0]

# get the marker, omit the first and last (avoid edge effects)
marker_meg = all_stimuli_meg[10:-10]
marker_eeg = all_stimuli_eeg[10:-10]
marker_eeg = marker_eeg[:len(marker_meg)]

# %%
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
interpolate_win_ms_meg = [-2, 3]
interpolate_win_meg = np.round(np.array(interpolate_win_ms_meg) / 1000. * meg_srate).astype(int)
MEG_preprocessed_data = meet.interpolateEEG(MEG_data.copy(), all_stimuli_meg, interpolate_win_meg)

interpolate_win_ms_eeg = [-5, 8]
interpolate_win_eeg = np.round(np.array(interpolate_win_ms_eeg) / 1000. * eeg_srate).astype(int)
EEG_preprocessed_data = meet.interpolateEEG(EEG_data.copy(), all_stimuli_eeg, interpolate_win_eeg)

# %%
data = MEG_data[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='raw signal')

data = MEG_preprocessed_data[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='preprocessed signal')

plt_header('Removing stimulus artifacts from MEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

# %%
data = MEG_data[0][:meg_srate]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='raw signal')

data = MEG_preprocessed_data[0][:meg_srate]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='preprocessed signal')

plt_header('Removing stimulus artifacts from MEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

# %%
start_time = marker_meg[0]/meg_srate - 0.01 
end_time = marker_meg[0]/meg_srate + 0.01

data = MEG_data[0][int(start_time*meg_srate):int(end_time*meg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='raw signal')

data = MEG_preprocessed_data[0][int(start_time*meg_srate):int(end_time*meg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='preprocessed signal')

plt_header('Removing stimulus artifacts from MEG ch0 data')
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.legend()
plt_show_save_fig()

# %%
data = EEG_data[0]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='raw signal')

data = EEG_preprocessed_data[0]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='preprocessed signal')

plt_header('Removing stimuli artifacts from EEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

# %%
data = EEG_data[0][:eeg_srate]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='raw signal')

data = EEG_preprocessed_data[0][:eeg_srate]
data_x = np.linspace(0, (len(data)-1)/eeg_srate, len(data))
data_y = data
plt.plot(data_x, data_y, label='preprocessed signal')

plt_header('Removing stimulus artifacts from EEG ch0 data')
plt.xlabel('t [s]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

# %%
start_time = marker_eeg[0]/eeg_srate - 0.01 
end_time = marker_eeg[0]/eeg_srate + 0.01

data = EEG_data[0][int(start_time*eeg_srate):int(end_time*eeg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='raw signal')

data = EEG_preprocessed_data[0][int(start_time*eeg_srate):int(end_time*eeg_srate)]
data_x = np.linspace(start_time*1000, end_time*1000, len(data))
data_y = data
plt.plot(data_x, data_y, label='preprocessed signal')

plt_header('Removing stimulus artifacts from EEG ch0 data')
plt.xlabel('t [ms]')
plt.ylabel(eeg_unit)
plt.legend()
plt_show_save_fig()

# %%
# apply fc=1Hz hp filter to remove DC component
sos = sig.butter(2, 1, 'highpass', fs=meg_srate, output='sos')
MEG_preprocessed_hp_data = sig.sosfiltfilt(sos, MEG_preprocessed_data)

# apply fc=1Hz hp filter to remove DC component
sos = sig.butter(2, 1, 'highpass', fs=eeg_srate, output='sos')
EEG_preprocessed_hp_data = sig.sosfiltfilt(sos, EEG_preprocessed_data)

# %%
# amplitude spectral density
def asd(data, nfft, srate):

    yf, xf = mlab.psd(x=data, NFFT=nfft, Fs=srate, window=sig.windows.hann(nfft), noverlap=nfft//2)
    
    return xf, np.sqrt(yf)

# %%
plt_header('ASD of preprocessed MEG signal')

nfft = 2**(int(np.log2(meg_srate))-2)

for i in range(len(MEG_preprocessed_hp_data)):
    data = MEG_preprocessed_hp_data[i]
    xf2, yf2 = asd(data, nfft, meg_srate)
    plt.plot(xf2, yf2, label='ch%d' % i)

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
plt_header('ASD of preprocessed EEG signal')

nfft = 2**(int(np.log2(eeg_srate))-2)

for i in range(len(EEG_preprocessed_hp_data)):
    data = EEG_preprocessed_hp_data[i]
    xf2, yf2 = asd(data, nfft, eeg_srate)
    plt.plot(xf2, yf2, label='ch%d' % i)

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
# apply 450Hz-850Hz band-pass filter for sigma band extraction and removing outliers
lfreq_sigma = 450
rfreq_sigma = 850
sigma_freq_range_str = '%sHz-%sHz' % (lfreq_sigma, rfreq_sigma)

meg_sigma_band_data = meet.iir.butterworth(MEG_preprocessed_hp_data, fs=(lfreq_sigma-50, rfreq_sigma+50),
                                        fp=(lfreq_sigma, rfreq_sigma), s_rate=meg_srate)

eeg_sigma_band_data = meet.iir.butterworth(EEG_preprocessed_hp_data, fs=(lfreq_sigma-50, rfreq_sigma+50),
                                        fp=(lfreq_sigma, rfreq_sigma), s_rate=eeg_srate)

# %%
# extract meg trials to remove outliers

whole_trial_len = int(np.round(np.mean(np.diff(all_stimuli_meg))))
whole_trial_win_samples = [0,whole_trial_len]
whole_trial_t = (np.arange(whole_trial_win_samples[0], whole_trial_win_samples[1], 1)/float(meg_srate)*1000)

MEG_preprocessed_hp_whole_trials = meet.epochEEG(MEG_preprocessed_hp_data, marker_meg, whole_trial_win_samples)
MEG_sigma_band_whole_trials = meet.epochEEG(meg_sigma_band_data, marker_meg, whole_trial_win_samples)

sigma_win_ms = [10, 35]
sigma_win_samples = np.round(np.array(sigma_win_ms)/1000.*meg_srate).astype(int)
MEG_sigma_burst_trials = meet.epochEEG(meg_sigma_band_data, marker_meg, sigma_win_samples)

# %%
not_outliers_meg = []
thresholds_sigma_band_rms_meg = []
thresholds_sigma_burst_rms_meg = []

# calculate sigma band whole trials rms and percentiles to remove outliers

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

# %%
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

# %%
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

# %%
plt_header('MEG outliers rejection')
plt.plot(not_outliers_meg, linewidth=1, label='MEG, n=%d' % np.sum(not_outliers_meg == False))
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt.legend()
plt_show_save_fig()

# %%
# extract eeg trials to remove outliers

whole_trial_len = int(np.round(np.mean(np.diff(all_stimuli_eeg))))
whole_trial_win_samples = [0,whole_trial_len]
whole_trial_t = (np.arange(whole_trial_win_samples[0], whole_trial_win_samples[1], 1)/float(eeg_srate)*1000)

EEG_preprocessed_hp_whole_trials = meet.epochEEG(EEG_preprocessed_hp_data, marker_eeg, whole_trial_win_samples)
EEG_sigma_band_whole_trials = meet.epochEEG(eeg_sigma_band_data, marker_eeg, whole_trial_win_samples)

sigma_win_ms = [10, 35]
sigma_win_samples = np.round(np.array(sigma_win_ms)/1000.*eeg_srate).astype(int)
EEG_sigma_burst_trials = meet.epochEEG(eeg_sigma_band_data, marker_eeg, sigma_win_samples)

# %%
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

# %%
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

# %%
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

# %%
plt_header('MEG and EEG outliers rejection')
plt.plot(not_outliers_meg, linewidth=1, label='MEG, n=%d' % np.sum(not_outliers_meg == False))
plt.plot(not_outliers_eeg, linewidth=1, label='EEG, n=%d' % np.sum(not_outliers_eeg == False))
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt.legend()
plt_show_save_fig()

# %%
not_outliers = not_outliers_meg*not_outliers_eeg
not_outliers = np.convolve(not_outliers, np.ones(5)/5, mode='same')
not_outliers = (not_outliers > 0.999)

# %%
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

# %%
plt_header('Final outliers rejection, trials rejected = %d, trials remained = %d' % (np.sum(not_outliers == False), np.sum(not_outliers)))
plt.plot(not_outliers)
plt.xlabel('trial number')
plt.ylabel('not outlier')
plt_show_save_fig()

# %%
MEG_preprocessed_hp_whole_trials = MEG_preprocessed_hp_whole_trials[:,:,not_outliers]
MEG_sigma_band_whole_trials = MEG_sigma_band_whole_trials[:,:,not_outliers]
MEG_sigma_burst_trials = MEG_sigma_burst_trials[:,:,not_outliers]

EEG_preprocessed_hp_whole_trials = EEG_preprocessed_hp_whole_trials[:,:,not_outliers]
EEG_sigma_band_whole_trials = EEG_sigma_band_whole_trials[:,:,not_outliers]
EEG_sigma_burst_trials = EEG_sigma_burst_trials[:,:,not_outliers]

# %%
print(MEG_preprocessed_hp_whole_trials.shape)
print(MEG_sigma_band_whole_trials.shape)
print(MEG_sigma_burst_trials.shape)

print(EEG_preprocessed_hp_whole_trials.shape)
print(EEG_sigma_band_whole_trials.shape)
print(EEG_sigma_burst_trials.shape)

# %%
MEG_preprocessed_hp_data_no_outliers = np.concatenate(MEG_preprocessed_hp_whole_trials.T).T
meg_all_stimuli_no_outliers = (np.arange(MEG_preprocessed_hp_whole_trials.shape[-1])*MEG_preprocessed_hp_whole_trials.shape[-2])
meg_marker_no_outliers = meg_all_stimuli_no_outliers[1:]
MEG_preprocessed_hp_data_no_outliers = meet.interpolateEEG(MEG_preprocessed_hp_data_no_outliers, meg_marker_no_outliers, interpolate_win_meg)

EEG_preprocessed_hp_data_no_outliers = np.concatenate(EEG_preprocessed_hp_whole_trials.T).T
eeg_all_stimuli_no_outliers = (np.arange(EEG_preprocessed_hp_whole_trials.shape[-1])*EEG_preprocessed_hp_whole_trials.shape[-2])
eeg_marker_no_outliers = eeg_all_stimuli_no_outliers[1:]
EEG_preprocessed_hp_data_no_outliers = meet.interpolateEEG(EEG_preprocessed_hp_data_no_outliers, eeg_marker_no_outliers, interpolate_win_eeg)

# %%
plt_header('ASD of preprocessed MEG signal')

nfft = 2**(int(np.log2(meg_srate))-2)

data = MEG_preprocessed_hp_data[0]
xf2, yf2 = asd(data, nfft, meg_srate)
plt.plot(xf2, yf2, label='ch0 with outliers')

data = MEG_preprocessed_hp_data_no_outliers[0]
xf2, yf2 = asd(data, nfft, meg_srate)
plt.plot(xf2, yf2, label='ch0 outliers removed')

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
plt_header('ASD of preprocessed EEG signal, outliers removed')

nfft = 2**(int(np.log2(eeg_srate))-2)

for i in range(len(EEG_preprocessed_hp_data_no_outliers)):
    data = EEG_preprocessed_hp_data_no_outliers[i]
    xf2, yf2 = asd(data, nfft, eeg_srate)
    plt.plot(xf2, yf2, label='channel %d' % i)

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
MEG_preprocessed_hp_data = MEG_preprocessed_hp_data_no_outliers
all_stimuli_meg = meg_all_stimuli_no_outliers
marker_meg = meg_marker_no_outliers

EEG_preprocessed_hp_data = EEG_preprocessed_hp_data_no_outliers
all_stimuli_eeg = eeg_all_stimuli_no_outliers
marker_eeg = eeg_marker_no_outliers

# %%
MEG_trials_averaged = np.mean(MEG_preprocessed_hp_whole_trials, axis=-1)

data = MEG_trials_averaged[0]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('MEG ch0 average of broadband trials, n = %d' % len(MEG_preprocessed_hp_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.ylim((-600, 600))
plt_show_save_fig()

data = MEG_trials_averaged[0][:meg_srate//10]
data_x = np.linspace(0, (len(data)-1)/meg_srate, len(data))*1000
data_y = data
plt.plot(data_x, data_y)
plt_header('MEG ch0 average of broadband trials, n = %d' % len(MEG_preprocessed_hp_whole_trials.T))
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.ylim((-600, 600))
plt_show_save_fig()

# %%
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
plt.rcParams['image.cmap'] = 'coolwarm'

plt_header('MEG ch0 sigma burst, all trials')
data = MEG_sigma_band_whole_trials[0][:meg_srate//10,]
data_x = np.linspace(0, 1000*(len(data)-1)/meg_srate, len(data))
plt.pcolormesh(data_x, np.arange(len(data.T)), data.T, vmin=-25, vmax=25)
plt.ylabel('Trial number')
plt.xlabel('Time [s]')
clb = plt.colorbar()
clb.set_label(meg_unit)
plt_show_save_fig()

start_trial = 900
end_trial = 1101
plt_header('MEG ch0 sigma burst, exemplary trials')
data = MEG_sigma_band_whole_trials[0][:meg_srate//10,start_trial:end_trial]
data_x = np.linspace(0, 1000*(len(data)-1)/meg_srate, len(data))

plt.pcolormesh(data_x, np.arange(start_trial, end_trial), data.T, vmin=-25, vmax=25)
plt.ylabel('Trial number')
plt.xlabel('Time [s]')
clb = plt.colorbar()
clb.set_label(meg_unit)
plt_show_save_fig()

# %%
EEG_trials_averaged = np.mean(EEG_preprocessed_hp_whole_trials, axis=-1)

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
stimulus_meg_new = np.zeros(MEG_preprocessed_hp_data.shape[-1])
stimulus_meg_new[marker_meg] = 1.0
meg_out_data = np.concatenate((MEG_preprocessed_hp_data, [stimulus_meg_new]))
np.save(os.path.join(results_path, subject+'_meg_cleaned_data.npy'), meg_out_data)

stimulus_eeg_new = np.zeros(EEG_preprocessed_hp_data.shape[-1])
stimulus_eeg_new[marker_eeg] = 1.0
eeg_out_data = np.concatenate((EEG_preprocessed_hp_data, [stimulus_eeg_new]))
np.save(os.path.join(results_path, subject+'_eeg_cleaned_data.npy'), eeg_out_data)


