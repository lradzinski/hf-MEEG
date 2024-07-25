# %%
"""
Spatial_filters.py
Lukasz Radzinski
Charite Neurophysics Group, Berlin
Script for preprocessing combined
MEEG recordings 
"""

# %%
import os
import meet
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal as sig
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# set global parameters for plots
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['savefig.dpi'] = 600

# %%
# select subject [S1-S4]
subject = 'S1'
date = '2018-03-19'

data_input_folder = '../Data/cleaned_data'
plots_output_folder = '../Results/spatial_filters'

additional_plot_title = ''

subject_name = subject+'_MEEG_comb_sync_stim'
data_input_folder = os.path.join(data_input_folder, date, subject)
plots_output_folder = os.path.join(plots_output_folder, subject+'_'+date)

srate = 20000

meg_unit = 'B [fT]'
meg_asd_unit = '[fT/√HZ]'

eeg_unit = 'U [μV]'
eeg_asd_unit = '[μV/√HZ]'

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
        fig_name = subject+'_fig%02d.png' % plt_show_save_fig.counter

    print('--------------------\n'+fig_name)
    os.makedirs(plots_output_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_output_folder, fig_name), bbox_inches='tight')
    plt.show()

plt_show_save_fig.counter = 0

# %%
# load data from files
meeg_data_file = os.path.join(data_input_folder, subject_name+'.npy')
meeg_stim_data = np.load(meeg_data_file)

channames = np.loadtxt('EEG_channels.txt', dtype='str')
channames = np.append(channames, 'MEG')
channames_aligned = []
for i in range(len(channames)):
    channames_aligned.append(channames[i].ljust(3))

# %%
# remove the marker channel
meeg_data = meeg_stim_data[:-1]
marker = meet.getMarker(meeg_stim_data[-1])

marker_multiplied = []
for i in range(1):
    marker_multiplied.append(marker)

marker_multiplied = np.concatenate(np.stack(marker_multiplied).T)

# %%
# adjust personalized frequency bands
# for proper sigma burst extraction
if(subject == 'S1'):
    sigma_lfreq = 500
    sigma_rfreq = 800
elif(subject == 'S2'):
    sigma_lfreq = 500
    sigma_rfreq = 800
elif(subject == 'S3'):
    sigma_lfreq = 600
    sigma_rfreq = 850
elif(subject == 'S4'):
    sigma_lfreq = 600
    sigma_rfreq = 850

sos = sig.butter(2, [sigma_lfreq, sigma_rfreq], 'bandpass', fs=srate, output='sos')
meeg_sigma_data = sig.sosfiltfilt(sos, meeg_data)

sigma_freq_range_str = '%sHz-%sHz' % (sigma_lfreq, sigma_rfreq)

# %%
# extract trials
short_view_win_ms = [-20,80]
short_view_win_samples = np.round(np.array(short_view_win_ms)/1000.*srate).astype(int)

meeg_broadband_short_view_trials = meet.epochEEG(meeg_data, marker, short_view_win_samples)
meeg_sigma_short_view_trials = meet.epochEEG(meeg_sigma_data, marker, short_view_win_samples)
data_x_short_view = np.linspace(short_view_win_ms[0], short_view_win_ms[1], meeg_broadband_short_view_trials.shape[-2]+1)[:-1]

# %%
# estimate sigma burst boundaries
sigma_pow = np.mean(meeg_sigma_short_view_trials[-1], axis=-1).__abs__()**2
ref_pow = np.mean(sigma_pow[:srate*20//1000])
signal_noise_ratio = sigma_pow/ref_pow
sigma_target_win_ms = data_x_short_view[signal_noise_ratio > 50]
sigma_target_win_ms = [int(np.floor(sigma_target_win_ms[0])), int(np.ceil(sigma_target_win_ms[-1]))]
sigma_target_win_ms[0] = np.clip(sigma_target_win_ms[0], 13, 20)+2
sigma_target_win_ms[1] = np.clip(sigma_target_win_ms[1], 25, 33)-5
sigma_target_win_samples = np.round(np.array(sigma_target_win_ms)/1000.*srate).astype(int)

meeg_sigma_target_trials = meet.epochEEG(meeg_sigma_data, marker, sigma_target_win_samples)
meeg_sigma_trials_rms = np.sqrt(np.mean(meeg_sigma_target_trials**2, axis=-2))

# %%
# calculate whole noise of the trial
sigma_noise1_win_ms = [-100, sigma_target_win_ms[0]-5]
sigma_noise2_win_ms = [sigma_target_win_ms[1]+5, 200]

sigma_noise1_win_samples = np.round(np.array(sigma_noise1_win_ms)/1000.*srate).astype(int)
sigma_noise2_win_samples = np.round(np.array(sigma_noise2_win_ms)/1000.*srate).astype(int)

meeg_sigma_noise1_trials = meet.epochEEG(meeg_sigma_data, marker, sigma_noise1_win_samples)
meeg_sigma_noise2_trials = meet.epochEEG(meeg_sigma_data, marker, sigma_noise2_win_samples)
meeg_sigma_whole_noise_trials = np.concatenate((meeg_sigma_noise1_trials, meeg_sigma_noise2_trials), axis=1)
meeg_sigma_whole_noise_rms = np.sqrt(np.mean(meeg_sigma_whole_noise_trials**2, axis=-2))

# %%
# use multiplied (repeating) marker
# to achieve more trials with random noise
# which helps to achieve more stable result of CSP
# set seed for random noise extraction
# to achieve the same result in every script run
np.random.seed(0)
marker_random_noise = marker_multiplied+35*srate//1000
marker_random_noise = (marker_random_noise + np.random.random(len(marker_random_noise))*200*srate//1000).astype(int)
meeg_sigma_random_noise_multi_trials = meet.epochEEG(meeg_sigma_data, marker_random_noise, sigma_target_win_samples)
meeg_sigma_random_noise_multi_rms = np.sqrt(np.mean(meeg_sigma_random_noise_multi_trials**2, axis=-2))

meeg_sigma_target_multi_trials = meet.epochEEG(meeg_sigma_data, marker_multiplied, sigma_target_win_samples)
meeg_sigma_trials_multi_rms = np.sqrt(np.mean(meeg_sigma_target_multi_trials**2, axis=-2))

# %%
# plot EEG and MEG sigma burst (high-frequency somatosensory evoked response)

meeg_sigma_band_trials_averaged = np.mean(meeg_sigma_short_view_trials, axis=-1)

meeg_sigma_averaged_rms = np.sqrt(np.mean(np.mean(meeg_sigma_target_trials, axis=-1)**2, axis=-1))
meeg_sigma_noise_averaged_rms = np.sqrt(np.mean(np.mean(meeg_sigma_whole_noise_trials, axis=-1)**2, axis=-1))

snnr_st = np.mean(meeg_sigma_trials_multi_rms/meeg_sigma_random_noise_multi_rms, axis=-1)
snnr_er = meeg_sigma_averaged_rms/meeg_sigma_noise_averaged_rms

fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('Average of sigma burst trials, estimated boundaries = %s ms' % sigma_target_win_ms, use_suptitle=True, fontsize=14)
cnt = 0
for i in range(3):
    for j in range(3):

        data_y = meeg_sigma_band_trials_averaged[cnt]
        data_x = data_x_short_view

        if(cnt == 8):
            unit = meg_unit
            ylim = 30
        else:
            ylim = 0.30
            unit = eeg_unit

        axs[i][j].plot([], [], linewidth=1, label='%s st: %5.2f' % (channames_aligned[cnt], snnr_st[cnt]), c='tab:blue')
        axs[i][j].plot(data_x, data_y, linewidth=1, label='%s er: %5.2f' % (channames_aligned[cnt], snnr_er[cnt]))
        axs[i][j].axvline(sigma_target_win_ms[0], color='gray', linestyle=':')
        axs[i][j].axvline(sigma_target_win_ms[1], color='gray', linestyle=':')
        axs[i][j].axvline(0, color='silver')
        axs[i][j].set_ylim((-ylim, ylim))
        axs[i][j].set_ylabel(unit)
        axs[i][j].legend(prop={'family': 'DejaVu Sans Mono'}, title='MEEG ch         snnr')
        if(i==2):
            axs[i][j].set_xlabel('t [ms]')

        cnt+=1

fig.tight_layout()
plt_show_save_fig()

# %%
# plot correlations heatmap of channels
plt_header('Correlation between channels for sigma burst trials rms, t = %s ms' % sigma_target_win_ms)
sns.heatmap(np.corrcoef(meeg_sigma_trials_multi_rms), annot = True, fmt = '.2f', xticklabels=channames, yticklabels=channames,
            vmin=0, vmax=1, cmap='YlGnBu_r', cbar_kws={'label':'r coeff'}, annot_kws={'c':'black'})
plt_show_save_fig()

# %%
# plot correlations heatmap of channels
plt_header('Correlation between channels for sigma band noise trials rms, t > 35 ms')
sns.heatmap(np.corrcoef(meeg_sigma_random_noise_multi_rms), annot = True, fmt = '.2f', xticklabels=channames, yticklabels=channames,
            vmin=0, vmax=1, cmap='YlGnBu_r', cbar_kws={'label':'r coeff'}, annot_kws={'c':'black'})
plt_show_save_fig()

# %%
# plot correlations heatmap of channels
plt_header('Difference between correlations')
sns.heatmap(np.corrcoef(meeg_sigma_trials_multi_rms)-np.corrcoef(meeg_sigma_random_noise_multi_rms), annot = True, fmt = '.2f', xticklabels=channames, yticklabels=channames,
            vmin=-0.2, vmax=0.2, cmap='PuOr_r', cbar_kws={'label':'r coeff(sigma burst)  - r coeff (noise) '}, annot_kws={'c':'black'})
plt_show_save_fig()

# %%
# use spatial filters (CSP) to reconstruct source signal from EEG
sigma_eeg_sfilter, sigma_eeg_tfilter, sigma_eeg_seigvals, sigma_eeg_teigvals = meet.spatfilt.bCSTP(
meeg_sigma_target_multi_trials[:-1], meeg_sigma_random_noise_multi_trials[:-1], s_keep=1, t_keep=1, num_iter=30)

# %%
# get spatial patterns
sigma_mEeg_spattern = np.hstack([np.linalg.inv(sigma_eeg_sfilter[-1]), np.zeros([sigma_eeg_sfilter[-1].shape[-1], 1])])

# normalize spatial patterns
sigma_mEeg_spattern /= np.abs(sigma_mEeg_spattern).max(-1)[...,np.newaxis]

EEG_ref = 'Fp2'
channnames_csp = ['CSP0', 'CSP1', 'CSP2', 'CSP3', 'CSP4', 'CSP5', 'CSP6', 'CSP7', 'MEG']
coords = meet.sphere.getStandardCoordinates(list(channames[:-1]) + [EEG_ref])

# %%
# plot spatial patterns
fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('CSP spatial patterns for sigma burst detection' % sigma_target_win_ms, use_suptitle=True, fontsize=14)
cnt = 0

for i in range(3):
    for j in range(3):

        # interpolate the maps
        sigma_mEeg_potMap = meet.sphere.potMap(coords, sigma_mEeg_spattern[cnt])
        axs[i][j].set_title(channnames_csp[cnt])
        meet.sphere.addHead(axs[i][j])
        axs[i][j].pcolormesh(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], vmin=-1, vmax=1, rasterized=True, cmap='plasma')
        axs[i][j].contour(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], levels=[0], colors='w', linewidths=1)
        axs[i][j].set_xlim([-1.8, 1.8])
        axs[i][j].set_ylim([-1.1, 1.2])
        axs[i][j].axis('off')

        cnt +=1
    
        if(cnt >= 8):
            axs[2][2].axis('off')
            break
plt_show_save_fig()

# %%
# calculate spatially filtered EEG signal

eeg_sfiltered = np.dot(sigma_eeg_sfilter[-1].T, meeg_sigma_data[:-1])
eeg_sfiltered = 10*(eeg_sfiltered.T / np.std(eeg_sfiltered, axis=-1)).T

eeg_sfiltered = np.append(eeg_sfiltered, meeg_sigma_data[-1:], axis=0)
eeg_sfiltered_short_view_trials = meet.epochEEG(eeg_sfiltered, marker, short_view_win_samples)

eeg_sfiltered_target_trials = meet.epochEEG(eeg_sfiltered, marker, sigma_target_win_samples)
eeg_sfiltered_trials_rms = np.sqrt(np.mean(eeg_sfiltered_target_trials**2, axis=-2))

eeg_sfiltered_target_multi_trials = meet.epochEEG(eeg_sfiltered, marker_multiplied, sigma_target_win_samples)
eeg_sfiltered_trials_multi_rms = np.sqrt(np.mean(eeg_sfiltered_target_multi_trials**2, axis=-2))

eeg_sfiltered_noise1_trials = meet.epochEEG(eeg_sfiltered, marker, sigma_noise1_win_samples)
eeg_sfiltered_noise2_trials = meet.epochEEG(eeg_sfiltered, marker, sigma_noise2_win_samples)
eeg_sfiltered_whole_noise_trials = np.concatenate((eeg_sfiltered_noise1_trials, eeg_sfiltered_noise2_trials), axis=1)
eeg_sfiltered_whole_noise_rms = np.sqrt(np.mean(eeg_sfiltered_whole_noise_trials**2, axis=-2))

eeg_sfiltered_random_noise_multi_trials = meet.epochEEG(eeg_sfiltered, marker_random_noise, sigma_target_win_samples)
eeg_sfiltered_random_noise_multi_rms = np.sqrt(np.mean(eeg_sfiltered_random_noise_multi_trials**2, axis=-2))

# %%
# plot EEG (CSP) and MEG sigma burst (high-frequency somatosensory evoked response)

eeg_sfiltered_trials_averaged = np.mean(eeg_sfiltered_short_view_trials, axis=-1)

eeg_sfiltered_averaged_rms = np.sqrt(np.mean(np.mean(eeg_sfiltered_target_trials, axis=-1)**2, axis=-1))
eeg_sfiltered_noise_averaged_rms = np.sqrt(np.mean(np.mean(eeg_sfiltered_whole_noise_trials, axis=-1)**2, axis=-1))

snnr_st = np.mean(eeg_sfiltered_trials_multi_rms/eeg_sfiltered_random_noise_multi_rms, axis=-1)
snnr_er = eeg_sfiltered_averaged_rms/eeg_sfiltered_noise_averaged_rms

fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('Average of sigma burst trials, CSP transformation, estimated boundaries = %s ms' % sigma_target_win_ms, use_suptitle=True, fontsize=14)
cnt = 0
for i in range(3):
    for j in range(3):

        data_y = eeg_sfiltered_trials_averaged[cnt]
        data_x = data_x_short_view
        ylim = 30

        if(cnt == 8):
            unit = meg_unit
        else:
            unit = 'abstract unit'

        axs[i][j].plot([], [], linewidth=1, label='%s st: %5.2f' % (channnames_csp[cnt], snnr_st[cnt]), c='tab:blue')
        axs[i][j].plot(data_x, data_y, linewidth=1, label='%s er: %5.2f' % (channnames_csp[cnt], snnr_er[cnt]))
        axs[i][j].axvline(sigma_target_win_ms[0], color='gray', linestyle=':')
        axs[i][j].axvline(sigma_target_win_ms[1], color='gray', linestyle=':')
        axs[i][j].axvline(0, color='silver')
        axs[i][j].set_ylim((-ylim, ylim))
        axs[i][j].set_ylabel(unit)
        axs[i][j].legend(prop={'family': 'DejaVu Sans Mono'}, title='MEEG ch         snnr')
        if(i==2):
            axs[i][j].set_xlabel('t [ms]')

        if(cnt < 8):
            ax_head = inset_axes(axs[i][j], width='25%', height='45%', loc='lower right')
            sigma_mEeg_potMap = meet.sphere.potMap(coords, sigma_mEeg_spattern[cnt])
            meet.sphere.addHead(ax_head)
            ax_head.pcolormesh(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], vmin=-1, vmax=1, rasterized=True, cmap='plasma')
            ax_head.contour(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], levels=[0], colors='w', linewidths=1)
            ax_head.set_xlim([-1.15, 1.15])
            ax_head.set_ylim([-1.05, 1.25])
            ax_head.axis('off')

        cnt+=1

fig.tight_layout()
plt_show_save_fig()

# %%
# plot correlations heatmap of channels
plt_header('Correlation between channels for sigma burst trials rms, t = %s ms' % sigma_target_win_ms)
sns.heatmap(np.corrcoef(eeg_sfiltered_trials_multi_rms), annot = True, fmt = '.2f', xticklabels=channnames_csp, yticklabels=channnames_csp,
            vmin=0, vmax=1, cmap='YlGnBu_r', cbar_kws={'label':'r coeff'})
plt_show_save_fig()

# %%
# plot correlations heatmap of channels
plt_header('Correlation between channels for sigma band noise trials rms, t > 35 ms')
sns.heatmap(np.corrcoef(eeg_sfiltered_random_noise_multi_rms), annot = True, fmt = '.2f', xticklabels=channnames_csp, yticklabels=channnames_csp,
            vmin=0, vmax=1, cmap='YlGnBu_r', cbar_kws={'label':'r coeff'})
plt_show_save_fig()

# %%
# plot correlations heatmap of channels
plt_header('Difference between correlations')
sns.heatmap(np.corrcoef(eeg_sfiltered_trials_multi_rms)-np.corrcoef(eeg_sfiltered_random_noise_multi_rms), annot = True, fmt = '.2f', xticklabels=channnames_csp, yticklabels=channnames_csp,
            vmin=-0.2, vmax=0.2, cmap='PuOr_r', cbar_kws={'label':'r coeff(sigma burst)  - r coeff (noise) '})
plt_show_save_fig()

# %%
# plot sigma burst single trials stack on each other

plt_header('MEG, stacked all sigma burst trials')
data = eeg_sfiltered_short_view_trials[-1]
data_x = data_x_short_view

limit = 30

plt.pcolormesh(data_x, np.arange(len(data.T)), data.T,
               rasterized=True, cmap='coolwarm', vmin=-limit, vmax=limit)
plt.ylabel('Trial number')
plt.xlabel('Time [ms]')
clb = plt.colorbar(extend='both')
clb.set_label(unit)
plt.xlim(-20,80)
plt_show_save_fig()


plt_header('CSP0, stacked all sigma burst trials')
data = eeg_sfiltered_short_view_trials[0]
data_x = data_x_short_view

limit = 30

plt.pcolormesh(data_x, np.arange(len(data.T)), data.T,
               rasterized=True, cmap='coolwarm', vmin=-limit, vmax=limit)
plt.ylabel('Trial number')
plt.xlabel('Time [ms]')
clb = plt.colorbar(extend='both')
clb.set_label('abstract unit')
plt.xlim(-20,80)
plt_show_save_fig()

# %%
# plot sigma burst single trials stack on each other

start_trial = 500
end_trial = 600

plt_header('MEG, stacked exemplary sigma burst trials')
data = eeg_sfiltered_short_view_trials[-1][:,start_trial:end_trial+1]
data_x = data_x_short_view

limit = 30

plt.pcolormesh(data_x, np.arange(start_trial, end_trial+1, 1), data.T,
               rasterized=True, cmap='coolwarm', vmin=-limit, vmax=limit)
plt.ylabel('Trial number')
plt.xlabel('Time [ms]')
clb = plt.colorbar(extend='both')
clb.set_label(unit)
plt.xlim(-20,80)
plt_show_save_fig()


plt_header('CSP0, stacked exemplary sigma burst trials')
data = eeg_sfiltered_short_view_trials[0][:,start_trial:end_trial+1]
data_x = data_x_short_view

limit = 30

plt.pcolormesh(data_x, np.arange(start_trial, end_trial+1, 1), data.T,
               rasterized=True, cmap='coolwarm', vmin=-limit, vmax=limit)
plt.ylabel('Trial number')
plt.xlabel('Time [ms]')
clb = plt.colorbar(extend='both')
clb.set_label('abstract unit')
plt.xlim(-20,80)
plt_show_save_fig()

# %%
# percentile sorting mask function
def percentile_sorting_mask(percentiled_signal):

    trials_percentile_range = np.arange(0, 100, 10)
    trials_percentile_bins_names = []
    trials_percentile_list = []

    for i in trials_percentile_range:
        trials_percentile_list.append(scipy.stats.scoreatpercentile(percentiled_signal, i))

    trials_percentile_list[0] = -np.inf
    trials_percentile_list.append(np.inf)
    trials_percentile_range = np.append(trials_percentile_range, 100)

    for i in range(len(trials_percentile_range)-1):
        trials_percentile_bins_names.append('%s-%s' % (trials_percentile_range[i], trials_percentile_range[i+1]))
    
    percentile_mask_list = []

    for i in range(len(trials_percentile_list)-1):
        percentile_mask = (percentiled_signal > trials_percentile_list[i]) * (percentiled_signal <= trials_percentile_list[i+1])
        percentile_mask_list.append(percentile_mask)
    
    return percentile_mask_list, trials_percentile_bins_names

# %%
# plot EEG (CSP) and MEG sigma burst (high-frequency somatosensory evoked response)
# plot average of trials sorted by percentile bin of sigma burst rms

percentiled_signal = eeg_sfiltered_trials_rms[0]
percentile_mask_list, trials_percentile_bins_names = percentile_sorting_mask(percentiled_signal)

fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('Average of trials sorted by CSP0 sigma burst rms percentile bin', use_suptitle=True, fontsize=14)
cnt = 0
for i in range(3):
    for j in range(3):

        data = eeg_sfiltered_short_view_trials[cnt]
        data_x = data_x_short_view
        ylim = 40

        if(cnt == 8):
            unit = meg_unit
        else:
            unit = 'abstract unit'

        colors = plt.cm.viridis(np.linspace(0,1,len(percentile_mask_list)))
        for k in range(len(percentile_mask_list)):
            data_y = data[:,percentile_mask_list[k]]
            data_y = np.mean(data_y, axis=-1)
            axs[i][j].plot(data_x, data_y, color=colors[k], linewidth=1)

        axs[i][j].plot([], [], linewidth=0, label=' ')
        axs[i][j].axvline(sigma_target_win_ms[0], color='gray', linestyle=':')
        axs[i][j].axvline(sigma_target_win_ms[1], color='gray', linestyle=':')
        axs[i][j].axvline(0, color='silver')
        axs[i][j].set_ylim((-ylim, ylim))
        axs[i][j].set_ylabel(unit)
        axs[i][j].legend(title=channnames_csp[cnt])
        axs[i][j].set_xlim(10, 40)
        if(i==2):
            axs[i][j].set_xlabel('t [ms]')

        if(cnt < 8):
            ax_head = inset_axes(axs[i][j], width='25%', height='45%', loc='lower right')
            sigma_mEeg_potMap = meet.sphere.potMap(coords, sigma_mEeg_spattern[cnt])
            meet.sphere.addHead(ax_head)
            ax_head.pcolormesh(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], vmin=-1, vmax=1, rasterized=True, cmap='plasma')
            ax_head.contour(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], levels=[0], colors='w', linewidths=1)
            ax_head.set_xlim([-1.15, 1.15])
            ax_head.set_ylim([-1.05, 1.25])
            ax_head.axis('off')

        cnt+=1

fig.tight_layout()
plt_show_save_fig()

# %%
# plot EEG (CSP) and MEG sigma burst (high-frequency somatosensory evoked response)
# plot average of trials sorted by percentile bin of sigma burst rms

percentiled_signal = eeg_sfiltered_trials_rms[-1]
percentile_mask_list, trials_percentile_bins_names = percentile_sorting_mask(percentiled_signal)

fig, axs = plt.subplots(3, 3, figsize=(16, 9))
plt_header('Average of trials sorted by MEG sigma burst rms percentile bin', use_suptitle=True, fontsize=14)
cnt = 0
for i in range(3):
    for j in range(3):

        data = eeg_sfiltered_short_view_trials[cnt]
        data_x = data_x_short_view
        ylim = 40

        if(cnt == 8):
            unit = meg_unit
        else:
            unit = 'abstract unit'

        colors = plt.cm.viridis(np.linspace(0,1,len(percentile_mask_list)))
        for k in range(len(percentile_mask_list)):
            data_y = data[:,percentile_mask_list[k]]
            data_y = np.mean(data_y, axis=-1)
            axs[i][j].plot(data_x, data_y, color=colors[k], linewidth=1)

        axs[i][j].plot([], [], linewidth=0, label=' ')
        axs[i][j].axvline(sigma_target_win_ms[0], color='gray', linestyle=':')
        axs[i][j].axvline(sigma_target_win_ms[1], color='gray', linestyle=':')
        axs[i][j].axvline(0, color='silver')
        axs[i][j].set_ylim((-ylim, ylim))
        axs[i][j].set_ylabel(unit)
        axs[i][j].legend(title=channnames_csp[cnt])
        axs[i][j].set_xlim(10, 40)
        if(i==2):
            axs[i][j].set_xlabel('t [ms]')

        if(cnt < 8):
            ax_head = inset_axes(axs[i][j], width='25%', height='45%', loc='lower right')
            sigma_mEeg_potMap = meet.sphere.potMap(coords, sigma_mEeg_spattern[cnt])
            meet.sphere.addHead(ax_head)
            ax_head.pcolormesh(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], vmin=-1, vmax=1, rasterized=True, cmap='plasma')
            ax_head.contour(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], levels=[0], colors='w', linewidths=1)
            ax_head.set_xlim([-1.15, 1.15])
            ax_head.set_ylim([-1.05, 1.25])
            ax_head.axis('off')

        cnt+=1

fig.tight_layout()
plt_show_save_fig()

# %%
# plot average of trials sorted by percentile bin of sigma burst rms

percentiled_signal = eeg_sfiltered_trials_rms[0]
percentile_mask_list, trials_percentile_bins_names = percentile_sorting_mask(percentiled_signal)

data = eeg_sfiltered_short_view_trials[0]
data_x = data_x_short_view

fig, ax = plt.subplots()

colors = plt.cm.viridis(np.linspace(0,1,len(percentile_mask_list)))
for k in range(len(percentile_mask_list)):
    data_y = data[:,percentile_mask_list[k]]
    data_y = np.mean(data_y, axis=-1)
    ax.plot(data_x, data_y, color=colors[k], label=trials_percentile_bins_names[k], linewidth=1)

plt_header('Average of CSP0 trials sorted by CSP0 sigma burst rms percentile bin')
ax.legend(title='Percentile bin of CSP0 sigma burst rms', ncol=2, loc='upper right')
ax.set_xlabel('t [ms]')
ax.set_ylabel('abstract unit')
ax.set_xlim(10, 40)

ax_head = inset_axes(ax, width='25%', height='45%', loc='lower right')
sigma_mEeg_potMap = meet.sphere.potMap(coords, sigma_mEeg_spattern[0])
meet.sphere.addHead(ax_head)
ax_head.pcolormesh(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], vmin=-1, vmax=1, rasterized=True, cmap='plasma')
ax_head.contour(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], levels=[0], colors='w', linewidths=1)
ax_head.set_xlim([-1.15, 1.15])
ax_head.set_ylim([-1.05, 1.25])
ax_head.axis('off')

plt_show_save_fig()

# %%
# plot average of trials sorted by percentile bin of sigma burst rms

percentiled_signal = eeg_sfiltered_trials_rms[0]
percentile_mask_list, trials_percentile_bins_names = percentile_sorting_mask(percentiled_signal)

data = eeg_sfiltered_short_view_trials[-1]
data_x = data_x_short_view

colors = plt.cm.viridis(np.linspace(0,1,len(percentile_mask_list)))
for k in range(len(percentile_mask_list)):
    data_y = data[:,percentile_mask_list[k]]
    data_y = np.mean(data_y, axis=-1)
    plt.plot(data_x, data_y, color=colors[k], label=trials_percentile_bins_names[k], linewidth=1)

plt_header('Average of MEG trials sorted by CSP0 sigma burst rms percentile bin')
plt.legend(title='Percentile bin of CSP0 sigma burst rms', ncol=2, loc='upper right')
plt.xlabel('t [ms]')
plt.ylabel(meg_unit)
plt.xlim(10, 40)
plt_show_save_fig()

# %%
# plot average of trials sorted by percentile bin of sigma burst rms

percentiled_signal = eeg_sfiltered_trials_rms[-1]
percentile_mask_list, trials_percentile_bins_names = percentile_sorting_mask(percentiled_signal)

data = eeg_sfiltered_short_view_trials[-1]
data_x = data_x_short_view

fig, ax = plt.subplots()

colors = plt.cm.viridis(np.linspace(0,1,len(percentile_mask_list)))
for k in range(len(percentile_mask_list)):
    data_y = data[:,percentile_mask_list[k]]
    data_y = np.mean(data_y, axis=-1)
    ax.plot(data_x, data_y, color=colors[k], label=trials_percentile_bins_names[k], linewidth=1)

plt_header('Average of MEG trials sorted by MEG sigma burst rms percentile bin')
ax.legend(title='Percentile bin of MEG sigma burst rms', ncol=2, loc='upper right')
ax.set_xlabel('t [ms]')
ax.set_ylabel(meg_unit)
ax.set_xlim(10, 40)

plt_show_save_fig()

# %%
# plot average of trials sorted by percentile bin of sigma burst rms

percentiled_signal = eeg_sfiltered_trials_rms[-1]
percentile_mask_list, trials_percentile_bins_names = percentile_sorting_mask(percentiled_signal)

data = eeg_sfiltered_short_view_trials[0]
data_x = data_x_short_view

fig, ax = plt.subplots()

colors = plt.cm.viridis(np.linspace(0,1,len(percentile_mask_list)))
for k in range(len(percentile_mask_list)):
    data_y = data[:,percentile_mask_list[k]]
    data_y = np.mean(data_y, axis=-1)
    ax.plot(data_x, data_y, color=colors[k], label=trials_percentile_bins_names[k], linewidth=1)

plt_header('Average of CSP0 trials sorted by MEG sigma burst rms percentile bin')
ax.legend(title='Percentile bin of MEG sigma burst rms', ncol=2, loc='upper right')
ax.set_xlabel('t [ms]')
ax.set_ylabel('abstract unit')
ax.set_xlim(10, 40)

ax_head = inset_axes(ax, width='25%', height='45%', loc='lower right')
sigma_mEeg_potMap = meet.sphere.potMap(coords, sigma_mEeg_spattern[0])
meet.sphere.addHead(ax_head)
ax_head.pcolormesh(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], vmin=-1, vmax=1, rasterized=True, cmap='plasma')
ax_head.contour(sigma_mEeg_potMap[0], sigma_mEeg_potMap[1], sigma_mEeg_potMap[2], levels=[0], colors='w', linewidths=1)
ax_head.set_xlim([-1.15, 1.15])
ax_head.set_ylim([-1.05, 1.25])
ax_head.axis('off')

plt_show_save_fig()


