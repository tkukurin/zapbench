# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="Cl8agfloYQ8_"
# # Stimuli

# %% id="Vv2w7O8yYCya"
# Install dependencies
# !pip install matplotlib
# !pip install numpy
# !pip install scipy

# %% id="6HBx1U1wYYU5"
import io

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


CONDITION_NAMES = ('gain', 'dots', 'flash', 'taxis', 'turning', 'position',
                   'open loop', 'rotation', 'dark')


def load_stimuli_and_ephys(file_handle, num_channels=10):
  try:
    data = np.fromfile(file_handle, dtype=np.float32)
  except io.UnsupportedOperation:
    data = np.frombuffer(file_handle.read(), dtype=np.float32)
  if data.size % num_channels:
    raise ValueError(f'Data does not fit in num_channels: {num_channels}')
  return data.reshape((-1, num_channels)).T


# %% id="yFYZZhR9eDq3"
# Download raw stimuli and ephys time-series from GCS
# !gsutil cp gs://zapbench-release/volumes/20240930/stimuli_raw/stimuli_and_ephys.10chFlt .

# %% id="uAf1xY1weMly"
with open('./stimuli_and_ephys.10chFlt', 'rb') as f:
  stimuli_and_ephys = load_stimuli_and_ephys(f)

# %% [markdown] id="hIYchqr9iJXZ"
# ## Per-condition stimulus time-series
#
# See ZAPBench manuscript supplement for explanation of the different conditions.

# %% id="0j1xXGYhfE-R"
condition_indices = stimuli_and_ephys[4]
stimParam3 = stimuli_and_ephys[6]
stimParam4 = stimuli_and_ephys[3]
visual_velocity = stimuli_and_ephys[8]


def plot_condition(condition_index):
  _, axs = plt.subplots(figsize=(30, 10), nrows=3, sharex=True)
  mask = (condition_indices == condition_index + 1)
  axs[0].plot(stimParam3[mask])
  axs[0].set_title('stimParam3')
  axs[1].plot(stimParam4[mask])
  axs[1].set_title('stimParam4')
  axs[2].plot(visual_velocity[mask])
  axs[2].set_title('visualVelocity')
  for ax in axs:
    ax.set_xlim([9, len(stimParam3[mask])])
  plt.show()

for condition in range(9):
  print(CONDITION_NAMES[condition])
  plot_condition(condition)

# %% [markdown] id="T9QbrYtMiGeH"
# ## Timestep markers (TTLs)
#
# Extracts timestep markers to align stimulus time-series (which was recorded at higher temporal resolution) with imaging.

# %% id="MQs3QOQxeiUG"
timesteps = dict(emf3=7870)  # volume timesteps
fish = 'emf3'

ttls = stimuli_and_ephys[2]
ttls_high = scipy.signal.find_peaks(ttls, distance=500, height=3.55)[0]
ttls_low = scipy.signal.find_peaks(ttls, distance=50, height=1)[0]

# remove volume imaging start steps and only keep plane imaging steps
low_peaks = np.array([l for l in ttls_low if l not in ttls_high])
high_peaks = ttls_high

# functional data has 72 frames in z; expecting ratio of ~72 between low and high peaks
print(f'{len(low_peaks) / len(ttls_high)=}')
ttls_low = low_peaks

# 7872 entries corresponding to onset of imaging for t=0 ... 7871; no entries for final 7 frames of dataset
condition_index = stimuli_and_ephys[4]

# idx when a given condition begins
condition_onsets = np.where(np.diff(condition_index) != 0)[0] + 1
print(f'{condition_onsets=}')

def markers_for_timestep(t):
  assert t < timesteps[fish]
  condition = (ttls_high[t] <= ttls_low) & (ttls_high[t+1] > ttls_low)
  idx = np.where(condition)[0]
  return ttls_low[idx]

# high res timesteps per imaging timestep
markers = [markers_for_timestep(t) for t in range(timesteps[fish])]

def idx_to_t(idx):
  for t in range(len(markers)):
    if markers[t][0] <= idx < markers[t+1][0]:
      return t
  return None

# condition onsets translated to imaging timesteps
condition_onsets_imaging = [idx_to_t(idx) for idx in condition_onsets]
