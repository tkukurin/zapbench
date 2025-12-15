# https://colab.research.google.com/drive/1A5YWcz14JJlwJY_Tly9O1GAuf1zBfZny

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

# %% [markdown] id="G1Pc3oeULxKL"
# # Train and evaluate

# %% [markdown] id="mVQJ85TyZa7x"
# This tutorial explains how to train and eval forecasting methods on ZAPBench in a framework agnostic way. For this, we will be using [`grain`, a library for reading and processing ML training data](https://github.com/google/grain).
#

# %% [markdown] id="VFVIA-6tjKEJ"
# ## Training

# %% [markdown] id="uar0yiktkBum"
# `zapbench` provides data sources that are compatible with `grain`, e.g.:

# %% id="mhQAMdomm9JK"
# !pip install git+https://github.com/google-research/zapbench.git#egg=zapbench
# %% id="QhH3KuMtk_MT"
from zapbench import constants
from zapbench import data_utils
from zapbench.ts_forecasting import data_source


condition_name = 'turning'  # can be any name in constants.CONDITION_NAMES
num_timesteps_context = 4  # 4 for short context, 256 for long context
split = 'train'  # change to 'val' for validation set, e.g., for early stopping

config = data_source.TensorStoreTimeSeriesConfig(
    input_spec=data_utils.adjust_spec_for_condition_and_split(
        condition=constants.CONDITION_NAMES.index(condition_name),
        split=split,
        spec=data_utils.get_spec('240930_traces'),
        num_timesteps_context=num_timesteps_context),
    timesteps_input=num_timesteps_context,
    timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
)
source = data_source.TensorStoreTimeSeries(config)

print(f'{len(source)=}')

# %% id="mznamK0To7-n"
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)

# %% id="_CxkG54Xozzh"

# NOTE(tk) 3:11m on Mac M4 (render)
source[0]
# `series_input`: past activity of `num_timesteps_context`
# `series_output`: 32 timesteps of subsequent activity (ZAPBench prediction horizon).

# %% [markdown] id="vEO_xsi0p_1N"
# By enabling `prefetch` on `data_source.TensorStoreTimeSeries`, we can load the entire data into memory upfront. This makes indexing significantly faster once the source has been initialized.

# %% id="lktbCrGZqwQ7"

# NOTE(tk) 4:52m on Mac M4
source = data_source.TensorStoreTimeSeries(config, prefetch=True)

# %% [markdown] id="wAT0jhfe_f_p"
# We can also create a data source that combines data from all training conditions (should take about a minute to prefetch):

# %% id="S379VzI6_d6y"
sources = []

# Iterate over all training conditions (excludes 'taxis'), and create
# data sources.
for condition_id in constants.CONDITIONS_TRAIN:
  config = data_source.TensorStoreTimeSeriesConfig(
      input_spec=data_utils.adjust_spec_for_condition_and_split(
          condition=condition_id,
          split='train',
          spec=data_utils.get_spec('240930_traces'),
          num_timesteps_context=num_timesteps_context),
      timesteps_input=num_timesteps_context,
      timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
  )
  sources.append(data_source.TensorStoreTimeSeries(config, prefetch=True))

# Concatenate into a single source.
source = data_source.ConcatenatedTensorStoreTimeSeries(*sources)

f'{len(source)=}'

# %% [markdown] id="m14JiTXJ1DRG"
# Next, we set up an index sampler and construct a data loader with `grain`:

# %% id="easEO6aY1Cu_"
import grain.python as grain


batch_size = 8
num_epochs = 1
shuffle = True

index_sampler = grain.IndexSampler(
    num_records=len(source),
    num_epochs=num_epochs,
    shard_options=grain.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True),
    shuffle=shuffle,
    seed=101
)

data_loader = grain.DataLoader(
    data_source=source,
    sampler=index_sampler,
    operations=[
        grain.Batch(
            batch_size=batch_size, drop_remainder=True)
    ],
    worker_count=0
)

# We can iterate over the data loader which will get
# elements with a batch dimension in random order for
# `num_epochs`

# `grain` has many useful features -- for example, we can easily add operations to the data loader to adjust shapes, or add augmentations. More details are in [grain's DataLoader guide](https://google-grain.readthedocs.io/en/latest/tutorials/data_loader_tutorial.html).

# %% [markdown] id="5dR8YHzt3yo4"
# ## Evaluation

# %% [markdown] id="xz_l24rcji2C"
# Say we have trained a new baseline, how do we evaluate it?
#
# We are going to use the mean baseline from the manuscript as an example: It can easily be re-implemented in NumPy and does not require any training.

# %% id="Iv5mQQQYiNtB"
import numpy as np

npred = constants.PREDICTION_WINDOW_LENGTH

def f_mean(past_activity: np.ndarray) -> np.ndarray:
  """Mean baseline. (time x neurons) -> (pred_time x neurons)"""
  return past_activity.mean(axis=0).reshape((1, -1)).repeat(npred, axis=0)


# %% id="Qlywvmcqi0RZ"

# NOTE(tk) 24.8m
# indexing as in section 3.2
# https://openreview.net/pdf?id=oCHsDpyawq
infer_source = data_source.TensorStoreTimeSeries(
    data_source.TensorStoreTimeSeriesConfig(
        input_spec=data_utils.get_spec('240930_traces'),
        timesteps_input=num_timesteps_context,
        timesteps_output=constants.PREDICTION_WINDOW_LENGTH,
    ),
    prefetch=True
)

# %% id="NVMOhD1KD8t8"
from collections import defaultdict

from connectomics.jax import metrics
from tqdm import tqdm

# Placeholder for results
MAEs = defaultdict(list)

# Iterate over all conditions, and make predictions for all contiguous snippets
# of length 32 in the respective test set.
for condition_id, condition_name in tqdm(enumerate(constants.CONDITION_NAMES)):
  split = ('test' if condition_id not in constants.CONDITIONS_HOLDOUT
           else 'test_holdout')
  test_min, test_max = data_utils.adjust_condition_bounds_for_split(
      split,
      *data_utils.get_condition_bounds(condition_id),
      num_timesteps_context=num_timesteps_context)

  for window in range(
      data_utils.get_num_windows(test_min, test_max, num_timesteps_context)):
    element = infer_source[test_min + window]

    predictions = f_mean(element['series_input'])
    mae = metrics.mae(predictions=predictions, targets=element['series_output'])

    MAEs[condition_name].append(np.array(mae))


# %% id="3RlJY6t5HaKB"
import matplotlib.pyplot as plt


steps_ahead = np.arange(npred) + 1

for condition_name in constants.CONDITION_NAMES:
  mae = np.stack(MAEs[condition_name]).mean(axis=0)  # Average over windows
  plt.plot(steps_ahead, mae, label=condition_name)

plt.title('mean baseline, short context')
plt.xlabel('steps predicted ahead')
plt.ylabel('MAE')
plt.ylim((0.015, 0.06))
plt.xlim(1, npred)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %% [markdown] id="3V6o7v5VXTb2"
# Finally, we briefly check that these results match the ones in the manuscript:

# %% id="rkH6JVuuXTBF"
from connectomics.common import ts_utils
import pandas as pd


# Load dataframe with results reported in the manuscript.
df = pd.DataFrame(
    ts_utils.load_json(f'gs://zapbench-release/dataframes/20250131/combined.json'))
df.head()

# %% id="X1BA4QHkXv0F"
for condition_name in constants.CONDITION_NAMES:
  mae = np.stack(MAEs[condition_name]).mean(axis=0)
  mae_df = df.query(
      f'method == "mean" and context == 4 and condition == "{condition_name}"'
  ).sort_values('steps_ahead')['MAE'].to_numpy()
  np.testing.assert_array_almost_equal(mae, mae_df, decimal=8)
