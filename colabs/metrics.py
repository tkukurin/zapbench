# https://colab.research.google.com/drive/1sW-73KhbC7SBQbdMuApPyWpSYCMqOLFS#scrollTo=WDHIAUJaRtOp
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

# %% [markdown] id="7QjPa4bWOoCq"
# # Metrics

# %% [markdown] id="9Fzb8OE4Op0G"
# This tutorial explains how to load predictions made by the methods reported in the ZAPBench paper for additional analyses, e.g., to compute custom metrics.

# %% id="MKQngZ06PAsF"
# !pip install git+https://github.com/google-research/zapbench.git#egg=zapbench

# %% id="gL4cXGPvO1Yl"
from connectomics.common import ts_utils
import pandas as pd


# Load dataframe with results reported in the manuscript.
df = pd.DataFrame(
    ts_utils.load_json(f'gs://zapbench-release/dataframes/20250131/combined.json'))
df.head()

# %% [markdown] id="SR5kU2jFPgOl"
# Each run in the experiment has an experiment identifier (`xid`) in the dataframe.

# %% id="WDHIAUJaRtOp"
unique_xids = df.query(
    'method not in ("mean", "stimulus")'
).groupby(['method', 'context'])['xid'].unique().reset_index()
unique_xids

# %% [markdown] id="w0N_WFWQSOBy"
# Above table shows the unique experiment IDs. There are 3 xids per method for a given context length, as we report 3 seeds each.

# %% id="sScqLR0TZQhO"
unique_xids_naive_baselines = df.query(
    'method in ("mean", "stimulus")'
).groupby(['method', 'context'])['xid'].unique().reset_index()
unique_xids_naive_baselines

# %% [markdown] id="pC-UGz-IZVsn"
# For the naive baselines, stimulus, and mean, there are fewer experiment IDs -- since these baselines are deterministic, we only ran a single seed. The mean baseline for long context has two associated IDs since we used two different window lengths, as described in the manuscript.

# %% [markdown] id="MgwPlHSMRncV"
# Using the `xid`, we can obtain predictions and associated targets from Google Cloud storage for any given condition, for example:

# %% id="dZNLYQSHPbde"
import tensorstore as ts
from zapbench import constants


def get_data(xid, condition_id=0, subfolder='predictions', return_ds=False):
  holdout = 'holdout_' if condition_id in constants.CONDITIONS_HOLDOUT else ''
  ds = ts.open({
    'driver': 'zarr',
    'open': True,
    'kvstore': {
      'bucket': 'zapbench-release',
      'driver': 'gcs',
      'path': f'inference/20250131/{xid}/{subfolder}/test_{holdout}condition_{condition_id}/',
    },
  }).result()
  return ds if return_ds else ds.read().result()


def get_targets(xid, condition_id=0):
  return get_data(xid, condition_id, subfolder='targets')


def get_predictions(xid, condition_id=0):
  return get_data(xid, condition_id, subfolder='predictions')


xid = '146855456/1'  # linear model
condition_name = 'gain'

predictions = get_predictions(
    xid, constants.CONDITION_NAMES.index(condition_name))
targets = get_targets(
    xid, constants.CONDITION_NAMES.index(condition_name))

print(f'{predictions.shape=}', f'{targets.shape=}')

# %% [markdown] id="EUsK5xEfQvVm"
# The shape of predictions and corresponding targets is `window x timestep x neuron`.

# %% [markdown] id="kOoNKtQ7Q8HN"
# We can re-compute existing metrics on these arrays, or define our own, e.g.:

# %% id="OUPu5Wv8Qo_A"
import jax.numpy as jnp


def compute_mae_and_mse(targets, predictions):
  diff = jnp.nan_to_num(targets) - jnp.nan_to_num(predictions)
  mae = jnp.mean(jnp.abs(diff), axis=(0, 2))
  mse = jnp.mean(diff**2, axis=(0, 2))
  return mae, mse


mae, mse = compute_mae_and_mse(targets, predictions)
