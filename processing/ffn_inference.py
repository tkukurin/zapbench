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

# %% [markdown] id="Y-LMa0_zRyiE"
# # FFN inference
#

# %% id="UO9ixXAN7Hw-"
# Install the latest snapshot from the FFN repository.
# !pip install git+https://github.com/google/ffn

# %% id="P2BH-ACTDPgs"
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Ensure tensorstore does not attempt to use GCE credentials
os.environ['GCE_METADATA_ROOT'] = 'metadata.google.internal.invalid'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# %% id="2j8v1nH_G9jh"
import functools

from clu import checkpoint
from connectomics.jax.models import convstack
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy.ndimage import label
import tensorstore as ts

from ffn.inference import inference
from ffn.inference import inference_utils
from ffn.inference import inference_pb2
from ffn.inference import executor
from ffn.inference import seed
from ffn.training import model as ffn_model

# %% id="hLbhWzo1HNjW"
# Check for GPU presence. If this fails, use "Runtime > Change runtime type".
assert jax.devices()[0].platform in ('gpu', 'tpu')

# %% id="Odkxn5nyMNMA"
# Load dataset.
context = ts.Context({'cache_pool': {'total_bytes_limit': 1_000_000_000}})
anatomy_data = ts.open({
    'driver': 'zarr3',
    'kvstore': {'driver': 'gcs', 'bucket': 'zapbench-release'},
    'path': 'volumes/20240930/anatomy_clahe_ffn/',},
    read=True, context=context).result()

# %% id="dosI88JzMiR9"
# Load a subvolume for local processing.
x0, y0, z0 = 740, 800, 80
raw = anatomy_data[x0:x0+100, y0:y0+100, z0:z0+80].read().result()
raw = np.transpose(raw, [2, 1, 0])  # xyz -> zyx
raw = (raw.astype(np.float32) - 0.5) / 1.  # normalize data for inference

# %% id="IlVzzzb0SYY2"
# Load sample model checkpoint.
# !gsutil cp gs://zapbench-release/ffn_checkpoints/20240930/ckpt-332* .

ckpt = checkpoint.Checkpoint('')
state = ckpt.load_state(state=None, checkpoint='ckpt-332')

# %% id="fh957IM9XfTR"
# Instantiate model.
model = convstack.ResConvStack(convstack.ConvstackConfig(
    depth=8,
    padding='same',
    use_layernorm=True))
fov_size = 33, 33, 33
model_info = ffn_model.ModelInfo(
    deltas=(0, 0, 0),
    pred_mask_size=fov_size,
    input_seed_size=fov_size,
    input_image_size=fov_size)

@jax.jit
def _apply_fn(data):
  return model.apply({'params': state['params']}, data)


# %% id="wugXTgtyS1aB"
# Instantiate inference.
iface = executor.ExecutorInterface()
counters = inference_utils.Counters()
exc = executor.JAXExecutor(iface, model_info, _apply_fn, counters, 1)
exc.start_server()

options = inference_pb2.InferenceOptions(
    init_activation=0.95,
    pad_value=0.5,
    move_threshold=0.6,
    segment_threshold=0.5,
    min_boundary_dist={'x': 2, 'y': 2, 'z': 1},
    min_segment_size=100,
)
cv = inference.Canvas(
    model_info,
    exc.get_client(counters),
    raw,
    options,
    voxel_size_zyx=(2, 2, 2)
)
policy = functools.partial(
    seed.SequentialPolicies,
    **{'policies': [['PolicyImagePeaks3D2D', {}], ['PolicyImagePeaks2DDisk', {}]]}
)

# %% id="yb2Y9eJTRDWC"
# Segment subvolume.
cv.segment_all(seed_policy=policy)

# %% id="mGVmRSSY1PAO"
# !pip install neuroglancer

# %% id="VkJx8BAzZApX"
# Visualize results in neuroglancer.
import neuroglancer

# %% id="tIWzzhnQHVGv"
seg = cv.segmentation

dimensions = neuroglancer.CoordinateSpace(
    names=['x', 'y', 'z'],
    units='nm',
    scales=[1, 1, 1],
)
viewer = neuroglancer.Viewer()
with viewer.txn() as s:
  s.dimensions = dimensions
  s.layers['raw'] = neuroglancer.ImageLayer(
      source=neuroglancer.LocalVolume(np.transpose((raw).astype(np.float32), [2, 1, 0]),
      dimensions))
  s.layers['seg'] = neuroglancer.SegmentationLayer(
      source=neuroglancer.LocalVolume(np.transpose(seg.astype(np.uint64), [2, 1, 0]),
      dimensions),
      segments=[s for s in np.unique(seg) if s > 0])

viewer
