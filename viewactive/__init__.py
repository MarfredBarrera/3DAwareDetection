"""ViewActive (Wu et al., arXiv:2409.09997) reimplemented: a single-image
predictor of the 3D Viewpoint Quality Field. Predictor: ``viewactive.model.ViewActive``;
pipeline: ``python -m viewactive --config configs/viewactive*.yaml``."""
from viewactive.grid import Grid
from viewactive.labels import CHANNELS
