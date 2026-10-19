"""The one azimuth/elevation convention shared by every renderer and metric.

Both frames are y-up. A camera at (azimuth, elevation) sits at
``target + distance * direction(azimuth, elevation)`` and looks at ``target``;
azimuth 0 places it on +Z.

- Ground-truth harness (PyTorch3D): the object's reviewed canonical frame,
  ``target = HARNESS_LOOK_AT``, ``distance`` from the study config.
- SAM3D candidates (``notebook/inference.render_video``): the normalized
  reconstruction frame (largest extent 1), target at the origin, 60-degree FOV,
  distance ``reconstruction.candidate_distance``. This is the input camera's
  frame: (180, 0) re-renders the input view, and every candidate angle is
  relative to the input camera (not to gravity or the object's front).
  Verified visually on the smoke study, including elevated inputs. Candidate
  cameras use ``sam3d_gravity(input_elevation)`` as their up vector so renders
  are level; the frame's own +Y is the input camera's up, tilted by its elevation.
"""
import math

import numpy as np

HARNESS_LOOK_AT = (0.0, 0.35, 0.0)
SAM3D_FOV = 60.0
SAM3D_INPUT_VIEW = (180.0, 0.0)


def sam3d_gravity(input_elevation_deg):
    """World up in the SAM3D reconstruction frame for an input camera at this elevation.

    The input camera sits on -Z looking at +Z with up +Y; pitching it down by the
    elevation tilts world up toward the camera, i.e. (0, cos, -sin).
    """
    tilt = math.radians(input_elevation_deg)
    return np.array([0.0, math.cos(tilt), -math.sin(tilt)])


def direction(azimuth_deg, elevation_deg):
    """Unit vector from the look-at target toward the camera."""
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.sin(azimuth) * math.cos(elevation), math.sin(elevation),
                     math.cos(azimuth) * math.cos(elevation)], dtype=np.float32)
