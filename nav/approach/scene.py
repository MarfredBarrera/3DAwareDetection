"""Camera and pose of the agent's embodiment, and depth back-projection.

The camera matches the evaluated ObjectNav embodiment (ApexNav config/habitat_eval_hm3dv2.yaml): 640x480, hfov 79,
0.88 m high, radius 0.18, forward 0.25 m, turns 30 deg, no sliding; the planners' actions are move_forward /
turn_left / turn_right at camera pitch 0. ``sim`` below is a habitat_sim Simulator (agent 0, sensor "rgb").
"""
import math

import numpy as np

HFOV, WIDTH, HEIGHT, CAMERA_HEIGHT = 79.0, 640, 480, 0.88
FORWARD_M, TURN_DEG, AGENT_RADIUS = 0.25, 30.0, 0.18
DEPTH_MAX = 20.0          # the back-projection limit, metres (the callers pass ApexNav's 5 m depth)
FOCAL = (WIDTH / 2) / math.tan(math.radians(HFOV) / 2)


def agent_pose(sim):
    """(position xyz, yaw rad, camera pitch rad) of agent 0."""
    import quaternion
    state = sim.get_agent(0).get_state()
    forward = quaternion.rotate_vectors(state.rotation, np.array([0.0, 0.0, -1.0]))
    yaw = math.atan2(-forward[0], -forward[2])
    cam = state.sensor_states["rgb"]
    look = quaternion.rotate_vectors(cam.rotation, np.array([0.0, 0.0, -1.0]))
    pitch = math.asin(max(-1.0, min(1.0, look[1])))
    return np.asarray(state.position, dtype=float), yaw, pitch


def camera(sim):
    """(position, 3x3 rotation camera->world) of the rgb sensor."""
    import quaternion
    cam = sim.get_agent(0).get_state().sensor_states["rgb"]
    return np.asarray(cam.position, dtype=float), quaternion.as_rotation_matrix(cam.rotation)


def project(points, cam_pos, cam_rot):
    """World points -> (u, v, z) pixel coords and depth (z > 0 in front)."""
    local = (np.asarray(points) - cam_pos) @ cam_rot   # rows: R^T (p - t)
    z = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = WIDTH / 2 + FOCAL * local[:, 0] / z
        v = HEIGHT / 2 - FOCAL * local[:, 1] / z
    return u, v, z


def backproject(depth, mask, cam_pos, cam_rot, stride=2):
    """World points of the masked depth pixels (subsampled by ``stride``)."""
    vv, uu = np.nonzero(mask[::stride, ::stride])
    vv, uu = vv * stride, uu * stride
    z = depth[vv, uu]
    keep = (z > 0.05) & (z < DEPTH_MAX - 0.1)
    vv, uu, z = vv[keep], uu[keep], z[keep]
    local = np.stack([(uu - WIDTH / 2) * z / FOCAL, -(vv - HEIGHT / 2) * z / FOCAL, -z], 1)
    return local @ cam_rot.T + cam_pos
