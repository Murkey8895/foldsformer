import numpy as np
import pyflex
from copy import deepcopy


def set_scene(config, state=None):
    render_mode = 2
    camera_params = config["camera_params"][config["camera_name"]]
    env_idx = 0
    mass = config["mass"] if "mass" in config else 0.5
    scene_params = np.array(
        [
            *config["ClothPos"],
            *config["ClothSize"],
            *config["ClothStiff"],
            render_mode,
            *camera_params["pos"][:],
            *camera_params["angle"][:],
            camera_params["width"],
            camera_params["height"],
            mass,
            config["flip_mesh"],
        ]
    )

    pyflex.set_scene(env_idx, scene_params, 0)

    if state is not None:
        set_state(state)


def set_state(state_dict):
    pyflex.set_positions(state_dict["particle_pos"])
    pyflex.set_velocities(state_dict["particle_vel"])
    pyflex.set_shape_states(state_dict["shape_pos"])
    pyflex.set_phases(state_dict["phase"])
    camera_params = deepcopy(state_dict["camera_params"])
    update_camera(camera_params, "default_camera")


def update_camera(camera_params, camera_name="default_camera"):
    camera_param = camera_params[camera_name]
    pyflex.set_camera_params(
        np.array([*camera_param["pos"], *camera_param["angle"], camera_param["width"], camera_param["height"]])
    )


def get_state(camera_params):
    pos = pyflex.get_positions()
    vel = pyflex.get_velocities()
    shape_pos = pyflex.get_shape_states()
    phase = pyflex.get_phases()
    camera_params = deepcopy(camera_params)
    return {
        "particle_pos": pos,
        "particle_vel": vel,
        "shape_pos": shape_pos,
        "phase": phase,
        "camera_params": camera_params,
    }
