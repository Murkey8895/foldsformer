import numpy as np
import pyflex
from copy import deepcopy
from softgym.utils.pyflex_utils import center_object
from tqdm import tqdm
import pickle
from softgym.envs.flex_utils import update_camera, set_scene, get_state
import argparse
import os


def rotate_particles(angle):
    pos = pyflex.get_positions().reshape(-1, 4)
    center = np.mean(pos, axis=0)
    pos -= center
    new_pos = pos.copy()
    new_pos[:, 0] = np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2]
    new_pos[:, 2] = np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2]
    new_pos += center
    pyflex.set_positions(new_pos)


def get_default_config():
    cam_pos, cam_angle = np.array([0, 0.65, 0]), np.array([0 * np.pi, -90 / 180.0 * np.pi, 0])
    config = {
        "ClothPos": [0, 0, 0],
        "ClothSize": [55, 55],
        "ClothStiff": [2.0, 0.5, 1.0],
        "mass": 0.0054,
        "camera_name": "default_camera",
        "camera_params": {
            "default_camera": {
                "pos": cam_pos,
                "angle": cam_angle,
                "width": 720,
                "height": 720,
            }
        },
        "flip_mesh": 0,
    }
    return config


def vary_cloth_size(cloth_type):
    assert cloth_type in ["square", "rectangle", "random"], f"input mode is {cloth_type}"
    if cloth_type == "square":
        dim = np.random.randint(50, 60)
        return dim, dim
    elif cloth_type == "rectangle":
        ratio = np.random.uniform(0.7, 0.9)
        dim = np.random.randint(50, 60)
        return dim, int(dim * ratio)
    elif cloth_type == "random":
        p = np.random.uniform(0, 1)
        if p > 0.5:
            return np.random.randint(50, 60), np.random.randint(50, 60)
        else:
            dim = np.random.randint(50, 60)
            return dim, dim


def generate_cached_configs(nums, cloth_type):
    max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
    stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
    generated_configs, generated_states = [], []
    default_config = get_default_config()
    pyflex.init(True, True, 720, 720)

    for i in tqdm(range(nums)):
        config = deepcopy(default_config)
        update_camera(config["camera_params"], config["camera_name"])
        cloth_dimx, cloth_dimy = vary_cloth_size(cloth_type)
        config["ClothSize"] = [cloth_dimx, cloth_dimy]

        set_scene(config)
        pos = pyflex.get_positions().reshape(-1, 4)
        pos[:, :3] -= np.mean(pos, axis=0)[:3]
        pos[:, 1] = 0.005
        pos[:, 3] = 1
        pyflex.set_positions(pos.flatten())
        pyflex.set_velocities(np.zeros_like(pos))
        for _ in range(5):  # In case if the cloth starts in the air
            pyflex.step()

        for _ in range(max_wait_step):
            pyflex.step()
            curr_vel = pyflex.get_velocities()
            if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                break

        center_object()
        angle = (np.random.random() - 0.5) * np.pi / 2
        rotate_particles(angle)

        generated_configs.append(deepcopy(config))
        generated_states.append(deepcopy(get_state(config["camera_params"])))

    return generated_configs, generated_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cached Configs.")
    parser.add_argument("--num_cached", type=int, default=1000, help="Number of cached configs to be generated")
    parser.add_argument("--cloth_type", type=str, default="square", help="Cloth type(square, rectangle, random)")
    args = parser.parse_args()

    cached_path = os.path.join("cached configs", args.cloth_type + str(args.num_cached) + ".pkl")
    generated_configs, generated_states = generate_cached_configs(args.num_cached, args.cloth_type)

    os.makedirs("cached configs", exist_ok=True)

    with open(cached_path, "wb+") as handle:
        pickle.dump((generated_configs, generated_states), handle)
