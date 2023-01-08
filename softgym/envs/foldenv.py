import numpy as np
import pickle
import pyflex
from softgym.action_space.action_space import PickerPickPlace
from softgym.envs.flex_utils import set_scene
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt


class FoldEnv:
    def __init__(self, cached_path, gui=False, render_dim=224, particle_radius=0.00625):
        # environment state variables
        self.grasp_state = False
        self.particle_radius = particle_radius
        self.image_dim = render_dim

        # visualizations
        self.gui = gui
        self.gui_render_freq = 2
        self.gui_step = 0

        # setup env
        self.setup_env()

        # primitives parameters
        self.grasp_height = self.action_tool.picker_radius
        self.default_speed = 1e-2
        self.reset_pos = [0.5, 0.5, 0.5]

        # configs
        with open(cached_path, "rb") as handle:
            self.configs, self.init_states = pickle.load(handle)
            self.num_configs = len(self.init_states)
            print("{} config and state pairs loaded from {}".format(self.num_configs, cached_path))

    def setup_env(self):
        pyflex.init(not self.gui, True, 720, 720)
        self.action_tool = PickerPickPlace(
            num_picker=1,
            particle_radius=self.particle_radius,
            picker_threshold=0.005,
            picker_low=(-10.0, 0.0, -10.0),
            picker_high=(10.0, 10.0, 10.0),
        )

    def reset(self, config_id):
        config, state = self.configs[config_id], self.init_states[config_id]
        set_scene(config=config, state=state)
        self.current_config = deepcopy(config)
        self.camera_params = deepcopy(state["camera_params"])

        self.action_tool.reset(self.reset_pos)
        self.step_simulation()
        self.set_grasp(False)

    def step_simulation(self):
        pyflex.step()
        if self.gui and self.gui_step % self.gui_render_freq == 0:
            pyflex.render()
        self.gui_step += 1

    def set_grasp(self, grasp):
        if grasp == True:
            self.grasp_state = 1
        else:
            self.grasp_state = 0

    def render_image(self):
        rgb, depth = pyflex.render()
        rgb = rgb.reshape((720, 720, 4))[::-1, :, :3]
        depth = depth.reshape((720, 720))[::-1]
        rgb = cv2.resize(rgb, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR)
        return rgb, depth

    #################################################
    ######################PIcker######################
    #################################################\
    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_pos()[0].squeeze(0)
            delta = target_pos - curr_pos
            dist = np.linalg.norm(delta)
            if dist < eps and (min_steps is None or step > min_steps):
                return
            action = []

            if dist < speed:
                action.extend([*target_pos, float(self.grasp_state)])
            else:
                delta = delta / dist
                action.extend([*(curr_pos + delta * speed), float(self.grasp_state)])
            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)

    def pick_and_place(self, pick_pos, place_pos, lift_height=0.1):
        pick_pos[1] = self.grasp_height
        place_pos[1] = self.grasp_height

        prepick_pos = pick_pos.copy()
        prepick_pos[1] = lift_height

        preplace_pos = place_pos.copy()
        preplace_pos[1] = lift_height

        # execute action
        self.movep(prepick_pos, speed=5e-3)
        self.movep(pick_pos, speed=5e-3)
        self.set_grasp(True)
        self.movep(prepick_pos, speed=5e-3)
        self.movep(preplace_pos, speed=5e-3)
        self.movep(place_pos, speed=5e-3)
        self.set_grasp(False)
        self.movep(preplace_pos, speed=5e-3)

        # reset
        self.movep(self.reset_pos, speed=5e-3)

    #################################################
    ###################Ground Truth###################
    #################################################
    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    # Cloth Keypoints are defined:
    #  0  5  2
    #  4  8  7
    #  1  6  3
    def get_keypoints_idx(self):
        """The keypoints are defined as the four corner points of the cloth"""
        dimx, dimy = self.current_config["ClothSize"]
        idx_p1 = 0
        idx_p2 = dimx * (dimy - 1)
        idx_p3 = dimx - 1
        idx_p4 = dimx * dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4])

    def get_corners(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self.get_keypoints_idx(), :3]
        return keypoint_pos

    def get_center(self):
        curr_pos = pyflex.get_positions()
        pos = np.reshape(curr_pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return np.array([0.5 * (min_x + max_x), 0, 0.5 * (min_y + max_y)])

    def get_edge_middles(self):
        dimx, dimy = self.current_config["ClothSize"]
        idx_4 = int((dimy - 1) / 2) * dimx
        idx_5 = int((dimx - 1) / 2)
        idx_6 = dimx * (dimy - 1) + int((dimx - 1) / 2)
        idx_7 = int((dimy - 1) / 2) * dimx + dimx - 1
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        edge_middle_idx = np.array([idx_4, idx_5, idx_6, idx_7])
        edge_middle_pos = particle_pos[edge_middle_idx, :3]
        return edge_middle_pos
