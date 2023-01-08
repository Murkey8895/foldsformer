import numpy as np


class DoubleTriangle:
    def __init__(self):
        self.pick_idxs = [2, 3]

    def get_action(self, curr_corners, pick_idx):
        place_idx = 3 - pick_idx
        pick_pos = curr_corners[pick_idx]
        place_pos = curr_corners[place_idx]

        return pick_pos, place_pos


class AllCornersInward:
    def __init__(self):
        self.gamma = 0.9

    def get_action(self, curr_corners, center, pick_idx):
        pick_pos = curr_corners[pick_idx]
        place_pos = pick_pos + self.gamma * (center - pick_pos)

        return pick_pos, place_pos


class DoubleStraight:
    def __init__(self):
        self.pickplace_idxs = [(0, 1), (2, 3), (4, 7)]

    def get_action(self, curr_corners, edge_middles, pickplace_idx):
        all_keypoints = np.concatenate((curr_corners, edge_middles), axis=0)
        pick_pos, place_pos = all_keypoints[pickplace_idx[0]], all_keypoints[pickplace_idx[1]]
        return pick_pos, place_pos


class CornersEdgesInward:
    def __init__(self):
        self.pickplace_idxs = [(0, 8), (2, 8), (1, 4), (3, 7)]
        self.gamma = 0.9

    def get_action(self, curr_corners, edge_middles, center, pickplace_idx):
        all_keypoints = np.concatenate((curr_corners, edge_middles, np.array([center])), axis=0)
        pick_pos = all_keypoints[pickplace_idx[0]]
        target_pos = all_keypoints[pickplace_idx[1]]
        place_pos = pick_pos + self.gamma * (target_pos - pick_pos)

        return pick_pos, place_pos


Demonstrator = {
    "DoubleTriangle": DoubleTriangle,
    "DoubleStraight": DoubleStraight,
    "AllCornersInward": AllCornersInward,
    "CornersEdgesInward": CornersEdgesInward,
}
