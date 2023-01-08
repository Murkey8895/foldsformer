import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from einops import rearrange
import torchvision.transforms.functional as F


def make_gaussmap(point, img_size, sigma=5):
    center_x = round(point[0])
    center_y = round(point[1])
    xy_grid = np.arange(0, img_size)
    [x, y] = np.meshgrid(xy_grid, xy_grid)
    dist = (x - center_x) ** 2 + (y - center_y) ** 2
    gauss_map = np.exp(-dist / (2 * sigma * sigma))
    return gauss_map


class ClothDataSet(Dataset):
    def __init__(self, data_index_filename, img_size, spatial_augment):
        super().__init__()
        self.data_index_filename = data_index_filename
        self.img_size = img_size
        self.spatial_augment = spatial_augment
        with open(self.data_index_filename, "rb") as f:
            data = pickle.load(f)
            self.depth_indices = data["depth"]
            self.actions = data["action"]

    def __len__(self):
        return len(self.depth_indices)

    def __getitem__(self, index):
        # depth
        depths = []
        pick_pixel, place_pixel = self.actions[index][0], self.actions[index][1]
        for (i, depth_index) in enumerate(self.depth_indices[index]):
            depth = np.array(Image.open(depth_index))
            depth = self.preprocess(depth)
            depth = torch.FloatTensor(depth).unsqueeze(0)
            if i == 0 and self.spatial_augment:
                angle = np.random.randint(-5, 6)
                dx = np.random.randint(-5, 6)
                dy = np.random.randint(-5, 6)
                depth, pick_pixel, place_pixel = self.aug_spatial(
                    depth, self.actions[index][0], self.actions[index][1], angle, dx, dy
                )
            depths.append(depth.unsqueeze(0))
        depths = torch.cat(depths, dim=0)

        # labels
        pick_map = make_gaussmap(pick_pixel, self.img_size)
        place_map = make_gaussmap(place_pixel, self.img_size)
        pick_map = torch.tensor(pick_map)
        place_map = torch.tensor(place_map)

        #  rearrange
        rgbs = rearrange(depths, "t c h w -> c t h w")

        return (rgbs, pick_map, place_map)

    def preprocess(self, depth):
        depth = depth / 255
        # generate a mask
        mask = depth.copy()
        mask[mask == depth.max()] = 0
        mask[mask != 0] = 1
        depth = depth * mask
        return depth

    def aug_spatial(self, img, pick, place, angle, dx, dy):
        img = F.affine(img, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        pick = self.aug_pixel(pick.astype(np.float64)[None, :], -angle, dx, dy, size=self.img_size - 1)
        pick = pick.squeeze().astype(int)
        place = self.aug_pixel(place.astype(np.float64)[None, :], -angle, dx, dy, size=self.img_size - 1)
        place = place.squeeze().astype(int)
        return img, pick, place

    def aug_pixel(self, pixel, angle, dx, dy, size):
        rad = np.deg2rad(-angle)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pixel -= size / 2
        pixel = np.dot(R, pixel.T).T
        pixel += size / 2
        pixel[:, 0] += dx
        pixel[:, 1] += dy
        pixel = np.clip(pixel, 0, size)
        return pixel
