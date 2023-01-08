import argparse
import numpy as np
from utils.visual import get_pixel_coord_from_world
import pyflex
import os
from tqdm import tqdm
import imageio
import pickle
from softgym.envs.foldenv import FoldEnv


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # Oracle demonstration
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--corner_bias", action="store_true", help="Task name")
    parser.add_argument("--img_size", type=int, default=224, help="Size of rendered image")
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument("--horizon", type=int, default=8, help="Number of horizons in a episode")
    args = parser.parse_args()

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # save settings
    save_path = os.path.join("data", "random", "corner bias" if args.corner_bias else "random")
    os.makedirs(save_path, exist_ok=True)

    # other settings
    rgb_shape = (args.img_size, args.img_size)
    num_data = env.num_configs

    dirs = os.listdir(save_path)
    if dirs == []:
        max_index = -1
    else:
        existed_index = np.array(dirs).astype(np.int)
        max_index = existed_index.max()

    for config_id in tqdm(range(num_data)):
        # folders
        save_folder = os.path.join(save_path, str(config_id + max_index + 1))
        save_folder_rgb = os.path.join(save_folder, "rgb")
        save_folder_depth = os.path.join(save_folder, "depth")
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_rgb, exist_ok=True)
        os.makedirs(save_folder_depth, exist_ok=True)

        pick_pixels = []
        place_pixels = []

        # env reset
        env.reset(config_id=config_id)
        camera_params = env.camera_params
        rgb, depth = env.render_image()
        imageio.imwrite(os.path.join(save_folder_rgb, str(0) + ".png"), rgb)
        depth = depth * 255
        depth = depth.astype(np.uint8)
        imageio.imwrite(os.path.join(save_folder_depth, str(0) + ".png"), depth)

        center = np.zeros(3)
        if args.corner_bias:
            # corner bias
            for i in range(args.horizon):
                # set direction
                corners = env.get_corners()
                pick_pos = corners[np.random.randint(0, 4)]

                diff = center - pick_pos
                direction = np.where(diff >= 0, 1, -1)

                random_action = np.random.uniform(0.05, 0.3, (3,))
                random_action = random_action * direction
                random_action[1] = 0

                place_pos = pick_pos + random_action
                place_pos = np.clip(place_pos, -0.4, 0.4)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())

                rgb, depth = env.render_image()
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)

        else:
            # random action
            for i in range(args.horizon):
                particle_pos = env.action_tool._get_pos()[1]
                pick_pos = particle_pos[np.random.randint(particle_pos.shape[0])][0:3]

                diff = center - pick_pos
                direction = np.where(diff >= 0, 1, -1)
                random_action = np.random.uniform(0.05, 0.2, (3,))
                random_action = random_action * direction
                random_action[1] = 0

                place_pos = pick_pos + random_action
                place_pos = np.clip(place_pos, -0.4, 0.4)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())

                rgb, depth = env.render_image()
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)

        with open(os.path.join(save_folder, "info.pkl"), "wb+") as f:
            data = {"pick": pick_pixels, "place": place_pixels}
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
