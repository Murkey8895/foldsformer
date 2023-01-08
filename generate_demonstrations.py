import argparse
import numpy as np
from utils.visual import get_pixel_coord_from_world
import pyflex
import os
from tqdm import tqdm
import imageio
import pickle
from utils.visual import action_viz
from softgym.envs.foldenv import FoldEnv
from Demonstrator.demonstrator import Demonstrator


def main():
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=224, help="Size of rendered image")
    parser.add_argument("--cached", type=str, help="Cached filename")
    args = parser.parse_args()

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # demonstrator settings
    demonstrator = Demonstrator[args.task]()

    # save settings
    save_path = os.path.join("data", "demonstrations", args.task)
    os.makedirs(save_path, exist_ok=True)

    # other settings
    rgb_shape = (args.img_size, args.img_size)
    num_data = env.num_configs

    for config_id in tqdm(range(num_data)):
        # folders
        save_folder = os.path.join(save_path, str(config_id))
        save_folder_rgb = os.path.join(save_folder, "rgb")
        save_folder_depth = os.path.join(save_folder, "depth")
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(save_folder_rgb, exist_ok=True)
        os.makedirs(save_folder_depth, exist_ok=True)

        pick_pixels = []
        place_pixels = []
        rgbs = []

        # env reset
        env.reset(config_id=config_id)
        camera_params = env.camera_params
        rgb, depth = env.render_image()
        imageio.imwrite(os.path.join(save_folder_rgb, str(0) + ".png"), rgb)
        depth = depth * 255
        depth = depth.astype(np.uint8)
        imageio.imwrite(os.path.join(save_folder_depth, str(0) + ".png"), depth)
        rgbs.append(rgb)

        if args.task == "DoubleTriangle":
            pick_idxs = demonstrator.pick_idxs
            for i, pick_idx in enumerate(pick_idxs):
                curr_corners = env.get_corners()
                pick_pos, place_pos = demonstrator.get_action(curr_corners, pick_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        elif args.task == "AllCornersInward":
            pick_idxs = np.arange(4)
            curr_corners = env.get_corners()
            center = env.get_center()
            for (i, pick_idx) in enumerate(pick_idxs):
                pick_pos, place_pos = demonstrator.get_action(curr_corners, center, pick_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        elif args.task == "CornersEdgesInward":
            center = env.get_center()
            for (i, pickplace_idx) in enumerate(demonstrator.pickplace_idxs):
                curr_corners = env.get_corners()
                edge_middles = env.get_edge_middles()
                pick_pos, place_pos = demonstrator.get_action(curr_corners, edge_middles, center, pickplace_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        elif args.task == "DoubleStraight":
            pickplace_idxs = demonstrator.pickplace_idxs
            for (i, pickplace_idx) in enumerate(pickplace_idxs):
                curr_corners = env.get_corners()
                edge_middles = env.get_edge_middles()
                pick_pos, place_pos = demonstrator.get_action(curr_corners, edge_middles, pickplace_idx)
                pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                env.pick_and_place(pick_pos.copy(), place_pos.copy())
                rgb, depth = env.render_image()

                # save
                pick_pixels.append(pick_pixel)
                place_pixels.append(place_pixel)
                imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                depth = depth * 255
                depth = depth.astype(np.uint8)
                imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                rgbs.append(rgb)

        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]

        with open(os.path.join(save_folder, "info.pkl"), "wb+") as f:
            data = {"pick": pick_pixels, "place": place_pixels, "pos": particle_pos}
            pickle.dump(data, f)

        # action viz
        save_folder_viz = os.path.join(save_folder, "rgbviz")
        os.makedirs(save_folder_viz, exist_ok=True)

        num_actions = len(pick_pixels)

        for i in range(num_actions + 1):
            if i < num_actions:
                img = action_viz(rgbs[i], pick_pixels[i], place_pixels[i])
            else:
                img = rgbs[i]
            imageio.imwrite(os.path.join(save_folder_viz, str(i) + ".png"), img)


if __name__ == "__main__":
    main()
