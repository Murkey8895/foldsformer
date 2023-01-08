import argparse
import numpy as np
from softgym.envs.foldenv import FoldEnv
from utils.visual import get_world_coord_from_pixel, action_viz, nearest_to_mask
import pyflex
from utils.setup_model import get_configs, setup_model
import torch
import os
import pickle
from tqdm import tqdm
from einops import rearrange
from utils.load_configs import get_configs
import imageio


def get_mask(depth):
    mask = depth.copy()
    mask[mask > 0.646] = 0
    mask[mask != 0] = 1
    return mask


def preprocess(depth):
    mask = get_mask(depth)
    depth = depth * mask
    return depth


def main():
    parser = argparse.ArgumentParser(description="Evaluate Foldsformer")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--task", type=str, default="DoubleTriangle", help="Task name")
    parser.add_argument("--img_size", type=int, default=224, help="Size of rendered image")
    parser.add_argument("--model_config", type=str, help="Evaluate which model")
    parser.add_argument("--model_file", type=str, help="Evaluate which trained model")
    parser.add_argument("--cached", type=str, help="Cached filename")
    args = parser.parse_args()

    # task
    task = args.task
    if task == "CornersEdgesInward":
        frames_idx = [0, 1, 2, 3, 4]
        steps = 4
    elif task == "AllCornersInward":
        frames_idx = [0, 1, 2, 3, 4]
        steps = 4
    elif task == "DoubleStraight":
        frames_idx = [0, 1, 2, 3, 3]
        steps = 3
    elif task == "DoubleTriangle":
        frames_idx = [0, 1, 1, 2, 2]
        steps = 2

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # create transformer model & load parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config_path = os.path.join("train", "train configs", args.model_config + ".yaml")
    configs = get_configs(model_config_path)
    trained_model_path = os.path.join("train", "trained model", configs["save_model_name"], args.model_file + ".pth")
    net = setup_model(configs)
    net = net.to(device)
    net.load_state_dict(torch.load(trained_model_path)["model"])
    print(f"load trained model from {trained_model_path}")
    net.eval()

    # set goal
    depth_load_path = os.path.join("data", "demo", args.task, "depth")
    goal_frames = []
    for i in frames_idx:
        frame = imageio.imread(os.path.join(depth_load_path, str(i) + ".png")) / 255
        frame = torch.FloatTensor(preprocess(frame)).unsqueeze(0).unsqueeze(0)
        goal_frames.append(frame)
    goal_frames = torch.cat(goal_frames, dim=0)

    for config_id in tqdm(range(env.num_configs)):
        rgb_save_path = os.path.join("eval result", args.task, str(config_id), "rgb")
        depth_save_path = os.path.join("eval result", args.task, str(config_id), "depth")
        if not os.path.exists(rgb_save_path):
            os.makedirs(rgb_save_path)
        if not os.path.exists(depth_save_path):
            os.makedirs(depth_save_path)

        # record action's pixel info
        test_pick_pixels = []
        test_place_pixels = []
        rgbs = []

        # env settings
        env.reset(config_id=config_id)
        camera_params = env.camera_params

        # initial state
        rgb, depth = env.render_image()
        depth_save = depth.copy() * 255
        depth_save = depth_save.astype(np.uint8)
        imageio.imwrite(os.path.join(depth_save_path, "0.png"), depth_save)
        imageio.imwrite(os.path.join(rgb_save_path, "0.png"), rgb)
        rgbs.append(rgb)

        for i in range(steps):
            current_state = torch.FloatTensor(preprocess(depth)).unsqueeze(0).unsqueeze(0)
            current_frames = torch.cat((current_state, goal_frames), dim=0).unsqueeze(0)
            current_frames = rearrange(current_frames, "b t c h w -> b c t h w")
            current_frames = current_frames.to(device)

            # get action
            pickmap, placemap = net(current_frames)
            pickmap = torch.sigmoid(torch.squeeze(pickmap))
            placemap = torch.sigmoid(torch.squeeze(placemap))
            pickmap = pickmap.detach().cpu().numpy()
            placemap = placemap.detach().cpu().numpy()

            test_pick_pixel = np.array(np.unravel_index(pickmap.argmax(), pickmap.shape))
            test_place_pixel = np.array(np.unravel_index(placemap.argmax(), placemap.shape))

            mask = get_mask(depth)
            test_pick_pixel_mask = nearest_to_mask(test_pick_pixel[0], test_pick_pixel[1], mask)
            test_pick_pixel[0], test_pick_pixel[1] = test_pick_pixel_mask[1], test_pick_pixel_mask[0]
            test_place_pixel[0], test_place_pixel[1] = test_place_pixel[1], test_place_pixel[0]
            test_pick_pixels.append(test_pick_pixel)
            test_place_pixels.append(test_place_pixel)

            # convert the pixel cords into world cords
            test_pick_pos = get_world_coord_from_pixel(test_pick_pixel, depth, camera_params)
            test_place_pos = get_world_coord_from_pixel(test_place_pixel, depth, camera_params)

            # pick & place
            env.pick_and_place(test_pick_pos.copy(), test_place_pos.copy())

            # render & update frames & save
            rgb, depth = env.render_image()
            depth_save = depth.copy() * 255
            depth_save = depth_save.astype(np.uint8)
            imageio.imwrite(os.path.join(depth_save_path, str(i + 1) + ".png"), depth_save)
            imageio.imwrite(os.path.join(rgb_save_path, str(i + 1) + ".png"), rgb)
            rgbs.append(rgb)

        particle_pos = pyflex.get_positions().reshape(-1, 4)[:, :3]
        with open(os.path.join("eval result", args.task, str(config_id), "info.pkl"), "wb+") as f:
            data = {"pick": test_pick_pixels, "place": test_place_pixels, "pos": particle_pos}
            pickle.dump(data, f)

        # action viz
        save_folder = os.path.join("eval result", args.task, str(config_id), "rgbviz")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for i in range(steps + 1):
            if i < steps:
                img = action_viz(rgbs[i], test_pick_pixels[i], test_place_pixels[i])
            else:
                img = rgbs[i]
            imageio.imwrite(os.path.join(save_folder, str(i) + ".png"), img)


if __name__ == "__main__":
    main()
