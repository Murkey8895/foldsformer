import os
import pickle
import numpy as np


def create_frames_list(folder_path, demo_frames):
    rgbs_list = []
    depths_list = []
    actions_list = []
    with open(os.path.join(folder_path, "info.pkl"), "rb") as f:
        action_data = pickle.load(f)
        pick_pixels = action_data["pick"]
        place_pixels = action_data["place"]
    num_groups = len(pick_pixels)

    for group_idx in range(num_groups - demo_frames + 2):
        demo_indices = [idx for idx in range(group_idx, group_idx + demo_frames)]
        for state_idx in range(group_idx, group_idx + demo_frames - 1):
            rgbs_sublist = []
            depths_sublist = []
            total_indices = demo_indices.copy()
            total_indices.insert(0, state_idx)
            for i in total_indices:
                rgb_path = os.path.join(folder_path, "rgb", str(i) + ".png")
                depth_path = os.path.join(folder_path, "depth", str(i) + ".png")
                rgbs_sublist.append(rgb_path)
                depths_sublist.append(depth_path)
            rgbs_list.append(rgbs_sublist)
            depths_list.append(depths_sublist)
            actions_list.append((pick_pixels[state_idx], place_pixels[state_idx]))

    return rgbs_list, depths_list, actions_list


def split_datatset(rgbs_list, depths_list, actions_list, ratio=0.95):
    num_data = len(rgbs_list)
    num_train_data = round(num_data * ratio)
    num_test_data = num_data - num_train_data
    train_rgbs_list = rgbs_list[:num_train_data]
    train_depths_list = depths_list[:num_train_data]
    train_actions_list = actions_list[:num_train_data]

    test_rgbs_list = rgbs_list[-num_test_data:]
    test_depths_list = depths_list[-num_test_data:]
    test_actions_list = actions_list[-num_test_data:]

    return train_rgbs_list, train_depths_list, train_actions_list, test_rgbs_list, test_depths_list, test_actions_list


# use with split_dataset
def shuffle_dataset(rgbs_list, depths_list, actions_list):
    shuffle_rgbs_list = []
    shuffle_depths_list = []
    shuffle_actions_list = []
    random_index = np.arange(len(rgbs_list))
    np.random.shuffle(random_index)
    for index in random_index:
        shuffle_rgbs_list.append(rgbs_list[index])
        shuffle_depths_list.append(depths_list[index])
        shuffle_actions_list.append(actions_list[index])
    return shuffle_rgbs_list, shuffle_depths_list, shuffle_actions_list


if __name__ == "__main__":
    root = "data/random"
    save_file_path = "data/data index"
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    demo_frames = 5
    save_file_folder = os.path.join(save_file_path, "train" + str(demo_frames))
    if not os.path.exists(save_file_folder):
        os.makedirs(save_file_folder)

    save_trainfile_name = os.path.join(save_file_folder, "TrainIndex.pkl")
    save_testfile_name = os.path.join(save_file_folder, "TestIndex.pkl")

    #  original data
    total_rgbs_list = []
    total_depths_list = []
    total_actions_list = []
    task_names = os.listdir(root)
    for task_name in task_names:
        task_path = os.path.join(root, task_name)
        folder_indices = os.listdir(task_path)
        for folder_index in folder_indices:
            folder_path = os.path.join(task_path, folder_index)
            rgbs_list, depths_list, actions_list = create_frames_list(folder_path, demo_frames)
            total_rgbs_list.extend(rgbs_list)
            total_depths_list.extend(depths_list)
            total_actions_list.extend(actions_list)

    #  shuffle data
    shuffle_rgbs_list, shuffle_depths_list, shuffle_actions_list = shuffle_dataset(
        total_rgbs_list, total_depths_list, total_actions_list
    )

    # split data
    (
        train_rgbs_list,
        train_depths_list,
        train_actions_list,
        test_rgbs_list,
        test_depths_list,
        test_actions_list,
    ) = split_datatset(shuffle_rgbs_list, shuffle_depths_list, shuffle_actions_list)

    # train data
    with open(save_trainfile_name, "wb+") as f_train:
        traindata = {"rgb": train_rgbs_list, "depth": train_depths_list, "action": train_actions_list}
        pickle.dump(traindata, f_train)

    # test data
    with open(save_testfile_name, "wb+") as f_test:
        testdata = {"rgb": test_rgbs_list, "depth": test_depths_list, "action": test_actions_list}
        pickle.dump(testdata, f_test)
