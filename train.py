import torch
import torch.nn as nn
import os
from utils.setup_model import setup_model, construct_optimizer
from utils.clothdataset import ClothDataSet
from torch.utils.data import DataLoader
from utils.load_configs import get_configs
import pickle
from tqdm import tqdm
import argparse


def train_net(configs):
    num_workers = configs["num_workers"]
    batch_size = configs["batch_size"]
    img_size = configs["img_size"]
    epochs = configs["epochs"]
    train_data_index_path = configs["train_data_index_path"]
    test_data_index_path = configs["test_data_index_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("training net on" + " " + device)

    # create folder to save trained model
    save_dir = configs["save_dir"]
    save_model_name = configs["save_model_name"]
    save_folder = os.path.join(save_dir, save_model_name)
    os.makedirs(save_folder, exist_ok=True)

    # create network && optimizer
    model = setup_model(configs)
    model = model.to(device)
    optimizer = construct_optimizer(model, configs)

    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Dataloader
    train_dataset = ClothDataSet(train_data_index_path, img_size, spatial_augment=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = ClothDataSet(test_data_index_path, img_size, spatial_augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    train_loss = []
    test_loss = []
    num_train_batch = len(train_dataloader)
    num_test_batch = len(test_dataloader)

    for epoch in tqdm(range(0, epochs)):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_dataloader)
        for batch, (depths, pick_map, place_map) in enumerate(train_bar):
            depths, pick_map, place_map = (
                depths.to(device),
                pick_map.to(device),
                place_map.to(device),
            )

            # forward
            pick_pred_map, place_pred_map = model(depths)
            pick_pred_map = pick_pred_map.squeeze(1)
            place_pred_map = place_pred_map.squeeze(1)

            loss_pick = loss_fn(pick_pred_map, pick_map)
            loss_place = loss_fn(place_pred_map, place_map)
            loss = loss_pick + loss_place

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calc total loss
            total_train_loss += loss.item()
            train_bar.set_description("loss:{}".format(total_train_loss / (batch + 1)))

        average_train_loss = total_train_loss / num_train_batch
        train_loss.append(average_train_loss)

        # runing on the test dataset
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch, (depths, pick_map, place_map) in enumerate(test_dataloader):
                depths, pick_map, place_map = (
                    depths.to(device),
                    pick_map.to(device),
                    place_map.to(device),
                )

                # forward
                pick_pred_map, place_pred_map = model(depths)
                pick_pred_map = pick_pred_map.squeeze(1)
                place_pred_map = place_pred_map.squeeze(1)

                loss_pick = loss_fn(pick_pred_map, pick_map)
                loss_place = loss_fn(place_pred_map, place_map)
                loss = loss_pick + loss_place

                # calc total loss
                total_test_loss += loss.item()

        average_test_loss = total_test_loss / num_test_batch
        test_loss.append(average_test_loss)

        # save model
        if (epoch + 1) % 5 == 0:
            model_path = os.path.join(save_folder, f"epoch{epoch}.pth")
            model_state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(model_state_dict, model_path)

    with open(os.path.join(save_folder, "loss_info.pkl"), "wb+") as f:
        loss_info = {"train_loss": train_loss, "test_loss": test_loss}
        pickle.dump(loss_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    # Oracle demonstration
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--config_path", type=str, help="train configs path")
    args = parser.parse_args()

    config_path = os.path.join("train/train configs", args.config_path + ".yaml")
    configs = get_configs(config_path)
    print(configs["save_model_name"])
    train_net(configs)
