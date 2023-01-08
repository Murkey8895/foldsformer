import torch
import sys

sys.path.append("")
from Model.net import Foldsformer
from utils.load_configs import get_configs


class ModelConfig:
    def __init__(self, configs):
        self.img_size = configs["img_size"]
        self.patch_size = configs["patch_size"]
        self.embed_dim = configs["embed_dim"]
        self.depth = configs["depth"]
        self.num_heads = configs["num_heads"]
        self.mlp_ratio = configs["mlp_ratio"]
        self.drop_rate = configs["drop_rate"]
        self.attn_drop_rate = configs["attn_drop_rate"]
        self.drop_path_rate = configs["drop_path_rate"]
        self.num_frames = configs["num_frames"]


def setup_model(configs):
    cfg = ModelConfig(configs)
    Model = Foldsformer(cfg)
    total_params = sum(p.numel() for p in Model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    return Model


def construct_optimizer(model, configs):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(configs["lr"]),
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=float(configs["weight_decay"]),
    )
    return optimizer


if __name__ == "__main__":
    filepath = "train/train configs/train.yaml"
    configs = get_configs(filepath)
    Model = setup_model(configs)
    optimizer = construct_optimizer(Model, configs)
    print(Model)
    print(optimizer)
