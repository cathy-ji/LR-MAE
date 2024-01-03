# Copyright (c) Facebook, Inc. and its affiliates.
from .model_umae import build_umae
from .model_3detr import build_3detr
from .model_umae_3detr import build_umae_3detr
from .model_umae_all import build_umae_all
from .model_upmae import build_upmae
from .model_3detr_upmae import build_3detr_upmae

from .model_ablation import build_ablation
MODEL_FUNCS = {
    "3detr": build_3detr,
    "umae_3detr": build_umae_3detr,#
    "umae": build_umae,#
    "umae_all": build_umae_all,
    "up_mae":build_upmae,#2023-10-23
    "up_mae_3detr":build_3detr_upmae,#2023-10-25
    "ablation_mae":build_ablation,#2023-11-20
}

def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor