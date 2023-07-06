from configs.config import ModelParams

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def get_model(cfg: ModelParams):
    if cfg.trainable_backbone_layers != 0:
        model = maskrcnn_resnet50_fpn_v2(weights_backbone=ResNet50_Weights.IMAGENET1K_V2, trainable_backbone_layers=cfg.trainable_backbone_layers, num_classes=cfg.num_classes)
    else:
        model = maskrcnn_resnet50_fpn_v2(num_classes=cfg.num_classes)

    model.transform = GeneralizedRCNNTransform(
        min_size=cfg.transform_params.min_size,
        max_size=cfg.transform_params.max_size,
        image_mean=cfg.transform_params.image_mean,
        image_std=cfg.transform_params.image_std
    )
    
    return model
