from dataclasses import dataclass
from typing import List


@dataclass
class DatasetParams:
    use_aug: bool
    shuffle: bool
    batch_size: int
    annotations_folder: str
    image_folder: str


@dataclass
class DumpParams:
    dump_model_name: str
    dump_every_k_epoch: int

    
@dataclass
class TransformParams:
    min_size: int
    max_size: int
    image_mean: List[float]
    image_std: List[float]


@dataclass 
class ModelParams:
    num_classes:int 
    trainable_backbone_layers: int
    transform_params: TransformParams


@dataclass
class TrainParams:
    num_epochs: int
    scheduler_gamma: float
    use_adamw: bool
    lr: float
    
    model_params: ModelParams
    train_dataset_params: DatasetParams
    val_dataset_params: DatasetParams
    dump_params: DumpParams


@dataclass
class TestParams:
    mask_threshold: float
    model_path: str
    model_params: ModelParams
    dataset_params: DatasetParams
    

@dataclass
class PredictParams:
    input_data_path: str
    model_path: str
    save_path: str


@dataclass
class SegmentParams:
    img_path: str
    save_path: str
    mask_threshold: float
    confidence: float
    model_path: str
    model_params: ModelParams
