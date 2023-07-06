from configs.config import TestParams
from dataset import ChromosomeDataset, get_dataloader
from model import get_model

import hydra
import logging
import mlflow
import omegaconf
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


log = logging.getLogger(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@hydra.main(version_base=None, config_path='configs', config_name='test_config')
def test(cfg: TestParams):
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(cfg_dict)
    model = get_model(cfg.model_params).to(device)
    dataloader = get_dataloader(cfg.dataset_params)
    if device == torch.device('cuda'):
        model_state_dict = torch.load(cfg.model_path)
    else:
        model_state_dict = torch.load(cfg.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.to(device)
    metric_seg = MeanAveragePrecision(box_format='xyxy', iou_type='segm')
    metric_box = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images = [image.to(device) for image in data[0]]
            targets = [{k: v.to(device) for k, v in target.items()} for target in data[1]]
            pred = model(images, targets)
            for p in pred:
                p['masks'] = torch.squeeze((p['masks'] > cfg.mask_threshold), 1)
            metric_seg.update(pred, targets)
            metric_box.update(pred, targets)
            torch.cuda.empty_cache()
    seg_metric = metric_seg.compute()
    box_metric = metric_box.compute()
    mlflow.log_metrics({'map_seg': float(seg_metric['map']), 'map50_seg': float(seg_metric['map_50']), 'map75_seg': float(seg_metric['map_75'])})
    mlflow.log_metrics({'map_box': float(box_metric['map']), 'map50_box': float(box_metric['map_50']), 'map75_box': float(box_metric['map_75'])})


if __name__ == "__main__":
    test()
