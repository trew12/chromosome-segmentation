from configs.config import SegmentParams
from dataset import transform
from model import get_model

import cv2
import hydra
import numpy as np
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
colours = (
        (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
        (0.75, 0, 0), (0, 0.75, 0), (0, 0, 0.75), (0.75, 0.75, 0), (0.75, 0, 0.75), (0, 0.75, 0.75),
        (0.25, 0, 0), (0, 0.25, 0), (0, 0, 0.25), (0.25, 0.25, 0), (0.25, 0, 0.25), (0, 0.25, 0.25)
)


def get_coloured_mask(mask, label):
    r = np.zeros_like(mask).astype(np.float32)
    g = np.zeros_like(mask).astype(np.float32)
    b = np.zeros_like(mask).astype(np.float32)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[label - 1]
    coloured_mask = np.stack([r, g, b], axis=2)
    
    return coloured_mask


def get_prediction(img, model, mask_threshold=0.5, confidence=0.5):
    img = img.to(device)
    model.eval()
    with torch.no_grad():
        pred = model([img])[0]
    masks = (pred['masks'] > mask_threshold).cpu().squeeze().numpy()
    pred_class = pred['labels'].cpu().numpy()
    pred_boxes = pred['boxes'].cpu().numpy()
    pred_t = pred['scores'].cpu() > confidence
    masks = masks[pred_t]
    pred_boxes = pred_boxes[pred_t]
    pred_class = pred_class[pred_t]
    
    return masks, pred_boxes, pred_class


def segment_instance(img, masks, boxes, label, rect_thickness=1, text_size=3, text_thickness=2):
    img = img.numpy().transpose(1, 2, 0)
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i], label[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        x1, y1, x2, y2 = map(round, boxes[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 1, 0), thickness=rect_thickness)
        cv2.putText(img, str(label[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_thickness)
        
    return img


@hydra.main(version_base=None, config_path='configs', config_name='segment_config')
def segment_image(cfg: SegmentParams):
    img = cv2.imread(cfg.img_path)
    print(img.shape)
    img = transform(image=img)['image'] / 255
    model = get_model(cfg.model_params)
    if device == torch.device('cuda'):
        model_state_dict = torch.load(cfg.model_path)
    else:
        model_state_dict = torch.load(cfg.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    model.to(device)
    masks, boxes, pred_cls = get_prediction(img, model, cfg.mask_threshold, cfg.confidence)
    segmented_img = segment_instance(img, masks, boxes, pred_cls)
    cv2.imwrite(cfg.save_path, segmented_img * 255)


if __name__ == "__main__":
    segment_image()
