from configs.config import TrainParams
from dataset import ChromosomeDataset, get_dataloader
from model import get_model

import hydra
import mlflow
import omegaconf
import torch
from torch.optim.lr_scheduler import ExponentialLR


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


@hydra.main(version_base=None, config_path='configs', config_name='train_config')
def train(cfg: TrainParams):
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(cfg_dict)
    model = get_model(cfg.model_params).to(device)
    train_loader = get_dataloader(cfg.train_dataset_params)
    val_loader = get_dataloader(cfg.val_dataset_params)
    
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = ExponentialLR(optimizer, gamma=cfg.scheduler_gamma)
    
    all_train_losses = []
    all_val_losses = []
    for epoch in range(cfg.num_epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            images = [image.to(device) for image in data[0]]
            targets = [{k: v.to(device) for k, v in target.items()} for target in data[1]]
            optimizer.zero_grad()
            loss = model(images, targets)
            losses = sum([l for l in loss.values()])
            train_epoch_loss += losses.cpu().detach().numpy()
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            mlflow.log_metric('Train loss', losses.cpu(), step=i)
            torch.cuda.empty_cache()
        scheduler.step()
        mlflow.log_metric('Train loss by epoch', train_epoch_loss / len(train_loader), step=epoch)
        if (epoch + 1) % cfg.dump_params.dump_every_k_epoch == 0 and epoch > 1:
            torch.save(model.state_dict(), cfg.dump_params.dump_model_name + f'-{epoch + 1}.pth')
        
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                images = [image.to(device) for image in data[0]]
                targets = [{k: v.to(device) for k, v in target.items()} for target in data[1]]
                loss = model(images, targets)
                losses = sum([l for l in loss.values()])
                val_epoch_loss += losses.cpu().detach().numpy()
                mlflow.log_metric('Val loss', losses.cpu(), step=j)
                torch.cuda.empty_cache()
        mlflow.log_metric('Val loss by epoch', val_epoch_loss / len(val_loader), step=epoch)
        
        
if __name__ == "__main__":
    train()
