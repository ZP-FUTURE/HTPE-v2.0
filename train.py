import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from data.dataset import PendulumDataset
from utils.functions import load_model_class
from utils.common import set_seed, count_parameters

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log = logging.getLogger("Train")
    set_seed(42)
    
    # 1. 准备数据
    log.info(f"📂 Loading Dataset from: {cfg.generation.save_dir}")
    try:
        full_ds = PendulumDataset(cfg.generation.save_dir)
    except FileNotFoundError as e:
        log.error(e)
        return

    val_size = int(len(full_ds) * cfg.train.val_split)
    train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.global_batch_size, shuffle=True, 
        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.global_batch_size, shuffle=False, 
        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory
    )
    
    # 2. 加载模型
    log.info(f"🏗️  Initializing Model: {cfg.model.identifier}")
    ModelClass = load_model_class(cfg.model.identifier)
    model = ModelClass(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info(f"🧠 Parameters: {count_parameters(model):,}")
    
    # 3. 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=(cfg.train.optimizer.beta1, cfg.train.optimizer.beta2),
        weight_decay=cfg.train.optimizer.weight_decay
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.train.optimizer.lr * 10,
        total_steps=cfg.train.epochs * len(train_loader), pct_start=0.1
    )
    
    criterion = nn.MSELoss()
    
    # 4. 训练循环
    best_loss = float('inf')
    
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证 (每 5 Epoch)
        if epoch % 5 == 0:
            model.eval()
            err_k1, err_k2, err_n = 0, 0, 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    err_k1 += nn.functional.mse_loss(pred[:,0], y[:,0]).item()
                    err_k2 += nn.functional.mse_loss(pred[:,1], y[:,1]).item()
                    err_n  += nn.functional.mse_loss(pred[:,2], y[:,2]).item()
            
            steps = len(val_loader)
            avg_val_loss = (err_k1 + err_k2 + err_n) / 3 / steps # 简单平均
            
            log.info(f"Ep {epoch} | Train: {avg_train_loss:.6f} | Val MSE: K1 {err_k1/steps:.6f}, K2 {err_k2/steps:.6f}")
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), "best.pt")

if __name__ == "__main__":
    main()