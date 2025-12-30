import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import os
import time

# 引入 Rich 库组件
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeRemainingColumn, 
    SpinnerColumn
)
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

# 引入项目模块
from data.dataset import PendulumDataset
from utils.functions import load_model_class
from utils.common import set_seed, count_parameters

# 初始化 Rich 控制台
console = Console()

def visualize_config(cfg: DictConfig):
    """打印漂亮的参数配置表"""
    table = Table(title="🔬 Experiment Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="bold white")

    # 遍历配置并添加到表格
    conf_dict = OmegaConf.to_container(cfg, resolve=True)
    
    for section, params in conf_dict.items():
        if isinstance(params, dict):
            first = True
            for k, v in params.items():
                # 简化显示 list
                if isinstance(v, list): v = str(v)
                table.add_row(section if first else "", k, str(v))
                first = False
            table.add_section()
            
    console.print(table)

def visualize_metrics(epoch, train_loss, val_loss, k1_err, k2_err, n_err, lr):
    """打印 Epoch 结果表格"""
    table = Table(box=None, show_header=True)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    # 颜色逻辑：Loss 越小越绿
    loss_color = "green" if val_loss < 0.001 else "yellow"
    
    table.add_row("Train Loss", f"{train_loss:.6f}")
    table.add_row("Val Loss", f"[{loss_color}]{val_loss:.6f}[/]")
    table.add_row("K1 Error (Friction)", f"{k1_err:.6f}")
    table.add_row("K2 Error (Drag)", f"{k2_err:.6f}")
    table.add_row("Noise Error", f"{n_err:.6f}")
    table.add_row("Learning Rate", f"{lr:.2e}")
    
    panel = Panel(
        table, 
        title=f"Epoch {epoch} Summary", 
        border_style="blue",
        expand=False
    )
    console.print(panel)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 0. 启动画面
    console.print(Panel.fit(
        "[bold cyan]Hybrid Physics-AI Engine[/bold cyan]\n"
        "[dim]Temporal Folding & Resonant Encoding Architecture[/dim]",
        border_style="blue"
    ))
    
    set_seed(42)
    visualize_config(cfg)
    
    # 1. 准备数据
    console.rule("[bold yellow]Data Loading[/bold yellow]")
    data_dir = cfg.generation.save_dir
    
    try:
        with console.status(f"[bold green]Loading dataset from {data_dir}..."):
            full_ds = PendulumDataset(data_dir)
            
        val_size = int(len(full_ds) * cfg.train.val_split)
        train_ds, val_ds = random_split(full_ds, [len(full_ds)-val_size, val_size])
        
        console.print(f"✅ Dataset Loaded: [bold]{len(full_ds)}[/] total samples")
        console.print(f"   ├─ Train: [blue]{len(train_ds)}[/]")
        console.print(f"   └─ Val:   [magenta]{len(val_ds)}[/]")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.global_batch_size, shuffle=True, 
        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.global_batch_size, shuffle=False, 
        num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory
    )
    
    # 2. 加载模型
    console.rule("[bold yellow]Model Initialization[/bold yellow]")
    ModelClass = load_model_class(cfg.model.identifier)
    model = ModelClass(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 打印简化的模型结构信息
    param_count = count_parameters(model)
    console.print(f"🏗️  Model Architecture: [bold cyan]{cfg.model.identifier}[/]")
    console.print(f"🧠 Trainable Parameters: [bold green]{param_count:,}[/]")
    console.print(f"🔌 Device: [bold red]{device}[/]")
    
    # 3. 优化器 & 调度器
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
    
    # 4. 训练循环 (带高级进度条)
    console.rule("[bold yellow]Training Start[/bold yellow]")
    best_loss = float('inf')
    
    # 定义进度条布局
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("[bold blue]{task.fields[info]}", justify="right"),
    )

    with progress:
        # 总 Epoch 进度条
        epoch_task = progress.add_task("[green]Total Progress", total=cfg.train.epochs, info="Starting...")
        
        for epoch in range(cfg.train.epochs):
            model.train()
            total_loss = 0
            
            # Batch 进度条 (每轮创建，每轮销毁)
            batch_task = progress.add_task(f"Epoch {epoch+1}", total=len(train_loader), info="Loss: 0.000")
            
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                # 更新 Batch 进度条的 Loss 显示
                progress.update(batch_task, advance=1, info=f"Loss: {loss.item():.5f}")
            
            # 移除 Batch 进度条，保持界面整洁
            progress.remove_task(batch_task)
            
            avg_train_loss = total_loss / len(train_loader)
            
            # === 验证阶段 ===
            # 每隔几个 Epoch 验证一次 (或者每个都验证，取决于速度)
            if epoch % cfg.train.eval_interval == 0 or epoch == cfg.train.epochs - 1:
                model.eval()
                err_k1, err_k2, err_n = 0, 0, 0
                val_loss_accum = 0
                
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        pred = model(x)
                        val_loss_accum += criterion(pred, y).item()
                        
                        # 统计分量误差 (MSE)
                        err_k1 += nn.functional.mse_loss(pred[:,0], y[:,0]).item()
                        err_k2 += nn.functional.mse_loss(pred[:,1], y[:,1]).item()
                        err_n  += nn.functional.mse_loss(pred[:,2], y[:,2]).item()
                
                steps = len(val_loader)
                avg_val_loss = val_loss_accum / steps
                avg_k1 = err_k1 / steps
                avg_k2 = err_k2 / steps
                avg_n = err_n / steps
                
                # 暂时停止进度条刷新，打印统计表格
                progress.stop() 
                visualize_metrics(epoch+1, avg_train_loss, avg_val_loss, avg_k1, avg_k2, avg_n, optimizer.param_groups[0]['lr'])
                progress.start()

                # 保存最佳模型
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    torch.save(model.state_dict(), "best_model.pt")
                    console.print(f"   [bold green]💾 New Best Model Saved! (Loss: {best_loss:.6f})[/]")

            # 更新总进度条
            progress.update(epoch_task, advance=1, info=f"Best: {best_loss:.5f}")

    console.rule("[bold green]Training Completed[/bold green]")
    console.print(f"🏆 Best Validation Loss: [bold green]{best_loss:.6f}[/]")

if __name__ == "__main__":
    main()