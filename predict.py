import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
import numpy as np
import os
import glob
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

from utils.functions import load_model_class
from utils.common import count_parameters

console = Console()

class RealDataProcessor:
    def __init__(self, target_len=72000, to_radians=False):
        self.target_len = target_len
        self.to_radians = to_radians 

    def process(self, file_path):
        try:
            df = pd.read_csv(file_path)
            
            possible_cols = ['Angle', 'angle', 'Theta', 'theta', 'Angle_rad', 'Angle_deg', 'Value']
            col_name = next((c for c in possible_cols if c in df.columns), None)
            
            if col_name is None:
                data = df.iloc[:, 0].values
            else:
                data = df[col_name].values

            data = data.astype(np.float32)

            if self.to_radians or (np.max(np.abs(data)) > 7.0):
                data = np.radians(data)

            current_len = len(data)
            if current_len > self.target_len:
                data = data[:self.target_len]
            elif current_len < self.target_len:
                padding = np.zeros(self.target_len - current_len, dtype=np.float32)
                data = np.concatenate([data, padding])

            tensor = torch.tensor(data).unsqueeze(0).unsqueeze(-1)
            return tensor, col_name, current_len

        except Exception as e:
            console.print(f"[bold red]Error processing {file_path}: {e}[/]")
            return None, None, 0

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_model.pt" #模型路径
    
    real_data_dir = "real_experiments" #数据路径
    
    console.print(Panel.fit(
        "[bold cyan]Real-World Inference Engine[/bold cyan]\n"
        f"Model: {cfg.model.identifier}\n"
        f"Weights: {model_path}",
        border_style="blue"
    ))

    if not os.path.exists(model_path):
        console.print(f"[bold red]❌ Model weights not found at {model_path}![/]")
        console.print("Please train the model first or provide the correct path.")
        return

    ModelClass = load_model_class(cfg.model.identifier)
    model = ModelClass(cfg.model)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        console.print("[bold green]✅ Model weights loaded successfully.[/]")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load weights: {e}[/]")
        return

    model.to(device)
    model.eval()
    if not os.path.exists(real_data_dir):
        os.makedirs(real_data_dir)
        console.print(f"[yellow]⚠️ Directory '{real_data_dir}' created. Please put your CSV files there.[/]")
        return

    csv_files = glob.glob(os.path.join(real_data_dir, "*.csv"))
    if not csv_files:
        console.print(f"[yellow]⚠️ No CSV files found in '{real_data_dir}'.[/]")
        return

    processor = RealDataProcessor(target_len=72000, to_radians=False) # 如果你的数据是度数，把这里改为 True
    
    results_table = Table(title="🧪 Experimental Results Inference", show_lines=True)
    results_table.add_column("File Name", style="cyan", no_wrap=True)
    results_table.add_column("Length", style="dim")
    results_table.add_column("K1 (Friction)", style="bold magenta")
    results_table.add_column("K2 (Air Drag)", style="bold green")
    results_table.add_column("Noise (N)", style="blue")

    console.print(f"🔍 Found {len(csv_files)} experimental files. Analyzing...")

    with torch.no_grad():
        for file_path in track(csv_files, description="Inferring..."):
            file_name = os.path.basename(file_path)
            
            input_tensor, col_name, original_len = processor.process(file_path)
            
            if input_tensor is not None:
                input_tensor = input_tensor.to(device)
                

                pred = model(input_tensor)
                
                k1_pred = pred[0, 0].item()
                k2_pred = pred[0, 1].item()
                n_pred  = pred[0, 2].item()

                results_table.add_row(
                    file_name,
                    f"{original_len}",
                    f"{k1_pred:.5f}",
                    f"{k2_pred:.5f}",
                    f"{n_pred:.4f}"
                )

    console.print("\n")
    console.print(results_table)
    console.print("[dim italic]* K2 is the coefficient of quadratic air drag.[/dim]")

if __name__ == "__main__":
    main()