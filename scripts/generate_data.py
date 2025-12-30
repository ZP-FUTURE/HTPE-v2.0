import sys
import os
# Hack to find local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import itertools
from tqdm import tqdm

def pendulum_derivs(state, t, k1, k2, m, g, L):
    theta, omega = state
    damping = (k1 / m) * omega + (k2 / m) * omega * np.abs(omega)
    return [omega, -(g / L) * np.sin(theta) - damping]

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    phy = cfg.physics
    gen = cfg.generation
    
    os.makedirs(gen.save_dir, exist_ok=True)
    
    # 参数网格
    k1s = np.round(np.arange(gen.k1_range[0], gen.k1_range[1] + 1e-9, gen.k1_step), 4)
    k2s = np.round(np.arange(gen.k2_range[0], gen.k2_range[1] + 1e-9, gen.k2_step), 4)
    ns  = np.round(np.arange(gen.noise_range[0], gen.noise_range[1] + 1e-9, gen.noise_step), 3)
    
    t_values = np.arange(0.0, phy.t_max, phy.dt)
    init_state = [np.radians(phy.theta0_deg), 0.0]
    
    combinations = list(itertools.product(k1s, k2s))
    total_files = len(combinations) * len(ns)
    
    print(f"🚀 Generating {total_files} files in {gen.save_dir}...")
    print(f"📏 Columns: [Time, Angle_rad]")
    
    for k1, k2 in tqdm(combinations):
        sol = odeint(pendulum_derivs, init_state, t_values, args=(k1, k2, phy.m, phy.g, phy.L))
        clean_theta = sol[:, 0]
        
        for n in ns:
            theta = clean_theta + np.random.normal(0, n, size=clean_theta.shape) if n > 0 else clean_theta
            df = pd.DataFrame({
                'Time': t_values.astype(np.float32),
                'Angle_rad': theta.astype(np.float32)
            })
            
            fname = f"K1={k1:.4f}_K2={k2:.4f}_N={n:.3f}.csv"
            df.to_csv(os.path.join(gen.save_dir, fname), index=False)

if __name__ == "__main__":
    main()