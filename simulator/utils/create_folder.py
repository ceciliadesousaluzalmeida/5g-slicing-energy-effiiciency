import os
from datetime import datetime

def create_simulation_folder(base_dir="results"):
    """
    Create a new timestamped folder inside base_dir to store simulation outputs.
    Example: results/2025-10-13_21-45-02/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sim_dir = os.path.join(base_dir, timestamp)
    os.makedirs(sim_dir, exist_ok=True)
    print(f"[INFO] Simulation folder created: {sim_dir}")
    return sim_dir
