import arc_agi
from arcengine import GameAction
import torch
import numpy as np
import random
import os
import json
import dotenv

dotenv.load_dotenv()
arc = arc_agi.Arcade(arc_api_key=os.getenv("ARC_API_KEY"))

ARROW_ACTIONS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]

def save_to_json(all_frames, all_actions, all_coords):
    # Stack frames properly - each trajectory should be [T+1, H, W]
    frames_list = [np.stack(traj).tolist() for traj in all_frames]
    
    data = {
        "frames": frames_list,
        "actions": all_actions,  # Already lists
        "coords": all_coords      # Already lists
    }
    
    with open("collected_data_arrows.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(all_frames)} trajectories to collected_data_arrows.json")

def collect_dataset(env_name, num_trajectories=1000, T=8):
    all_frames, all_actions, all_coords = [], [], []

    try:
        for _ in range(num_trajectories):
            env = arc.make(env_name)
            env.reset()
            obs = env.observation_space
            frames, actions, coords = [obs.frame], [], []

            for _ in range(T):
                action = random.choice(ARROW_ACTIONS)
                env.step(action)
                obs = env.observation_space
                actions.append(ARROW_ACTIONS.index(action))
                coords.append([0.0, 0.0])
                frames.append(obs.frame)

            if len(actions) == T:
                all_frames.append(frames)  # Keep as list of numpy arrays
                all_actions.append(actions)
                all_coords.append(coords)
                
    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving {len(all_frames)} collected trajectories...")

    if all_frames:
        save_to_json(all_frames, all_actions, all_coords)

collect_dataset("ls20-cb3b57cc", num_trajectories=1000, T=8)