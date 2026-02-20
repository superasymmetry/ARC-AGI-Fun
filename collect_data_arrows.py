import arc_agi
from arcengine import GameAction
import torch
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
    with open("collected_data_arrows.json", "r+") as f:
        data = json.load(f)
        for fr, ac, co in zip(all_frames, all_actions, all_coords):
            data["data"].append({"frame": fr.tolist(), "actions": ac.tolist(), "coords": co.tolist()})
        f.seek(0); json.dump(data, f, indent=4); f.truncate()
    print(f"Saved {len(all_frames)} trajectories to collected_data_arrows.json")

def collect_dataset(env_name, num_trajectories=1000, T=8):
    all_frames, all_actions, all_coords = [], [], []

    try:
        for _ in range(num_trajectories):
            env = arc.make(env_name)
            env.reset()

            obs = env.observation_space
            frames, actions, coords = [obs.state], [], []

            for _ in range(T):
                action = random.choice(ARROW_ACTIONS)
                env.step(action)

                obs = env.observation_space
                actions.append(ARROW_ACTIONS.index(action))  # 0-3
                coords.append([0.0, 0.0])
                frames.append(obs.state)

                if not obs:
                    break

            print(obs.frame)
            print(frames)
            if len(actions) == T:
                all_frames.append(torch.tensor(obs.frame))
                all_actions.append(torch.tensor(actions))
                all_coords.append(torch.tensor(coords).float())
    except KeyboardInterrupt:
        print(f"\nInterrupted! Saving {len(all_frames)} collected trajectories...")

    if all_frames:
        save_to_json(all_frames, all_actions, all_coords)

    if all_frames:
        dataset = {
            "frames":     torch.stack(all_frames),
            "action_ids": torch.stack(all_actions),
            "coords":     torch.stack(all_coords),
        }
        torch.save(dataset, f"{env_name}.pt")
        return dataset

collect_dataset("ls20-cb3b57cc", num_trajectories=1000, T=8)