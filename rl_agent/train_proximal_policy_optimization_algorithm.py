"""
Train an energy-aware exploration agent on the 7x7 grid using
MaskablePPO (sb3-contrib). This is the exact v5 configuration that
achieved 100% coverage (49/49 cells), 0 wall collisions, in 77 steps.

Install requirements first:
    pip install stable-baselines3 sb3-contrib gymnasium --break-system-packages
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from envs.toy_grid_env import ToyGridEnv


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 7

TOTAL_TIMESTEPS = 2_000_000

MODEL_PATH = "trained_ppo_agent_7x7_v5"


# ============================================================
# MASK FUNCTION
# ============================================================

def mask_fn(env):
    return env.action_masks()


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("TRAINING PPO AGENT - 7x7 ENERGY-AWARE GRID WORLD")
print("=" * 75)


# ============================================================
# ENVIRONMENT
# ============================================================

base_env = ToyGridEnv(
    grid_size=GRID_SIZE
)

print("\n[INFO] Checking Gymnasium environment...")

check_env(
    base_env,
    warn=True
)

print(
    "[INFO] Environment check completed successfully."
)

env = ActionMasker(base_env, mask_fn)

print("\n[INFO] Environment created.")

print(
    f"[INFO] Grid Size       : "
    f"{base_env.grid_size} x {base_env.grid_size}"
)

print(
    f"[INFO] Total Cells     : "
    f"{base_env.total_cells}"
)

print(
    f"[INFO] Max Steps       : "
    f"{base_env.max_steps}"
)

print(
    f"[INFO] Observation Dim : "
    f"{base_env.observation_space.shape[0]}"
)

print(
    f"[INFO] Action Space    : "
    f"{base_env.action_space}"
)


# ============================================================
# PPO MODEL (Maskable)
# ============================================================

model = MaskablePPO(
    policy="MlpPolicy",

    env=env,

    # Learning rate
    learning_rate=0.0003,

    # Rollout size
    n_steps=2048,

    # Mini-batch
    batch_size=256,

    # Discount factor
    gamma=0.995,

    # GAE
    gae_lambda=0.95,

    # PPO clipping
    clip_range=0.2,

    # Exploration
    ent_coef=0.025,

    # Value function coefficient
    vf_coef=0.5,

    # Gradient clipping
    max_grad_norm=0.5,

    # Neural network
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        )
    ),

    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 75)

print(
    f"TRAINING STARTED FOR "
    f"{TOTAL_TIMESTEPS:,} TIMESTEPS"
)

print("=" * 75)


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# QUICK SANITY EVALUATION
# ============================================================

mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=20, deterministic=True
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)

print(
    "TRAINING COMPLETE"
)

print("=" * 75)

print(
    f"Model saved as:"
)

print(
    f"    {MODEL_PATH}.zip"
)

print(
    f"Mean eval reward : {mean_reward:.2f} +/- {std_reward:.2f}"
)

print("=" * 75)