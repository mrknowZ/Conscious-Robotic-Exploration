from sb3_contrib import MaskablePPO

from envs.toy_grid_env import ToyGridEnv


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 7

MODEL_PATH = "trained_ppo_agent_7x7_v5"


# ============================================================
# ENVIRONMENT
# ============================================================

env = ToyGridEnv(
    grid_size=GRID_SIZE
)

MAX_TEST_STEPS = env.max_steps


# ============================================================
# LOAD MODEL
# ============================================================

model = MaskablePPO.load(
    MODEL_PATH
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)

print(
    "TESTING PPO AGENT ON 7x7 GRID"
)

print("=" * 75)

print(
    f"Grid Size       : "
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    f"Total Cells     : "
    f"{env.total_cells}"
)

print(
    f"Maximum Steps   : "
    f"{MAX_TEST_STEPS}"
)


# ============================================================
# RESET
# ============================================================

obs, _ = env.reset()

total_reward = 0.0

total_energy = 0.0

step_count = 0

wall_collisions = 0

done = False

info = {}


# ============================================================
# ACTION NAMES
# ============================================================

action_names = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}


# ============================================================
# TEST LOOP
# ============================================================

while not done and step_count < MAX_TEST_STEPS:

    # --------------------------------------------------------
    # Action mask (agent can only pick legal moves)
    # --------------------------------------------------------

    action_masks = env.action_masks()

    # --------------------------------------------------------
    # Predict action
    # --------------------------------------------------------

    action, _states = model.predict(
        obs,
        action_masks=action_masks,
        deterministic=True
    )

    # --------------------------------------------------------
    # Environment step
    # --------------------------------------------------------

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    step_count += 1

    total_reward += reward

    total_energy += info["energy_spent"]

    if info["wall_collision"]:

        wall_collisions += 1

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    pos_x = info["position"][0]

    pos_y = info["position"][1]

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage = (
        info["coverage"] * 100.0
    )

    # --------------------------------------------------------
    # Action name
    # --------------------------------------------------------

    action_name = action_names[
        int(action)
    ]

    # --------------------------------------------------------
    # Print step
    # --------------------------------------------------------

    print(
        f"Step {step_count:03d} | "
        f"Action: {action_name:5s} | "
        f"Pos: [{pos_x},{pos_y}] | "
        f"Coverage: {coverage:6.2f}% | "
        f"Reward: {reward:7.2f} | "
        f"Wall: {info['wall_collision']}"
    )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    done = (
        terminated
        or truncated
    )


# ============================================================
# FINAL RESULTS
# ============================================================

visited_cells = info["visited_cells"]

final_coverage = (
    visited_cells
    / env.total_cells
) * 100.0


print("\n" + "=" * 75)

print(
    "FINAL EVALUATION RESULTS"
)

print("=" * 75)


print(
    f"Grid Size              : "
    f"{GRID_SIZE} x {GRID_SIZE}"
)

print(
    f"Total Cells             : "
    f"{env.total_cells}"
)

print(
    f"Visited Cells           : "
    f"{visited_cells} / {env.total_cells}"
)

print(
    f"Exploration Coverage    : "
    f"{final_coverage:.2f}%"
)

print(
    f"Total Steps             : "
    f"{step_count}"
)

print(
    f"Total Energy Consumed   : "
    f"{total_energy:.2f} J"
)

print(
    f"Wall Collisions         : "
    f"{wall_collisions}"
)

print(
    f"Total Cumulative Reward : "
    f"{total_reward:.2f}"
)


# ============================================================
# STATUS
# ============================================================

if final_coverage >= 100.0:

    print(
        "\nSTATUS: FULL EXPLORATION SUCCESS"
    )

elif final_coverage >= 90.0:

    print(
        "\nSTATUS: VERY GOOD EXPLORATION"
    )

elif final_coverage >= 75.0:

    print(
        "\nSTATUS: PARTIAL EXPLORATION"
    )

else:

    print(
        "\nSTATUS: TRAINING NEEDS IMPROVEMENT"
    )


print("=" * 75)