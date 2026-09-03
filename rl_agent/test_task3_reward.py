from envs.toy_grid_env import ToyGridEnv

# Environment Create Karein
env = ToyGridEnv(grid_size=5)
state, _ = env.reset()

print("=" * 55)
print("🎯 TASK 3: ENERGY-AWARE REWARD SHAPING TEST 🎯")
print("=" * 55)
print(f"Initial Start Position: {state}\n")

# Step 1: Robot Moves Right to [1, 0] (Unexplored Cell)
state, reward, term, trunc, info = env.step(3)
print(f"Step 1 (Right) -> Pos: {state} | Reward: {reward:.2f} | Energy Spent: {info['energy_spent']} J")

# Step 2: Robot Moves Left back to [0, 0] (Visited Cell)
state, reward, term, trunc, info = env.step(2)
print(f"Step 2 (Left)  -> Pos: {state} | Reward: {reward:.2f} | Energy Spent: {info['energy_spent']} J")

print("=" * 55)