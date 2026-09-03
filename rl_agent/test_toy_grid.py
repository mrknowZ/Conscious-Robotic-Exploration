from envs.toy_grid_env import ToyGridEnv

# 1. Custom Toy Grid Environment load karein (5x5 Grid)
env = ToyGridEnv(grid_size=5)

# 2. Environment reset karke initial state check karein
obs, info = env.reset()

print("=" * 30)
print("🤖 TOY GRID ENVIRONMENT TEST 🤖")
print("=" * 30)
print(f"Initial Robot Position [x, y]: {obs}")
print("\nInitial Grid State:")
env.render()

# 3. Random 3 Steps lekar step() function test karein
for i in range(3):
    action = env.action_space.sample()  # Random action: 0:Up, 1:Down, 2:Left, 3:Right
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n---> Step {i+1} <---")
    print(f"Action Taken: {action} (0:Up, 1:Down, 2:Left, 3:Right)")
    print(f"New Position [x, y]: {obs}")
    print(f"Reward Received: {reward}")
    print(f"Total Visited Cells: {info['visited_count']}")
    
    # Grid state visual output
    env.render()