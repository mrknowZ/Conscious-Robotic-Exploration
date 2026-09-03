# Exact import according to your file: energy_model/ina219_energy_model.py
from energy_model.ina219_energy_model import INA219EnergyModel

# Energy Model Object Create Karein
energy_sensor = INA219EnergyModel(voltage_nom=12.0)

print("=" * 45)
print("🔋 INA219 ENERGY MODEL SENSOR TEST 🔋")
print("=" * 45)

# Test 1: Robot Idle (Khara hai)
e_idle = energy_sensor.get_energy_consumption(action_type="idle", step_duration_sec=1.0)
print(f"1. Robot Idle (1 sec)       -> Energy Spent: {e_idle:.2f} Joules")

# Test 2: Robot Move Flat Terrain
e_flat = energy_sensor.get_energy_consumption(action_type="flat", step_duration_sec=1.0)
print(f"2. Robot Flat Move (1 sec)  -> Energy Spent: {e_flat:.2f} Joules")

# Test 3: Robot Move Rough Terrain
e_rough = energy_sensor.get_energy_consumption(action_type="rough", step_duration_sec=1.0)
print(f"3. Robot Rough Move (1 sec) -> Energy Spent: {e_rough:.2f} Joules")

print("=" * 45)