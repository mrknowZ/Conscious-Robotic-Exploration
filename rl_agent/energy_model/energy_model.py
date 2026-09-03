# rl_agent/energy_model.py

class EnergyModel:
    def __init__(self, c_dist=0.05, c_comp=0.01):
        # Coefficients: distance (motors) and computation (Jetson/GPU)
        self.c_dist = c_dist
        self.c_comp = c_comp

    def estimate_cost(self, distance_traversed, compute_time):
        """
        Calculates estimated energy consumption:
        E_total = E_motion + E_computation
        """
        e_motion = self.c_dist * distance_traversed
        e_compute = self.c_comp * compute_time
        
        return e_motion + e_compute