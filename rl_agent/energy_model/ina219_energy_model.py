class INA219EnergyModel:
    """
    Simulated INA219-based energy model.

    Power:
        P = V * I

    Energy:
        E = P * dt

    Energy is returned in Joules.
    """

    def __init__(self, voltage_nom=12.0):

        # Nominal battery voltage
        self.voltage = voltage_nom

        # Current profiles (Amperes)
        self.I_idle = 0.2
        self.I_move_flat = 1.0
        self.I_move_rough = 2.5

    def get_energy_consumption(
        self,
        action_type="flat",
        step_duration_sec=1.0
    ):
        """
        Calculate energy consumed during one action.

        Parameters:
            action_type:
                "idle", "flat", or "rough"

            step_duration_sec:
                Duration of the movement in seconds.

        Returns:
            Energy consumption in Joules.
        """

        if action_type == "idle":
            current = self.I_idle

        elif action_type == "rough":
            current = self.I_move_rough

        elif action_type == "flat":
            current = self.I_move_flat

        else:
            raise ValueError(
                f"Unknown action_type: {action_type}"
            )

        # Power = Voltage × Current
        power_watts = self.voltage * current

        # Energy = Power × Time
        energy_joules = (
            power_watts * step_duration_sec
        )

        return energy_joules