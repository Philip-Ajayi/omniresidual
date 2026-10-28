"""
Fault Injection Module: Simulates hardware degradation, gear friction wear,
payload mass variations, and actuator stiction during live execution.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PhysicalDriftFault:
    name: str
    active: bool = False
    activation_time: float = 3.5  # Seconds after trajectory onset
    payload_mass_delta: float = 1.25  # kg
    joint_friction_multipliers: np.ndarray = None

    def __post_init__(self):
        if self.joint_friction_multipliers is None:
            # Default: Joint 4 transmission gear wear (+150% friction)
            self.joint_friction_multipliers = np.array([1.0, 1.0, 1.0, 2.5, 1.0, 1.0, 1.0])

class FaultInjectionEngine:
    def __init__(self, fault: PhysicalDriftFault = None):
        self.fault = fault if fault is not None else PhysicalDriftFault(name="J4_Wear_and_Payload_Surge")

    def get_actual_physical_parameters(
        self,
        current_time: float,
        nominal_viscous: np.ndarray,
        nominal_coulomb: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates physical parameters at current time t.
        If fault is active and t >= activation_time, applies physical drift.
        """
        if self.fault.active and current_time >= self.fault.activation_time:
            actual_viscous = nominal_viscous * self.fault.joint_friction_multipliers
            actual_coulomb = nominal_coulomb * np.sqrt(self.fault.joint_friction_multipliers)
            actual_payload = self.fault.payload_mass_delta
            drift_engaged = True
        else:
            actual_viscous = nominal_viscous.copy()
            actual_coulomb = nominal_coulomb.copy()
            actual_payload = 0.0
            drift_engaged = False

        return {
            "viscous_friction": actual_viscous,
            "coulomb_friction": actual_coulomb,
            "payload_mass": actual_payload,
            "drift_engaged": drift_engaged
        }
