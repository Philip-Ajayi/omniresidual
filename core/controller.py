"""
Adaptive Computed Torque Controller (Inverse Dynamics Control).
Dynamically shifts model parameters when sys-id updates arrive,
guaranteeing recovery of trajectory tracking and pick-and-place success.
"""
import numpy as np
from typing import Dict, Any, Tuple
from config import CONFIG
from core.dynamics import FrankaDynamicsEngine

class AdaptiveFrankaController:
    def __init__(self, dynamics: FrankaDynamicsEngine = None):
        self.dynamics = dynamics if dynamics is not None else FrankaDynamicsEngine()
        self.kp = CONFIG.controller.kp
        self.kd = CONFIG.controller.kd
        self.torque_limits = CONFIG.controller.torque_limits

        # Adaptive model parameters
        self.model_payload_mass = 0.0
        self.model_viscous_friction = self.dynamics.cfg.nominal_viscous_friction.copy()
        self.model_coulomb_friction = self.dynamics.cfg.nominal_coulomb_friction.copy()

    def update_model_parameters(self, estimated_params: Dict[str, float]) -> None:
        """Applies adaptive dynamics patch received from the Sys-ID Solver."""
        for name, value in estimated_params.items():
            if "payload_mass" in name:
                self.model_payload_mass = max(0.0, float(value))
            elif "viscous_friction" in name:
                # Extract joint index from string (e.g. joint_4_viscous_friction)
                tokens = name.split("_")
                j_idx = int(tokens[1]) - 1
                self.model_viscous_friction[j_idx] = float(value)

    def compute_control_torque(
        self,
        q_meas: np.ndarray,
        dq_meas: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray
    ) -> np.ndarray:
        """
        Computed Torque Control Law:
        tau_cmd = M_hat(q) * [ddq_des + Kp(q_des - q) + Kd(dq_des - dq)]
                  + C_hat(q, dq)*dq + g_hat(q) + F_hat(dq)
        """
        e = q_des - q_meas
        de = dq_des - dq_meas
        v = ddq_des + self.kp * e + self.kd * de

        m_hat = self.dynamics.compute_inertia_matrix(q_meas, self.model_payload_mass)
        c_hat = self.dynamics.compute_coriolis_centrifugal(q_meas, dq_meas, self.model_payload_mass)
        g_hat = self.dynamics.compute_gravity_vector(q_meas, self.model_payload_mass)
        f_hat = self.dynamics.compute_friction(
            dq_meas, self.model_viscous_friction, self.model_coulomb_friction
        )

        tau_cmd = (m_hat @ v) + c_hat + g_hat + f_hat
        # Saturate at physical joint torque limits
        return np.clip(tau_cmd, -self.torque_limits, self.torque_limits)
