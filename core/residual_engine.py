"""
Analytical Residual Engine: Evaluates nominal dynamics in real-time,
computes instantaneous torque residuals Delta tau, and extracts diagnostic signatures.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from config import CONFIG
from core.dynamics import FrankaDynamicsEngine

class AnalyticalResidualEngine:
    def __init__(self, dynamics: FrankaDynamicsEngine = None, threshold_nm: float = CONFIG.residual_threshold_nm):
        self.dynamics = dynamics if dynamics is not None else FrankaDynamicsEngine()
        self.threshold_nm = threshold_nm
        self.history_window_size = CONFIG.window_size

    def compute_instantaneous_residual(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau_measured: np.ndarray,
        nominal_model_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Delta tau = tau_measured - [M_hat(q)ddq + C_hat(q, dq)dq + g_hat(q) + F_hat]
        Returns:
            residual_vector: (7,)
            residual_norm: L2 norm
        """
        payload = nominal_model_params.get("payload_mass", 0.0) if nominal_model_params else 0.0
        viscous = nominal_model_params.get("viscous_friction", None) if nominal_model_params else None
        coulomb = nominal_model_params.get("coulomb_friction", None) if nominal_model_params else None

        tau_nominal = self.dynamics.inverse_dynamics(
            q=q,
            dq=dq,
            ddq=ddq,
            payload_mass=payload,
            viscous_coeffs=viscous,
            coulomb_coeffs=coulomb
        )
        delta_tau = tau_measured - tau_nominal
        norm = float(np.linalg.norm(delta_tau))
        return delta_tau, norm

    def extract_anomaly_features(
        self,
        window_residuals: np.ndarray,
        window_dq: np.ndarray,
        window_ddq: np.ndarray,
        window_q: np.ndarray
    ) -> Dict[str, Any]:
        """
        Performs physical signal attribution:
        - Velocity correlation reveals Viscous Friction drift.
        - Position (sinusoidal) correlation reveals Payload Mass gravity offsets.
        - Acceleration correlation reveals Inertia Matrix mismatches.
        """
        # window_residuals: shape (W, 7)
        rms_per_joint = np.sqrt(np.mean(window_residuals ** 2, axis=0))
        peak_per_joint = np.max(np.abs(window_residuals), axis=0)

        velocity_correlations = np.zeros(7)
        gravity_correlations = np.zeros(7)

        for j in range(7):
            std_tau = np.std(window_residuals[:, j])
            std_dq = np.std(window_dq[:, j])
            if std_tau > 1e-4 and std_dq > 1e-4:
                velocity_correlations[j] = np.corrcoef(window_residuals[:, j], window_dq[:, j])[0, 1]

            sin_q = np.sin(window_q[:, j])
            std_sin = np.std(sin_q)
            if std_tau > 1e-4 and std_sin > 1e-4:
                gravity_correlations[j] = np.corrcoef(window_residuals[:, j], sin_q)[0, 1]

        max_fault_joint = int(np.argmax(rms_per_joint))

        return {
            "rms_per_joint": rms_per_joint.tolist(),
            "peak_per_joint": peak_per_joint.tolist(),
            "velocity_correlations": velocity_correlations.tolist(),
            "gravity_correlations": gravity_correlations.tolist(),
            "primary_anomalous_joint": max_fault_joint + 1,  # 1-indexed for engineering readability
            "max_rms_nm": float(np.max(rms_per_joint)),
            "is_anomaly_detected": float(np.max(rms_per_joint)) > self.threshold_nm
        }
