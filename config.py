"""
OmniResidual Configuration Module
Centralizes mechanical specifications, control gains, and NVIDIA NIM / Microservice configs.
"""
import os
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PandaKinematicsConfig:
    """Modified Denavit-Hartenberg (MDH) and dynamic parameters for Franka Emika Panda."""
    d: np.ndarray = field(default_factory=lambda: np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107]))
    a: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]))
    link_masses: np.ndarray = field(default_factory=lambda: np.array([4.97, 0.65, 3.23, 3.59, 1.23, 1.67, 0.73]))
    nominal_viscous_friction: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.3, 0.3, 0.2, 0.15, 0.1]))
    nominal_coulomb_friction: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.4, 0.4, 0.3, 0.25, 0.2]))
    joint_lower_limits: np.ndarray = field(default_factory=lambda: np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]))
    joint_upper_limits: np.ndarray = field(default_factory=lambda: np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))

@dataclass
class ControllerConfig:
    kp: np.ndarray = field(default_factory=lambda: np.array([120.0, 120.0, 100.0, 100.0, 60.0, 40.0, 20.0]))
    kd: np.ndarray = field(default_factory=lambda: np.array([22.0, 22.0, 18.0, 18.0, 12.0, 8.0, 4.0]))
    torque_limits: np.ndarray = field(default_factory=lambda: np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]))
    grasp_tolerance_meters: float = 0.025  # 2.5 cm threshold for successful pick-and-place

@dataclass
class SystemConfig:
    freq_hz: int = int(os.getenv("TELEMETRY_FREQ_HZ", "50"))
    dt: float = 1.0 / freq_hz
    residual_threshold_nm: float = float(os.getenv("ANOMALY_THRESHOLD_NM", "3.5"))
    window_size: int = 50

    # NVIDIA NIM Microservice Configuration
    # Cloud NIM: https://integrate.api.nvidia.com/v1
    # Local container: http://localhost:8000/v1
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", os.getenv("NVIDIA_NIM_API_KEY", os.getenv("NEBIUS_API_KEY", "")))
    nvidia_nim_base_url: str = os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nvidia_nim_model: str = os.getenv("NVIDIA_NIM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

    # Backward compatibility aliases for existing Nebius calls & tests
    nebius_api_key: str = os.getenv("NEBIUS_API_KEY", os.getenv("NVIDIA_API_KEY", os.getenv("NVIDIA_NIM_API_KEY", "")))
    nebius_base_url: str = os.getenv("NEBIUS_BASE_URL", "https://integrate.api.nvidia.com/v1")
    nebius_model: str = os.getenv("NEBIUS_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

    parallel_envs: int = int(os.getenv("SYSID_PARALLEL_ENVS", "16"))
    max_sysid_iterations: int = int(os.getenv("SYSID_MAX_ITERATIONS", "25"))
    kinematics: PandaKinematicsConfig = field(default_factory=PandaKinematicsConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

CONFIG = SystemConfig()