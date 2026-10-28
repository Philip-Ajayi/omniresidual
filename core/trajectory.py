"""
50 Hz Trajectory Generator producing smooth minimum-jerk industrial pick-and-place profiles.
"""
import numpy as np
from typing import Tuple, List
from config import CONFIG

class FrankaTrajectoryGenerator:
    def __init__(self, dt: float = CONFIG.dt):
        self.dt = dt
        # Key reference waypoints in configuration space (radians)
        self.waypoints = [
            np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),       # 1. Home / Ready
            np.array([0.35, -0.45, 0.20, -1.95, 0.15, 1.65, 0.65]),       # 2. Pre-Grasp Approach
            np.array([0.40, -0.22, 0.25, -1.75, 0.18, 1.70, 0.60]),       # 3. Grasp Contact
            np.array([0.25, -0.65, 0.10, -2.15, 0.05, 1.55, 0.70]),       # 4. Payload Lift
            np.array([-0.45, -0.35, -0.30, -1.85, -0.25, 1.60, 0.85]),    # 5. Place Zone Approach
            np.array([-0.50, -0.18, -0.35, -1.70, -0.30, 1.65, 0.80]),    # 6. Place Target
            np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])        # 7. Return Home
        ]
        self.segment_durations = [2.0, 1.5, 1.5, 2.0, 1.5, 2.0]

    def _min_jerk_interpolate(self, q0: np.ndarray, q1: np.ndarray, duration: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates quintic polynomial (minimum jerk) position, velocity, and acceleration profiles."""
        steps = int(np.round(duration / self.dt))
        t = np.linspace(0.0, duration, steps)
        tau = t / duration

        # Quintic polynomial blending coefficients: s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
        s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
        ds = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / duration
        dds = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (duration ** 2)

        q = np.outer(1.0 - s, q0) + np.outer(s, q1)
        dq = np.outer(ds, q1 - q0)
        ddq = np.outer(dds, q1 - q0)
        return q, dq, ddq

    def generate_full_cycle(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds a full continuous pick-and-place operational cycle.
        Returns:
            times: (N,)
            q: (N, 7)
            dq: (N, 7)
            ddq: (N, 7)
        """
        all_q, all_dq, all_ddq = [], [], []

        for i in range(len(self.waypoints) - 1):
            q_seg, dq_seg, ddq_seg = self._min_jerk_interpolate(
                self.waypoints[i], self.waypoints[i + 1], self.segment_durations[i]
            )
            all_q.append(q_seg)
            all_dq.append(dq_seg)
            all_ddq.append(ddq_seg)

        q = np.vstack(all_q)
        dq = np.vstack(all_dq)
        ddq = np.vstack(all_ddq)
        times = np.arange(len(q)) * self.dt
        return times, q, dq, ddq
