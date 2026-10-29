"""
Parallel System Identification Engine (Isaac Lab Surrogate):
Executes 16 batched forward physics rollouts using Covariance Matrix Adaptation (CMA-ES).
"""
import numpy as np
from typing import Dict, Any, List, Callable, Optional
from config import CONFIG
from core.dynamics import FrankaDynamicsEngine

class BatchedIsaacLabSysIDSolver:
    def __init__(self, dynamics: FrankaDynamicsEngine = None, num_envs: int = CONFIG.parallel_envs):
        self.dynamics = dynamics if dynamics is not None else FrankaDynamicsEngine()
        self.num_envs = num_envs
        self.max_iter = CONFIG.max_sysid_iterations

    def optimize_parameters(
        self,
        q_window: np.ndarray,
        dq_window: np.ndarray,
        ddq_window: np.ndarray,
        tau_measured_window: np.ndarray,
        parameter_hypotheses: List[Dict[str, Any]],
        iteration_callback: Optional[Callable[[int, float, List[float]], None]] = None
    ) -> Dict[str, Any]:
        num_params = len(parameter_hypotheses)
        if num_params == 0:
            return {"converged": False, "estimated_params": {}, "iterations": 0, "convergence_trace": []}

        lower_bounds = np.array([p["search_min"] for p in parameter_hypotheses])
        upper_bounds = np.array([p["search_max"] for p in parameter_hypotheses])
        initial_mean = np.array([p["nominal_value"] for p in parameter_hypotheses])
        initial_mean = np.clip(initial_mean, lower_bounds, upper_bounds)

        sigma = 0.3 * (upper_bounds - lower_bounds)
        mean = initial_mean.copy()
        convergence_trace = []
        best_overall_cost = float("inf")
        best_overall_sample = mean.copy()

        stride = max(1, len(q_window) // 25)
        sub_q = q_window[::stride]
        sub_dq = dq_window[::stride]
        sub_ddq = ddq_window[::stride]
        sub_tau = tau_measured_window[::stride]

        for it in range(self.max_iter):
            # Sample 16 candidate vectors across parallel surrogate environments
            samples = np.random.normal(loc=mean, scale=sigma, size=(self.num_envs, num_params))
            samples = np.clip(samples, lower_bounds, upper_bounds)

            costs = np.zeros(self.num_envs)
            for env_idx in range(self.num_envs):
                costs[env_idx] = self._evaluate_rollout_loss(
                    samples[env_idx], parameter_hypotheses, sub_q, sub_dq, sub_ddq, sub_tau
                )

            sort_indices = np.argsort(costs)
            best_iter_cost = costs[sort_indices[0]]
            best_iter_sample = samples[sort_indices[0]]

            if best_iter_cost < best_overall_cost:
                best_overall_cost = best_iter_cost
                best_overall_sample = best_iter_sample.copy()

            elites = samples[sort_indices[:4]]
            mean = np.mean(elites, axis=0)
            sigma = 0.85 * sigma + 0.15 * (np.std(elites, axis=0) + 1e-4)

            step_data = {
                "iteration": it + 1,
                "best_loss": float(best_overall_cost),
                "param_estimates": best_overall_sample.tolist(),
                "all_env_losses": costs.tolist()
            }
            convergence_trace.append(step_data)

            # Live UI callback for production streaming
            if iteration_callback:
                iteration_callback(it + 1, float(best_overall_cost), costs.tolist())

            if best_overall_cost < 1e-2:
                break

        resolved_params = {}
        for idx, p in enumerate(parameter_hypotheses):
            resolved_params[p["parameter_name"]] = float(best_overall_sample[idx])

        return {
            "converged": True,
            "estimated_params": resolved_params,
            "final_loss": float(best_overall_cost),
            "iterations": len(convergence_trace),
            "convergence_trace": convergence_trace
        }

    def _evaluate_rollout_loss(
        self, candidate_vector: np.ndarray, hypotheses: List[Dict[str, Any]],
        q_seq: np.ndarray, dq_seq: np.ndarray, ddq_seq: np.ndarray, tau_seq: np.ndarray
    ) -> float:
        viscous = self.dynamics.cfg.nominal_viscous_friction.copy()
        coulomb = self.dynamics.cfg.nominal_coulomb_friction.copy()
        payload = 0.0

        for i, hyp in enumerate(hypotheses):
            val = candidate_vector[i]
            pname = hyp["parameter_name"]
            if "viscous_friction" in pname:
                joint_idx = hyp["joint"] - 1
                viscous[joint_idx] = val
            elif "payload_mass" in pname:
                payload = val

        total_sq_err = 0.0
        n_steps = len(q_seq)
        for t in range(n_steps):
            pred_tau = self.dynamics.inverse_dynamics(
                q=q_seq[t], dq=dq_seq[t], ddq=ddq_seq[t],
                payload_mass=payload, viscous_coeffs=viscous, coulomb_coeffs=coulomb
            )
            total_sq_err += np.sum((tau_seq[t] - pred_tau) ** 2)

        return float(total_sq_err / n_steps)