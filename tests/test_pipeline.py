"""
End-to-End Automated Test Suite for OmniResidual.
Verifies dynamics accuracy, residual detection, Nebius reasoning format,
and Sys-ID parameter convergence.
"""
import pytest
import numpy as np
from config import CONFIG
from core.dynamics import FrankaDynamicsEngine
from core.trajectory import FrankaTrajectoryGenerator
from core.residual_engine import AnalyticalResidualEngine
from core.nebius_client import NebiusReasoningEngine
from core.sysid_solver import BatchedIsaacLabSysIDSolver
from core.controller import AdaptiveFrankaController

@pytest.fixture
def dynamics():
    return FrankaDynamicsEngine()

@pytest.fixture
def trajectory():
    gen = FrankaTrajectoryGenerator()
    return gen.generate_full_cycle()

def test_forward_kinematics_dimensions(dynamics):
    q_zero = np.zeros(7)
    positions = dynamics.forward_kinematics_all_links(q_zero)
    assert len(positions) == 9  # Base + 7 joints + end-effector point
    for p in positions:
        assert p.shape == (3,)
        assert not np.isnan(p).any()

def test_positive_definite_inertia_matrix(dynamics):
    q_rand = np.random.uniform(-1.0, 1.0, 7)
    m = dynamics.compute_inertia_matrix(q_rand, payload_mass=1.5)
    assert m.shape == (7, 7)
    # Check symmetric
    assert np.allclose(m, m.T, atol=1e-5)
    # Check positive definite via eigenvalues
    eigenvalues = np.linalg.eigvals(m)
    assert np.all(eigenvalues > 0.0)

def test_residual_zero_on_nominal_model(dynamics, trajectory):
    times, q, dq, ddq = trajectory
    res_engine = AnalyticalResidualEngine(dynamics)

    # Compute exact nominal torque
    q_sample = q[10]
    dq_sample = dq[10]
    ddq_sample = ddq[10]
    tau_exact = dynamics.inverse_dynamics(q_sample, dq_sample, ddq_sample)

    delta_tau, norm = res_engine.compute_instantaneous_residual(
        q=q_sample, dq=dq_sample, ddq=ddq_sample, tau_measured=tau_exact
    )
    assert norm < 1e-4

def test_residual_breach_on_fault(dynamics, trajectory):
    times, q, dq, ddq = trajectory
    res_engine = AnalyticalResidualEngine(dynamics)

    q_s = q[20]
    dq_s = dq[20]
    ddq_s = ddq[20]

    # Ground truth with unmodeled +2.0 kg payload and doubled friction
    tau_degraded = dynamics.inverse_dynamics(
        q_s, dq_s, ddq_s,
        payload_mass=2.0,
        viscous_coeffs=dynamics.cfg.nominal_viscous_friction * 2.0
    )

    delta_tau, norm = res_engine.compute_instantaneous_residual(
        q=q_s, dq=dq_s, ddq_s, tau_measured=tau_degraded
    )
    assert norm > CONFIG.residual_threshold_nm

def test_nebius_reasoning_fallback_schema():
    client = NebiusReasoningEngine(api_key="")
    features = {
        "rms_per_joint": [0.2, 0.4, 0.3, 5.8, 0.2, 1.5, 1.4],
        "peak_per_joint": [0.4, 0.6, 0.5, 8.2, 0.4, 2.1, 2.0],
        "velocity_correlations": [0.05, 0.1, 0.08, 0.89, 0.02, 0.15, 0.1],
        "gravity_correlations": [0.0, 0.1, 0.2, 0.1, 0.05, 0.45, 0.5],
        "primary_anomalous_joint": 4,
        "max_rms_nm": 5.8,
        "is_anomaly_detected": True
    }
    hypothesis = client.infer_physical_hypothesis(features)
    assert "fault_mode" in hypothesis
    assert "parameter_hypotheses" in hypothesis
    assert len(hypothesis["parameter_hypotheses"]) >= 1
    # Check that Joint 4 viscous friction is isolated
    pnames = [p["parameter_name"] for p in hypothesis["parameter_hypotheses"]]
    assert any("joint_4_viscous_friction" in name for name in pnames)

def test_batched_sysid_convergence(dynamics, trajectory):
    times, q, dq, ddq = trajectory
    w_q = q[20:60]
    w_dq = dq[20:60]
    w_ddq = ddq[20:60]

    true_payload = 1.2
    # Generate measured torques with ground-truth payload
    w_tau = np.array([
        dynamics.inverse_dynamics(w_q[i], w_dq[i], w_ddq[i], payload_mass=true_payload)
        for i in range(len(w_q))
    ])

    hypotheses = [
        {"parameter_name": "payload_mass", "joint": None, "nominal_value": 0.0, "search_min": 0.1, "search_max": 2.5}
    ]

    solver = BatchedIsaacLabSysIDSolver(dynamics, num_envs=16)
    result = solver.optimize_parameters(w_q, w_dq, w_ddq, w_tau, hypotheses)

    assert result["converged"] is True
    est_payload = result["estimated_params"]["payload_mass"]
    # Check within 10% of true injected payload
    assert abs(est_payload - true_payload) < 0.15
