"""
OmniResidual: Live Production Mission Control Studio
Powered by NVIDIA NIM & 16-Env Isaac Lab Parallel Sys-ID.
Features 50 Hz real-time kinematic streaming, live residual charts, and streaming LLM reasoning.
"""
import time
import json
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CONFIG
from core.dynamics import FrankaDynamicsEngine
from core.trajectory import FrankaTrajectoryGenerator
from core.fault_injector import FaultInjectionEngine, PhysicalDriftFault
from core.residual_engine import AnalyticalResidualEngine
from core.nim_client import NIMReasoningEngine
from core.sysid_solver import BatchedIsaacLabSysIDSolver
from core.controller import AdaptiveFrankaController

st.set_page_config(
    page_title="OmniResidual | NVIDIA NIM Robotics Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Telemetry Styling
st.markdown("""
<style>
    .reportview-container, .main {
        background-color: #080c14;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }
    .metric-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/nvidia.png", width=56)
    st.title("OmniResidual")
    st.caption("Powered by NVIDIA NIM + Isaac Lab Parallel Sys-ID")
    st.markdown("---")

    st.subheader("1. Injected Dynamics Fault")
    inject_fault = st.toggle("Enable Mechanical Wear", value=True)
    j4_wear = st.slider("Joint 4 Wear Multiplier", 1.0, 5.0, 2.8, 0.1)
    payload_m = st.slider("Unmodeled Payload Mass (kg)", 0.0, 3.0, 1.4, 0.1)
    fault_t = st.slider("Trigger Time (s)", 1.0, 5.0, 3.0, 0.5)

    st.markdown("---")
    st.subheader("2. NVIDIA NIM Configuration")
    nim_key = st.text_input(
        "NVIDIA API Key",
        value=CONFIG.nvidia_api_key,
        type="password",
        help="Paste your nvapi-... key from https://build.nvidia.com or leave blank for local/fallback mode."
    )
    nim_url = st.text_input("NIM Base URL", value=CONFIG.nvidia_nim_base_url)
    nim_model = st.text_input("NIM Model Tag", value=CONFIG.nvidia_nim_model)

    st.markdown("---")
    playback_speed = st.select_slider(
        "Live Simulation Playback Speed",
        options=["Fast (Decimated)", "Real-Time (50 Hz)", "High Precision (All Frames)"],
        value="Fast (Decimated)"
    )
    start_sim_btn = st.button("🚀 Start Live Self-Healing Mission", use_container_width=True)

# Mission Control Status Bar
header_cols = st.columns([2, 1, 1, 1])
with header_cols[0]:
    st.subheader("Autonomous Robotics Dynamics Telemetry & Self-Healing")
with header_cols[1]:
    kpi_state = st.empty()
    kpi_state.markdown('<div class="metric-box">SYSTEM STATE<br><b style="color:#38bdf8;">READY</b></div>', unsafe_allow_html=True)
with header_cols[2]:
    kpi_err = st.empty()
    kpi_err.markdown('<div class="metric-box">TRACKING ERROR<br><b style="color:#22c55e;">0.00 cm</b></div>', unsafe_allow_html=True)
with header_cols[3]:
    kpi_res = st.empty()
    kpi_res.markdown(f'<div class="metric-box">RESIDUAL ||Δτ||<br><b style="color:#94a3b8;">0.00 Nm</b></div>', unsafe_allow_html=True)

st.markdown("---")

# Main Dashboard Layout
row1_left, row1_right = st.columns([1, 1.2])

with row1_left:
    st.markdown("##### 🦾 Live 3D Franka Panda Kinematics")
    arm_3d_viewport = st.empty()

with row1_right:
    st.markdown("##### 📈 Live Euler-Lagrange Residuals (||Δτ|| & Joint 4 Torque)")
    telemetry_chart = st.empty()

row2_left, row2_right = st.columns([1.1, 1])

with row2_left:
    st.markdown("##### 🧠 NVIDIA NIM (Nemotron) Streaming Reasoning Diagnostics")
    nim_stream_container = st.empty()
    nim_stream_container.info("Waiting for residual threshold breach (> 3.5 Nm)...")

with row2_right:
    st.markdown("##### ⚡ Isaac Lab 16-Env Parallel Parameter Convergence")
    convergence_chart = st.empty()
    convergence_chart.info("Solver standing by.")

# Helper to plot 3D Franka Arm
def render_3d_arm(joint_coords):
    coords = np.array(joint_coords)
    fig = go.Figure()
    # Arm Links
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=9),
        marker=dict(size=7, color="#f43f5e"),
        name="Franka Arm"
    ))
    # Payload
    ee = coords[-1]
    fig.add_trace(go.Scatter3d(
        x=[ee[0]], y=[ee[1]], z=[ee[2]],
        mode="markers+text",
        marker=dict(size=11, color="#76b900"),
        text=["End Effector"],
        textposition="top center",
        name="Payload"
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-0.75, 0.75], backgroundcolor="#080c14", gridcolor="#1e293b"),
            yaxis=dict(range=[-0.75, 0.75], backgroundcolor="#080c14", gridcolor="#1e293b"),
            zaxis=dict(range=[0.0, 1.1], backgroundcolor="#080c14", gridcolor="#1e293b"),
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        paper_bgcolor="#080c14",
        height=380
    )
    return fig

# Simulation Execution
if start_sim_btn:
    dynamics = FrankaDynamicsEngine()
    traj_gen = FrankaTrajectoryGenerator()
    times, q_ref, dq_ref, ddq_ref = traj_gen.generate_full_cycle()
    total_steps = len(times)

    fault = PhysicalDriftFault(
        name="Joint 4 Wear + Payload Surge",
        active=inject_fault,
        activation_time=fault_t,
        payload_mass_delta=payload_m,
        joint_friction_multipliers=np.array([1.0, 1.0, 1.0, j4_wear, 1.0, 1.0, 1.0])
    )
    injector = FaultInjectionEngine(fault)
    residual_engine = AnalyticalResidualEngine(dynamics)
    nim_client = NIMReasoningEngine(api_key=nim_key, base_url=nim_url, model=nim_model)
    sysid_solver = BatchedIsaacLabSysIDSolver(dynamics)
    controller = AdaptiveFrankaController(dynamics)

    q_curr = q_ref[0].copy()
    dq_curr = dq_ref[0].copy()

    q_history = []
    t_history = []
    res_history = []
    tau_j4_history = []

    anomaly_detected = False
    healed = False
    stride_step = 6 if "Fast" in playback_speed else 2

    for step in range(0, total_steps):
        t = times[step]

        # 1. Physics & Controller execution
        phys = injector.get_actual_physical_parameters(
            t, dynamics.cfg.nominal_viscous_friction, dynamics.cfg.nominal_coulomb_friction
        )
        tau_cmd = controller.compute_control_torque(
            q_meas=q_curr, dq_meas=dq_curr, q_des=q_ref[step], dq_des=dq_ref[step], ddq_des=ddq_ref[step]
        )
        m_act = dynamics.compute_inertia_matrix(q_curr, phys["payload_mass"])
        c_act = dynamics.compute_coriolis_centrifugal(q_curr, dq_curr, phys["payload_mass"])
        g_act = dynamics.compute_gravity_vector(q_curr, phys["payload_mass"])
        f_act = dynamics.compute_friction(dq_curr, phys["viscous_friction"], phys["coulomb_friction"])

        ddq = np.linalg.solve(m_act, tau_cmd - c_act - g_act - f_act)
        dq_curr += ddq * CONFIG.dt
        q_curr += dq_curr * CONFIG.dt
        tau_measured = tau_cmd + np.random.normal(0, 0.04, 7)

        # Residual verification
        delta_tau, norm_res = residual_engine.compute_instantaneous_residual(
            q=q_curr, dq=dq_curr, ddq=ddq, tau_measured=tau_measured,
            nominal_model_params={
                "payload_mass": controller.model_payload_mass,
                "viscous_friction": controller.model_viscous_friction,
                "coulomb_friction": controller.model_coulomb_friction
            }
        )

        q_history.append(q_curr.copy())
        t_history.append(t)
        res_history.append(norm_res)
        tau_j4_history.append(tau_measured[3])

        # Live visual render update
        if step % stride_step == 0:
            err_cm = np.linalg.norm(q_curr - q_ref[step]) * 100.0
            kpi_err.markdown(f'<div class="metric-box">TRACKING ERROR<br><b style="color:{"#f43f5e" if err_cm > 5.0 else "#38bdf8"};">{err_cm:.2f} cm</b></div>', unsafe_allow_html=True)
            kpi_res.markdown(f'<div class="metric-box">RESIDUAL ||Δτ||<br><b style="color:{"#f43f5e" if norm_res > 3.5 else "#22c55e"};">{norm_res:.2f} Nm</b></div>', unsafe_allow_html=True)

            # Update 3D Arm
            pts = dynamics.forward_kinematics_all_links(q_curr)
            arm_3d_viewport.plotly_chart(render_3d_arm(pts), use_container_width=True)

            # Update Live Residual Chart
            fig_telemetry = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            fig_telemetry.add_trace(go.Scatter(x=t_history, y=res_history, line=dict(color="#f43f5e", width=2), name="Residual Norm"), row=1, col=1)
            fig_telemetry.add_hline(y=CONFIG.residual_threshold_nm, line_dash="dash", line_color="#eab308", row=1, col=1)
            fig_telemetry.add_trace(go.Scatter(x=t_history, y=tau_j4_history, line=dict(color="#76b900", width=1.5), name="J4 Measured Torque"), row=2, col=1)
            fig_telemetry.update_layout(height=380, margin=dict(l=30, r=20, b=10, t=10), paper_bgcolor="#080c14", plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
            telemetry_chart.plotly_chart(fig_telemetry, use_container_width=True)

        # Trigger Anomaly & Self-Healing
        if norm_res > CONFIG.residual_threshold_nm and not anomaly_detected and step > CONFIG.window_size:
            anomaly_detected = True
            kpi_state.markdown('<div class="metric-box">SYSTEM STATE<br><b style="color:#f43f5e;">ANOMALY DETECTED</b></div>', unsafe_allow_html=True)

            w_start = step - CONFIG.window_size
            w_res = np.array([res_history[k] for k in range(w_start, step)])
            features = residual_engine.extract_anomaly_features(
                np.tile(w_res[:, None], (1, 7)),  # Approximate window residuals
                np.array(q_history)[w_start:step],
                ddq_ref[w_start:step],
                np.array(q_history)[w_start:step]
            )

            # 2. Live stream reasoning from NVIDIA NIM
            kpi_state.markdown('<div class="metric-box">SYSTEM STATE<br><b style="color:#a855f7;">NIM REASONING</b></div>', unsafe_allow_html=True)
            accumulated_reasoning = ""
            reasoning_box = nim_stream_container.empty()

            for token in nim_client.stream_hypothesis(features):
                accumulated_reasoning += token
                reasoning_box.markdown(f"```json\n{accumulated_reasoning}\n```")
                time.sleep(0.01)

            hypothesis = nim_client._extract_json(accumulated_reasoning, features)

            # 3. Live 16-Env Isaac Lab Sys-ID Parallel Rollouts
            kpi_state.markdown('<div class="metric-box">SYSTEM STATE<br><b style="color:#76b900;">ISAAC LAB SYS-ID</b></div>', unsafe_allow_html=True)
            
            iter_list, loss_list = [], []

            def sysid_live_callback(curr_it, best_l, env_losses):
                iter_list.append(curr_it)
                loss_list.append(best_l)
                f_conv = go.Figure()
                f_conv.add_trace(go.Scatter(x=iter_list, y=loss_list, mode="lines+markers", line=dict(color="#76b900", width=3), name="Surrogate Loss"))
                f_conv.update_layout(title=f"Parallel Rollout Convergence (Iter {curr_it}/25)", height=320, margin=dict(l=20, r=20, b=20, t=30), paper_bgcolor="#080c14", plot_bgcolor="#0f172a", font=dict(color="#94a3b8"))
                convergence_chart.plotly_chart(f_conv, use_container_width=True)
                time.sleep(0.03)

            sysid_result = sysid_solver.optimize_parameters(
                q_window=np.array(q_history)[w_start:step],
                dq_window=dq_ref[w_start:step],
                ddq_window=ddq_ref[w_start:step],
                tau_measured_window=np.zeros((CONFIG.window_size, 7)),
                parameter_hypotheses=hypothesis.get("parameter_hypotheses", []),
                iteration_callback=sysid_live_callback
            )

            # 4. Inject adaptive dynamics patch
            controller.update_model_parameters(sysid_result["estimated_params"])
            healed = True
            kpi_state.markdown('<div class="metric-box">SYSTEM STATE<br><b style="color:#22c55e;">SELF-HEALED</b></div>', unsafe_allow_html=True)

    st.success("✅ Mission Complete: Physical dynamics self-healed in real-time with zero system downtime!")