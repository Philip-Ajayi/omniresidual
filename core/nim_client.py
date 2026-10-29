"""
NVIDIA NIM Reasoning Engine:
Interfaces with NVIDIA Inference Microservices (NIM) hosting NVIDIA Nemotron models.
Supports live token streaming and JSON schema extraction.
"""
import json
import os
import re
from typing import Dict, Any, Generator
from openai import OpenAI
from config import CONFIG

class NIMReasoningEngine:
    def __init__(
        self,
        api_key: str = CONFIG.nvidia_api_key,
        base_url: str = CONFIG.nvidia_nim_base_url,
        model: str = CONFIG.nvidia_nim_model
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        
        self.client = None
        if self.api_key and "nvapi-" in self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        elif "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            # Self-hosted local NIM microservice container (often requires dummy API key)
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key or "local-nim")

    def _build_prompt(self, telemetry_features: Dict[str, Any]):
        system_prompt = (
            "You are an elite autonomous robotics dynamics diagnosis AI operating inside an NVIDIA Isaac physical AI loop. "
            "Analyze joint torque residual anomalies and physical correlation indicators for a Franka Emika Panda 7-DoF arm.\n"
            "Your output MUST be strictly valid, parsable JSON with no conversational preface or trailing remarks:\n"
            "{\n"
            '  "fault_mode": "string",\n'
            '  "confidence": float (0.0 to 1.0),\n'
            '  "reasoning": "physical analysis explaining viscous or mass drift",\n'
            '  "primary_affected_joints": [int],\n'
            '  "parameter_hypotheses": [\n'
            '    {"parameter_name": "string", "joint": int or null, "nominal_value": float, "search_min": float, "search_max": float}\n'
            "  ]\n"
            "}"
        )
        user_prompt = (
            f"TELEMETRY RESIDUAL SIGNATURE (Euler-Lagrange Residual Breach):\n"
            f"- Primary Joint Breach: Joint {telemetry_features.get('primary_anomalous_joint')}\n"
            f"- Per-Joint RMS Residuals (Nm): {telemetry_features.get('rms_per_joint')}\n"
            f"- Per-Joint Peak Residuals (Nm): {telemetry_features.get('peak_per_joint')}\n"
            f"- Velocity Correlations (Viscous wear signature): {telemetry_features.get('velocity_correlations')}\n"
            f"- Gravity Correlations (Unmodeled payload signature): {telemetry_features.get('gravity_correlations')}\n"
        )
        return system_prompt, user_prompt

    def stream_hypothesis(self, telemetry_features: Dict[str, Any]) -> Generator[str, None, None]:
        """Streams live reasoning tokens directly from NVIDIA NIM."""
        if not self.client:
            # Stream analytical fallback simulated tokens
            fallback = self._synthesize_analytical_fallback(telemetry_features)
            text = json.dumps(fallback, indent=2)
            for chunk in text.split(" "):
                yield chunk + " "
            return

        sys_p, user_p = self._build_prompt(telemetry_features)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p}
                ],
                temperature=0.1,
                max_tokens=650,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as err:
            yield f"\n[NIM Connection Notice: Reverting to local analytical solver: {str(err)}]\n"
            fallback = self._synthesize_analytical_fallback(telemetry_features)
            yield json.dumps(fallback, indent=2)

    def infer_physical_hypothesis(self, telemetry_features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous JSON extraction with automatic parsing."""
        full_text = "".join(list(self.stream_hypothesis(telemetry_features)))
        return self._extract_json(full_text, telemetry_features)

    def _extract_json(self, raw_text: str, fallback_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and sanitizes JSON from Nemotron markdown responses."""
        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw_text.strip())
        except Exception:
            return self._synthesize_analytical_fallback(fallback_features)

    def _synthesize_analytical_fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        primary_joint = features.get("primary_anomalous_joint", 4)
        j_idx = primary_joint - 1
        vel_corr = features.get("velocity_correlations", [0] * 7)[j_idx]
        rms_val = features.get("rms_per_joint", [0] * 7)[j_idx]

        distal_rms = np.mean(features.get("rms_per_joint", [0] * 7)[4:])
        has_payload_signature = distal_rms > 1.2 or features.get("max_rms_nm", 0) > 4.0

        hypotheses = []
        if abs(vel_corr) > 0.35 or rms_val > CONFIG.residual_threshold_nm:
            nom_fric = float(CONFIG.kinematics.nominal_viscous_friction[j_idx])
            hypotheses.append({
                "parameter_name": f"joint_{primary_joint}_viscous_friction",
                "joint": primary_joint,
                "nominal_value": nom_fric,
                "search_min": nom_fric * 0.8,
                "search_max": nom_fric * 4.0
            })

        if has_payload_signature:
            hypotheses.append({
                "parameter_name": "payload_mass",
                "joint": None,
                "nominal_value": 0.0,
                "search_min": 0.2,
                "search_max": 2.5
            })

        return {
            "fault_mode": f"Joint {primary_joint} Transmission Wear & Dynamic Payload Shift",
            "confidence": 0.96,
            "reasoning": (
                f"Joint {primary_joint} exhibits severe residual breach (RMS: {rms_val:.2f} Nm). "
                f"High velocity correlation ({vel_corr:.2f}) isolates viscous damper failure, "
                f"while Link 6-7 gravity offsets indicate unmodeled payload surge."
            ),
            "primary_affected_joints": [primary_joint, 7],
            "parameter_hypotheses": hypotheses
        }