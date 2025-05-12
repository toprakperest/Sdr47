import numpy as np
from algorithms.signal_processing import SignalProcessing
from algorithms.ai_models import AIModels
from data.calibration import CalibrationData
from data import DataLogger

class HybridAnalysis:
    def __init__(self, signal_processor: SignalProcessing, ai_model_handler: AIModels,
                 calibration_module=None, geology_data=None):
        self.signal_processor = signal_processor
        self.ai_models = ai_model_handler
        self.calibration_data = calibration_module
        self.geology_data = geology_data if geology_data else self._load_default_turabdin_geology()
        self.current_terrain_type = None
        self.max_scan_depth_cm = 300  # Varsayılan maksimum tarama derinliği
        print("HybridAnalysis initialized.")

    def _load_default_turabdin_geology(self):
        return {
            "Kalker": {"dielectric_typical": 7.0, "attenuation_char": "low", "expected_echo": "sharp"},
            "Bazalt": {"dielectric_typical": 4.5, "attenuation_char": "moderate", "expected_echo": "multiple"},
            "Kil": {"dielectric_typical": 15.0, "attenuation_char": "high", "expected_echo": "damped"},
            "Kumtaşı": {"dielectric_typical": 4.0, "attenuation_char": "low", "expected_echo": "hyperbolic_potential"},
            "Boşluk": {"dielectric_typical": 1.0, "attenuation_char": "very_low", "expected_echo": "strong_amplitude_phase_inversion"},
            "Metal": {"dielectric_typical": None, "attenuation_char": "total_reflection", "expected_echo": "very_strong_double_peak"}
        }

    def set_current_terrain_type(self, terrain_type: str):
        if terrain_type in self.geology_data:
            self.current_terrain_type = terrain_type
            print(f"Terrain type set to {terrain_type}")
        else:
            print(f"Unknown terrain type {terrain_type}. Using general model.")
            self.current_terrain_type = "General"

    def cross_signal_confirmation(self, live_signal_features: dict, reflection_signal_features: dict, depth_tolerance_m=0.1):
        confidence = 0.0
        match_reasons = []

        live_anomaly_depth = live_signal_features.get("anomaly_depth_m")
        refl_anomaly_depth = reflection_signal_features.get("echo_depth_m")
        if live_anomaly_depth is not None and refl_anomaly_depth is not None:
            if abs(live_anomaly_depth - refl_anomaly_depth) <= depth_tolerance_m:
                confidence += 0.4
                match_reasons.append(f"Depth match ({live_anomaly_depth:.2f}m vs {refl_anomaly_depth:.2f}m)")

        live_attenuation = live_signal_features.get("attenuation_dB_m")
        refl_strength = reflection_signal_features.get("echo_strength_dB")
        refl_phase_inversion = reflection_signal_features.get("phase_inversion", False)

        if live_attenuation is not None and refl_strength is not None:
            if live_attenuation > 5 and refl_strength > -20 and refl_phase_inversion:
                confidence += 0.3
                match_reasons.append("Attenuation/Reflection/Phase indicative of void")

        print(f"Cross-Signal Confirmation: Confidence={confidence:.2f}, Reasons: {match_reasons}")
        return min(confidence, 1.0)

    def terrain_aware_decision(self, signal_features: dict, ai_prediction_probs: np.ndarray):
        if self.current_terrain_type is None or self.current_terrain_type == "General":
            return ai_prediction_probs, 1.0

        terrain_props = self.geology_data.get(self.current_terrain_type)
        if not terrain_props:
            return ai_prediction_probs, 1.0

        adjusted_probs = ai_prediction_probs.copy()
        confidence_modifier = 1.0
        idx_void = 2
        idx_metal = 3

        if self.current_terrain_type == "Kalker":
            if np.argmax(adjusted_probs) == idx_void and adjusted_probs[idx_void] > 0.6:
                confidence_modifier = 1.1
            elif np.argmax(adjusted_probs) == idx_metal and adjusted_probs[idx_metal] > 0.7:
                confidence_modifier = 0.8

        elif self.current_terrain_type == "Mineralli Toprak":
            if np.argmax(adjusted_probs) == idx_metal and adjusted_probs[idx_metal] > 0.6:
                if signal_features.get("rcs_signature", "weak") == "strong_metallic":
                    confidence_modifier = 1.05
                else:
                    confidence_modifier = 0.85

        return adjusted_probs, confidence_modifier

    def apply_angular_rejection(self, iq_data, antenna_beamwidth_deg=60):
        print("Angular Rejection: Placeholder")
        return iq_data

    def statistical_noise_mapping_update(self, current_scan_features, location_key):
        print(f"Statistical Noise Mapping for {location_key}: Placeholder.")
        return current_scan_features

    def combine_results(self, processed_live_signal, processed_reflection_signal,
                        ai_preds_live, ai_preds_reflection,
                        rx2_calibration_features=None, current_location_key=None):
        final_decision = {"target_detected": False, "confidence": 0.0}
        overall_confidence = 0.0
        detected_targets = []

        if rx2_calibration_features:
            terrain = rx2_calibration_features.get("estimated_terrain_type", "General")
            self.set_current_terrain_type(terrain)

        cross_signal_conf = self.cross_signal_confirmation(processed_live_signal, processed_reflection_signal)
        overall_confidence += cross_signal_conf * 0.3

        if ai_preds_reflection is not None and len(ai_preds_reflection) > 0:
            main_ai_pred_probs = ai_preds_reflection[0] if isinstance(ai_preds_reflection, list) or ai_preds_reflection.ndim > 1 else ai_preds_reflection
            adjusted_ai_probs, terrain_modifier = self.terrain_aware_decision(processed_reflection_signal, main_ai_pred_probs)
            ai_max_prob = np.max(adjusted_ai_probs)
            ai_predicted_class_idx = np.argmax(adjusted_ai_probs)
            class_names = list(self.geology_data.keys())
            ai_predicted_class_name = class_names[ai_predicted_class_idx] if ai_predicted_class_idx < len(class_names) else "Unknown"

            overall_confidence += (ai_max_prob * terrain_modifier) * 0.5
            if ai_max_prob * terrain_modifier > 0.6:
                detected_targets.append({
                    "target_type": ai_predicted_class_name,
                    "depth_m": processed_reflection_signal.get("echo_depth_m"),
                    "ai_confidence": ai_max_prob,
                    "terrain_modifier": terrain_modifier,
                    "source": "AI_Reflection"
                })

        if processed_reflection_signal.get("phase_inversion") and processed_reflection_signal.get("echo_strength_dB", -100) > -15:
            is_void_already_high_conf = any(t["target_type"] == "Boşluk" and t["ai_confidence"] * t.get("terrain_modifier", 1) > 0.7 for t in detected_targets)
            if not is_void_already_high_conf:
                overall_confidence += 0.2
                target_info_rule = {
                    "target_type": "Boşluk",
                    "depth_m": processed_reflection_signal.get("echo_depth_m"),
                    "source": "Rule_PhaseInversion"
                }
                if not any(t["target_type"] == "Boşluk" and t["depth_m"] == target_info_rule["depth_m"] for t in detected_targets):
                    detected_targets.append(target_info_rule)

        final_confidence = min(overall_confidence, 1.0)
        layers = []

        if detected_targets:
            sorted_targets = sorted(detected_targets, key=lambda x: x["depth_m"] * 100)
            prev_depth = 0
            for target in sorted_targets:
                depth_cm = target["depth_m"] * 100
                layers.append((target["target_type"], prev_depth, depth_cm))
                prev_depth = depth_cm
            if prev_depth < self.max_scan_depth_cm:
                layers.append(("Bilinmeyen", prev_depth, self.max_scan_depth_cm))

            best_target = sorted(detected_targets, key=lambda t: t.get("ai_confidence", 0), reverse=True)[0]
            final_decision.update(best_target)
            final_decision["target_detected"] = True
            final_decision["confidence"] = final_confidence
            final_decision["layers"] = layers
        else:
            final_decision["confidence"] = final_confidence
            final_decision["layers"] = [("Bilinmeyen", 0, self.max_scan_depth_cm)]

        print(f"HybridAnalysis Decision: {final_decision}")
        return final_decision

# Example Usage
if __name__ == "__main__":
    class MockSignalProcessing:
        def __init__(self, sample_rate): self.sample_rate = sample_rate
    class MockAIModels:
        def predict_cnn(self, data): return np.array([[0.1, 0.2, 0.7]])
        def predict_lstm(self, data): return np.array([[0.2, 0.6, 0.2]])

    mock_sp = MockSignalProcessing(sample_rate=100e6)
    mock_ai = MockAIModels()
    hybrid_analyzer = HybridAnalysis(signal_processor=mock_sp, ai_model_handler=mock_ai)

    live_features = {"anomaly_depth_m": 2.1, "attenuation_dB_m": 6}
    refl_features = {"echo_depth_m": 2.0, "echo_strength_dB": -10, "phase_inversion": True}
    ai_preds_refl = mock_ai.predict_cnn(None)
    rx2_calib = {"estimated_terrain_type": "Kalker"}

    decision = hybrid_analyzer.combine_results(
        processed_live_signal=live_features,
        processed_reflection_signal=refl_features,
        ai_preds_live=None,
        ai_preds_reflection=ai_preds_refl,
        rx2_calibration_features=rx2_calib
    )

    print(f"Final Decision: {decision}")
