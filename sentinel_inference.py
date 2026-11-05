"""Smart Neonatal Incubator Sentinel — Two-Stage Cry Recognition + Incubator Context Fusion.

When Stage 1 detects an infant cry, Stage 2 computes the acoustic probability distribution over the
5 cry reasons (hungry, belly_pain, burping, discomfort, tired) and fuses it with the real-time
incubator sensor context (temperature, humidity, wet diaper/moisture sensor, feeding timer, awake duration).

Usage:
  python sentinel_inference.py                     # runs built-in demonstration scenarios
  python sentinel_inference.py --audio <wav_path> --temp 35.2 --humidity 76 --hours-fed 1.2 --wet-diaper
"""
import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from features import SILENCE_DBFS, load_audio, logmel, rms_dbfs, windows

MODELS_DIR = "models"
REASONS = ["hungry", "belly_pain", "burping", "discomfort", "tired"]

# Normal neonatal incubator comfort ranges
TEMP_MIN_C, TEMP_MAX_C = 32.0, 34.5
HUM_MIN_PCT, HUM_MAX_PCT = 50.0, 65.0


@dataclass
class IncubatorState:
    """Real-time incubator telemetry at the moment crying is detected."""
    temp_c: float = 33.2                  # Incubator air temperature (°C)
    humidity_pct: float = 58.0            # Relative humidity (%)
    hours_since_feeding: float = 1.5      # Hours elapsed since last feeding
    wet_diaper: bool = False              # Moisture / wet-mattress sensor state
    awake_minutes: float = 30.0           # Continuous awake/restless duration (minutes)


def compute_context_priors(state: IncubatorState) -> tuple[np.ndarray, list[str]]:
    """Compute context multiplier weights W_sensor for [hungry, belly_pain, burping, discomfort, tired]
    and human-readable incubator condition notes.
    """
    w = {r: 1.0 for r in REASONS}
    notes = []

    # 1. Temperature check (normal: 32.0 - 34.5 °C)
    if state.temp_c > TEMP_MAX_C:
        delta = state.temp_c - TEMP_MAX_C
        w["discomfort"] *= 2.2 + min(delta * 0.8, 1.8)
        notes.append(f"Suhu inkubator terlalu hangat ({state.temp_c:.1f} °C > {TEMP_MAX_C:.1f} °C)")
    elif state.temp_c < TEMP_MIN_C:
        delta = TEMP_MIN_C - state.temp_c
        w["discomfort"] *= 2.2 + min(delta * 0.8, 1.8)
        notes.append(f"Suhu inkubator terlalu dingin ({state.temp_c:.1f} °C < {TEMP_MIN_C:.1f} °C)")
    else:
        notes.append(f"Suhu inkubator normal ({state.temp_c:.1f} °C)")

    # 2. Humidity & wet diaper check (normal: 50 - 65 %)
    if state.wet_diaper:
        w["discomfort"] *= 2.8
        notes.append("Sensor kelembapan kasur/popok mendeteksi basah (wet_diaper = True)")
    if state.humidity_pct > HUM_MAX_PCT:
        w["discomfort"] *= 1.6
        notes.append(f"Kelembapan inkubator tinggi ({state.humidity_pct:.0f}% > {HUM_MAX_PCT:.0f}%)")
    elif state.humidity_pct < HUM_MIN_PCT:
        w["discomfort"] *= 1.4
        notes.append(f"Kelembapan inkubator rendah ({state.humidity_pct:.0f}% < {HUM_MIN_PCT:.0f}%)")
    else:
        notes.append(f"Kelembapan inkubator normal ({state.humidity_pct:.0f}%)")

    # 3. Feeding schedule check
    if state.hours_since_feeding >= 2.5:
        boost = 2.0 + min((state.hours_since_feeding - 2.5) * 0.8, 2.0)
        w["hungry"] *= boost
        w["burping"] *= 0.4
        notes.append(f"Sudah {state.hours_since_feeding:.1f} jam sejak menyusu terakhir (waktunya menyusu)")
    elif state.hours_since_feeding <= 0.5:
        w["hungry"] *= 0.25
        w["burping"] *= 3.2
        w["belly_pain"] *= 1.5
        notes.append(f"Baru selesai menyusu {state.hours_since_feeding * 60:.0f} menit lalu (indikasi perlu sendawa/kolik)")
    else:
        notes.append(f"Jarak waktu menyusu normal ({state.hours_since_feeding:.1f} jam lalu)")

    # 4. Awake / restless duration check
    if state.awake_minutes >= 75.0:
        w["tired"] *= 2.5 + min((state.awake_minutes - 75.0) / 60.0, 1.5)
        notes.append(f"Bayi sudah terjaga selama {state.awake_minutes:.0f} menit (indikasi kelelahan/mengantuk)")

    weights = np.array([w[r] for r in REASONS], dtype=np.float32)
    return weights, notes


class TFLiteInt8Model:
    """Wrapper around a full-int8 TFLite model and its normalisation metadata."""

    def __init__(self, tflite_path: str, norm_path: str):
        with open(norm_path) as f:
            self.meta = json.load(f)
        with open(tflite_path, "rb") as f:
            self.content = f.read()
        self.it = tf.lite.Interpreter(model_content=self.content)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def predict_logmel(self, X_logmel: np.ndarray) -> np.ndarray:
        X = np.maximum(X_logmel.astype(np.float32), self.meta["floor_db"])
        X_norm = (X - self.meta["mean"]) / self.meta["std"]
        self.it.resize_tensor_input(self.inp["index"], [len(X_norm), *X_norm.shape[1:]])
        self.it.allocate_tensors()
        s_in, z_in = self.inp["quantization"]
        q_in = np.clip(np.round(X_norm / s_in + z_in), -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp["index"], q_in)
        self.it.invoke()
        s_out, z_out = self.out["quantization"]
        probs = (self.it.get_tensor(self.out["index"]).astype(np.float32) - z_out) * s_out
        return probs


class SentinelPipeline:
    """End-to-end two-stage infant cry detector + incubator sensor fusion."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.stage1 = TFLiteInt8Model(
            os.path.join(models_dir, "stage1_cry_int8.tflite"),
            os.path.join(models_dir, "stage1_norm.json"),
        )
        self.stage2 = TFLiteInt8Model(
            os.path.join(models_dir, "stage2_reason_int8.tflite"),
            os.path.join(models_dir, "stage2_norm.json"),
        )

    def analyze(self, audio_path: str, state: IncubatorState) -> dict:
        x = load_audio(audio_path)
        w = windows(x)
        db = rms_dbfs(w)
        loud = db >= SILENCE_DBFS
        feats = logmel(w)

        # Stage 1 inference
        p1 = self.stage1.predict_logmel(feats)  # columns: [non-cry, cry, silence]
        p_cry_win = p1[:, 1]
        mean_p_cry = float(p_cry_win[loud].mean() if loud.any() else p_cry_win.mean())
        run3 = np.convolve(p_cry_win, np.ones(3) / 3, mode="valid") if len(p_cry_win) >= 3 else p_cry_win
        device_rule_score = float(run3.max())
        is_crying = bool(mean_p_cry >= 0.5 or device_rule_score >= 0.80)

        result = dict(
            audio_path=audio_path,
            n_windows=int(len(w)),
            non_silent_windows=int(loud.sum()),
            stage1_mean_p_cry=round(mean_p_cry, 4),
            stage1_device_rule_score=round(device_rule_score, 4),
            device_alarm_triggered=bool(device_rule_score >= 0.80),
            is_crying=is_crying,
            incubator_state=asdict(state),
        )

        if not is_crying:
            result["status"] = "AMAN (Tidak Terdeteksi Tangisan Bayi)"
            return result

        # Stage 2 inference on active (non-silent) cry windows
        active_feats = feats[loud] if loud.any() else feats
        p2_win = self.stage2.predict_logmel(active_feats)
        p_audio = p2_win.mean(axis=0)
        p_audio = np.maximum(p_audio, 1e-4)
        p_audio = p_audio / p_audio.sum()

        # Context fusion: P_fused(c) = normalize(P_audio(c) * W_sensor(c))
        w_sensor, notes = compute_context_priors(state)
        p_fused = p_audio * w_sensor
        p_fused = p_fused / p_fused.sum()

        audio_top_idx = int(p_audio.argmax())
        fused_top_idx = int(p_fused.argmax())

        result.update(
            status="PERINGATAN: Tangisan Bayi Terdeteksi!",
            incubator_notes=notes,
            stage2_audio_only={r: round(float(p), 4) for r, p in zip(REASONS, p_audio)},
            stage2_audio_top_reason=REASONS[audio_top_idx],
            context_weights={r: round(float(wk), 2) for r, wk in zip(REASONS, w_sensor)},
            stage2_context_fused={r: round(float(p), 4) for r, p in zip(REASONS, p_fused)},
            final_predicted_reason=REASONS[fused_top_idx],
            final_confidence=round(float(p_fused[fused_top_idx]), 4),
        )
        return result


def print_report(res: dict, title: str = ""):
    print("=" * 78)
    if title:
        print(f"SKENARIO: {title}")
        print("-" * 78)
    print(f"File Audio        : {res['audio_path']}")
    print(f"Stage 1 P(cry)    : Mean = {res['stage1_mean_p_cry']:.1%}, "
          f"3-Window Device Rule = {res['stage1_device_rule_score']:.1%} "
          f"[Alarm: {'AKTIF' if res['device_alarm_triggered'] else 'TIDAK'}]")
    print(f"Status Deteksi    : {res['status']}")
    if not res["is_crying"]:
        print("=" * 78 + "\n")
        return

    st = res["incubator_state"]
    print(f"Kondisi Inkubator : Suhu={st['temp_c']}°C | Lembap={st['humidity_pct']}% | "
          f"Menyusu Terakhir={st['hours_since_feeding']} jam lalu | "
          f"Popok Basah={st['wet_diaper']} | Terjaga={st['awake_minutes']} mnt")
    for n in res["incubator_notes"]:
        print(f"  - {n}")
    print("\nPerbandingan Probabilitas Alasan Tangisan (Stage 2 Audio vs Audio + Konteks Inkubator):")
    print(f"  {'Alasan':<14} | {'Audio Murni':>12} | {'Bobot Sensor':>12} | {'Hasil Gabungan (Fused)':>22}")
    print("  " + "-" * 68)
    for r in REASONS:
        pa = res["stage2_audio_only"][r]
        ws = res["context_weights"][r]
        pf = res["stage2_context_fused"][r]
        mark = " <-- UTAMA" if r == res["final_predicted_reason"] else ""
        print(f"  {r:<14} | {pa:>11.1%} | {ws:>11.2f}x | {pf:>20.1%}{mark}")
    print(f"\n>>> KESIMPULAN AKHIR: {res['final_predicted_reason'].upper()} "
          f"(Confidence: {res['final_confidence']:.1%}, Audio Murni: {res['stage2_audio_top_reason']})")
    print("=" * 78 + "\n")


def run_demo():
    df = pd.read_csv("data_clips.csv", keep_default_na=False)
    pipe = SentinelPipeline()

    def pick_clip(source: str, reason: str = "", category: str = "", idx: int = 0) -> str:
        m = df["source"] == source
        if reason:
            m &= df["reason"] == reason
        if category:
            m &= df["esc_category"] == category
        return df[m].iloc[idx]["path"]

    demos = [
        (
            "1. Bayi Menangis + Inkubator Terlalu Panas & Popok Basah -> Discomfort",
            pick_clip("dac", reason="discomfort", idx=0),
            IncubatorState(temp_c=35.4, humidity_pct=74.0, hours_since_feeding=1.2, wet_diaper=True, awake_minutes=25),
        ),
        (
            "2. Bayi Menangis + Baru Selesai Menyusu 15 Menit Lalu -> Burping (Perlu Sendawa)",
            pick_clip("dac", reason="burping", idx=0),
            IncubatorState(temp_c=33.3, humidity_pct=56.0, hours_since_feeding=0.25, wet_diaper=False, awake_minutes=20),
        ),
        (
            "3. Bayi Menangis + Sudah 3,5 Jam Belum Menyusu (Inkubator Normal) -> Hungry",
            pick_clip("dac", reason="hungry", idx=2),
            IncubatorState(temp_c=33.1, humidity_pct=58.0, hours_since_feeding=3.5, wet_diaper=False, awake_minutes=15),
        ),
        (
            "4. Bayi Menangis + Sudah Terjaga 110 Menit (Inkubator Normal) -> Tired",
            pick_clip("dac", reason="tired"),
            IncubatorState(temp_c=33.0, humidity_pct=55.0, hours_since_feeding=1.2, wet_diaper=False, awake_minutes=110),
        ),
        (
            "5. Suara Bising Lingkungan (ESC-50) -> Stage 1 Menahan Alarm (Aman)",
            pick_clip("esc", category="vacuum_cleaner"),
            IncubatorState(temp_c=35.0, humidity_pct=70.0, hours_since_feeding=4.0, wet_diaper=True, awake_minutes=90),
        ),
    ]

    out_records = []
    for title, path, state in demos:
        res = pipe.analyze(path, state)
        print_report(res, title=title)
        out_records.append(dict(scenario=title, **res))

    with open("sentinel_demo_results.json", "w") as f:
        json.dump(out_records, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="Path to a WAV file to analyze (omit to run built-in demo scenarios)")
    ap.add_argument("--temp", type=float, default=33.2, help="Incubator temperature in °C")
    ap.add_argument("--humidity", type=float, default=58.0, help="Incubator relative humidity in %%")
    ap.add_argument("--hours-fed", type=float, default=1.5, help="Hours since last feeding")
    ap.add_argument("--wet-diaper", action="store_true", help="Wet diaper / mattress moisture detected")
    ap.add_argument("--awake-min", type=float, default=30.0, help="Continuous awake duration in minutes")
    a = ap.parse_args()

    if not a.audio:
        run_demo()
    else:
        pipe = SentinelPipeline()
        state = IncubatorState(
            temp_c=a.temp,
            humidity_pct=a.humidity,
            hours_since_feeding=a.hours_fed,
            wet_diaper=a.wet_diaper,
            awake_minutes=a.awake_min,
        )
        res = pipe.analyze(a.audio, state)
        print_report(res, title="Custom Audio & Incubator State")


if __name__ == "__main__":
    main()

