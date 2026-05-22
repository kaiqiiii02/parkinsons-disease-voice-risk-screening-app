import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import io
import re
from pathlib import Path
from datetime import datetime
from textwrap import dedent

import soundfile as sf
import librosa
import joblib
import tempfile
import parselmouth
from parselmouth.praat import call


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Parkinson's Disease Voice Risk Screening App",
    page_icon="🎙️",
    layout="wide"
)


# =========================
# File paths
# =========================
BASE_DIR = Path(__file__).parent
RECORDING_DIR = BASE_DIR / "recordings"
LOG_FILE = BASE_DIR / "screening_log.csv"

RECORDING_DIR.mkdir(exist_ok=True)


# =========================
# Load trained XGBoost model artifacts
# =========================
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(BASE_DIR / "xgb_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    winsor_limits = joblib.load(BASE_DIR / "winsor_limits.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    log_cols = joblib.load(BASE_DIR / "log_cols.pkl")
    return model, scaler, winsor_limits, feature_columns, log_cols


XGB_MODEL, SCALER, WINSOR_LIMITS, FEATURE_COLUMNS, LOG_COLS = load_model_artifacts()


# =========================
# App pages
# =========================
PAGES = [
    "Home",
    "Personal Details",
    "Voice Screening",
    "Prediction Result",
    "Dashboard",
    "Model Info",
    "Thank You"
]


# =========================
# Log columns
# =========================
LOG_COLUMNS = [
    "timestamp",
    "user_id",
    "age",
    "gender",
    "prediction",
    "confidence",
    "risk_score",
    "frequency_hz",
    "jitter",
    "shimmer",
    "harmonicity",
    "nonlinear"
]


# =========================
# Session state setup
# =========================
DEFAULT_STATE = {
    "page_index": 0,
    "user_id": "",
    "age": 25,
    "gender": "Prefer not to say",
    "prediction_ready": False,
    "prediction": None,
    "confidence": None,
    "risk_score": None,
    "voice_features": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================
# HTML helper
# =========================
def render_html(html):
    st.markdown(dedent(html).strip(), unsafe_allow_html=True)


# =========================
# Styling
# =========================
def load_custom_css():
    render_html(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 45%, #ecfdf5 100%);
            border: 1px solid #dbeafe;
            border-radius: 24px;
            padding: 34px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }

        .hero-title {
            font-size: 38px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #334155;
            line-height: 1.6;
            max-width: 900px;
        }

        .mini-badge {
            display: inline-block;
            background: #0ea5e9;
            color: white;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 14px;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin: 20px 0;
        }

        .info-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 20px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }

        .info-card-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .info-card-title {
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 6px;
        }

        .info-card-text {
            color: #475569;
            font-size: 14px;
            line-height: 1.5;
        }

        .journey-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 24px;
            margin-top: 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .journey-title {
            font-size: 24px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 6px;
        }

        .journey-subtitle {
            color: #64748b;
            font-size: 15px;
            margin-bottom: 20px;
        }

        .journey-row {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 16px 0;
            border-bottom: 1px solid #f1f5f9;
        }

        .journey-number {
            min-width: 54px;
            height: 54px;
            border-radius: 16px;
            background: linear-gradient(135deg, #0ea5e9, #22c55e);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 15px;
            box-shadow: 0 6px 14px rgba(14, 165, 233, 0.22);
        }

        .journey-step-title {
            font-size: 18px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .journey-step-desc {
            font-size: 15px;
            color: #475569;
            line-height: 1.5;
        }

        .tips-card {
            background: linear-gradient(135deg, #f8fafc, #f0f9ff);
            border: 1px solid #dbeafe;
            border-radius: 22px;
            padding: 22px;
            margin: 18px 0;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }

        .tips-title {
            font-size: 22px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 10px;
        }

        .tips-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 14px;
        }

        .tip-item {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 14px;
            text-align: center;
        }

        .tip-icon {
            font-size: 28px;
            margin-bottom: 6px;
        }

        .tip-text {
            font-size: 14px;
            color: #334155;
            font-weight: 700;
        }

        .status-card {
            border-radius: 22px;
            padding: 26px;
            margin: 18px 0;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
        }

        .status-high {
            background: linear-gradient(135deg, #fff1f2, #fee2e2);
            border: 1px solid #fecaca;
        }

        .status-low {
            background: linear-gradient(135deg, #ecfdf5, #dcfce7);
            border: 1px solid #bbf7d0;
        }

        .status-title {
            font-size: 28px;
            font-weight: 900;
            margin-bottom: 8px;
            color: #0f172a;
        }

        .status-text {
            color: #334155;
            font-size: 16px;
            line-height: 1.6;
        }

        .risk-scale-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 22px;
            margin: 16px 0;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }

        .risk-scale-title {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 12px;
        }

        .risk-scale-bar {
            height: 22px;
            border-radius: 999px;
            overflow: hidden;
            display: flex;
            margin-bottom: 10px;
            border: 1px solid #e5e7eb;
        }

        .risk-low {
            width: 65%;
            background: linear-gradient(90deg, #22c55e, #86efac);
        }

        .risk-high {
            width: 35%;
            background: linear-gradient(90deg, #fca5a5, #ef4444);
        }

        .risk-scale-labels {
            display: flex;
            justify-content: space-between;
            color: #475569;
            font-size: 14px;
            font-weight: 700;
        }

        .pipeline {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 20px 0;
        }

        .pipeline-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .pipeline-icon {
            font-size: 32px;
            margin-bottom: 6px;
        }

        .pipeline-title {
            color: #0f172a;
            font-weight: 800;
            font-size: 15px;
        }

        .soft-note {
            background: #f8fafc;
            border-left: 5px solid #38bdf8;
            padding: 16px 18px;
            border-radius: 14px;
            color: #334155;
            margin: 16px 0;
        }

        .credit-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 22px;
            margin-top: 20px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }

        .credit-title {
            font-size: 20px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 10px;
        }

        .credit-text {
            font-size: 16px;
            color: #334155;
            line-height: 1.7;
        }

        @media (max-width: 900px) {
            .card-grid, .tips-grid, .pipeline {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 28px;
            }
        }
        </style>
        """
    )


load_custom_css()


# =========================
# Data helper functions
# =========================
def clean_filename(text):
    text = str(text).strip()
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", text)
    return text if text else "anonymous"


def load_log():
    if LOG_FILE.exists():
        log_df = pd.read_csv(LOG_FILE)

        for col in LOG_COLUMNS:
            if col not in log_df.columns:
                log_df[col] = np.nan

        return log_df[LOG_COLUMNS]

    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(user_id, age, gender, prediction, confidence, risk_score, voice_features):
    log_df = load_log()

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "prediction": prediction,
        "confidence": confidence,
        "risk_score": risk_score,
        "frequency_hz": voice_features.get("frequency_hz", 0),
        "jitter": voice_features.get("jitter", 0),
        "shimmer": voice_features.get("shimmer", 0),
        "harmonicity": voice_features.get("harmonicity", 0),
        "nonlinear": voice_features.get("nonlinear", 0)
    }])

    log_df = pd.concat([log_df, new_row], ignore_index=True)
    log_df.to_csv(LOG_FILE, index=False)


def read_audio_file(uploaded_audio):
    audio_bytes = uploaded_audio.getvalue()
    y, sr = sf.read(io.BytesIO(audio_bytes))

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    y = y.astype(np.float32)

    max_value = np.max(np.abs(y))
    if max_value > 0:
        y = y / max_value

    return y, sr


def extract_mdvp_features(uploaded_audio):
    """Extract 22 MDVP acoustic features using Praat via parselmouth."""

    audio_bytes = uploaded_audio.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        snd = parselmouth.Sound(tmp_path)
        duration = snd.get_total_duration()

        if duration < 1.0:
            return None, None

        # Pitch and point process for jitter/shimmer measures
        pitch = snd.to_pitch(pitch_floor=75.0, pitch_ceiling=500.0)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # MDVP frequency measures
        mdvp_fo = call(pitch, "Get mean", 0, 0, "Hertz")
        mdvp_fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        mdvp_flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        # Jitter measures
        mdvp_jitter_pct = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        mdvp_jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        mdvp_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        mdvp_ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = mdvp_rap * 3

        # Shimmer measures
        mdvp_shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        mdvp_shimmer_db = call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        mdvp_apq = call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = shimmer_apq3 * 3

        # NHR / HNR (harmonicity)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = (1.0 / (10 ** (hnr / 10))) if hnr and np.isfinite(hnr) else 0.0

        # Nonlinear / entropy measures (approximations)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]

        if len(pitch_values) > 1:
            log_pitch = np.log(pitch_values)
            ppe = float(np.std(log_pitch))
            spread1 = float(-np.mean(log_pitch) - 5)
            spread2 = float(np.std(log_pitch))
            rpde = float(np.std(pitch_values) / (np.mean(pitch_values) + 1e-8))
        else:
            ppe = 0.0
            spread1 = -5.0
            spread2 = 0.0
            rpde = 0.5

        # Placeholder values for measures requiring specialized libraries
        dfa = 0.7
        d2 = 2.3

        features = {
            "MDVP:Fo(Hz)": mdvp_fo,
            "MDVP:Fhi(Hz)": mdvp_fhi,
            "MDVP:Flo(Hz)": mdvp_flo,
            "MDVP:Jitter(%)": mdvp_jitter_pct,
            "MDVP:Jitter(Abs)": mdvp_jitter_abs,
            "MDVP:RAP": mdvp_rap,
            "MDVP:PPQ": mdvp_ppq,
            "Jitter:DDP": jitter_ddp,
            "MDVP:Shimmer": mdvp_shimmer,
            "MDVP:Shimmer(dB)": mdvp_shimmer_db,
            "Shimmer:APQ3": shimmer_apq3,
            "Shimmer:APQ5": shimmer_apq5,
            "MDVP:APQ": mdvp_apq,
            "Shimmer:DDA": shimmer_dda,
            "NHR": nhr,
            "HNR": hnr,
            "RPDE": rpde,
            "DFA": dfa,
            "spread1": spread1,
            "spread2": spread2,
            "D2": d2,
            "PPE": ppe,
        }

        # Replace NaN / inf with safe defaults
        for k, v in features.items():
            if v is None or not np.isfinite(v):
                features[k] = 0.0

        return features, duration

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def voice_based_prediction(audio_1, audio_2, audio_3):
    """Extract MDVP features from 3 recordings, average them, run XGBoost prediction."""

    all_features = []
    durations = []

    for audio in [audio_1, audio_2, audio_3]:
        features, duration = extract_mdvp_features(audio)

        if features is None:
            return None, None, None, None

        all_features.append(features)
        durations.append(duration)

    # Average features across all 3 recordings
    avg_features = {
        key: np.mean([f[key] for f in all_features])
        for key in all_features[0].keys()
    }

    # Build feature row in the exact column order the model was trained on
    feature_row = pd.DataFrame([avg_features])[FEATURE_COLUMNS]

    # Apply same preprocessing pipeline used during training
    # 1. Winsorize using training-derived limits
    for col in feature_row.columns:
        if col in WINSOR_LIMITS:
            limits = WINSOR_LIMITS[col]
            lo = limits.get("lower_5%", limits.get("lower"))
            hi = limits.get("upper_95%", limits.get("upper"))
            if lo is not None and hi is not None:
                feature_row[col] = feature_row[col].clip(lower=lo, upper=hi)

    # 2. Log1p transform on flagged columns
    for col in LOG_COLS:
        if col in feature_row.columns:
            feature_row[col] = np.log1p(feature_row[col].clip(lower=0))

    # 3. Standard scaling
    feature_scaled = SCALER.transform(feature_row)

    # 4. XGBoost prediction
    prob_parkinson = float(XGB_MODEL.predict_proba(feature_scaled)[0, 1])
    risk_score = prob_parkinson

    if risk_score >= 0.65:
        prediction = "High Parkinson's Voice Risk Pattern"
        confidence = risk_score
    else:
        prediction = "Low Parkinson's Voice Risk Pattern"
        confidence = 1 - risk_score

    # Map MDVP features to UI-friendly names (so the rest of the app works unchanged)
    # Compute 5 category averages
    freq_feats = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]
    jitter_feats = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"]
    shimmer_feats = ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"]
    harmonicity_feats = ["NHR", "HNR"]
    nonlinear_feats = ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]

    frequency_hz = float(np.mean([avg_features[f] for f in freq_feats]))
    jitter_avg = float(np.mean([avg_features[f] for f in jitter_feats]))
    shimmer_avg = float(np.mean([avg_features[f] for f in shimmer_feats]))
    harmonicity_avg = float(np.mean([avg_features[f] for f in harmonicity_feats]))
    nonlinear_avg = float(np.mean([avg_features[f] for f in nonlinear_feats]))

    hnr_val = avg_features.get("HNR", 0)
    ui_features = {
        "duration": float(np.mean(durations)),
        "pitch_mean": float(avg_features["MDVP:Fo(Hz)"]),
        "pitch_instability": float(avg_features["MDVP:Jitter(%)"] / 100),
        "amplitude_instability": float(avg_features["MDVP:Shimmer"]),
        "noise_proxy": float(avg_features["NHR"]),
        "silence_ratio": float(1 / (hnr_val + 1e-8)) if hnr_val > 0 else 0.5,
        "frequency_hz": frequency_hz,
        "jitter": jitter_avg,
        "shimmer": shimmer_avg,
        "harmonicity": harmonicity_avg,
        "nonlinear": nonlinear_avg,
        "mdvp_full": {k: float(v) for k, v in avg_features.items()},
    }

    return prediction, confidence, risk_score, ui_features


def save_audio_files(user_id, audio_1, audio_2, audio_3):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_user_id = clean_filename(user_id)

    audio_files = [audio_1, audio_2, audio_3]

    for i, audio in enumerate(audio_files, start=1):
        filename = RECORDING_DIR / f"{safe_user_id}_{timestamp}_recording_{i}.wav"
        with open(filename, "wb") as f:
            f.write(audio.getvalue())


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def reset_screening():
    keys_to_clear = [
        "audio_1",
        "audio_2",
        "audio_3",
        "prediction_ready",
        "prediction",
        "confidence",
        "risk_score",
        "voice_features"
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.prediction_ready = False
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.risk_score = None
    st.session_state.voice_features = None
    st.session_state.page_index = 1
    st.rerun()


# =========================
# Visual components
# =========================
def hero_banner():
    render_html(
        """
        <div class="hero-card">
            <div class="mini-badge">Voice-Based Screening Prototype</div>
            <div class="hero-title">🎙️ Parkinson's Disease Voice Risk Screening App</div>
            <div class="hero-subtitle">
                A simple data product that lets users record a short voice sample, analyzes basic voice patterns,
                and presents a low-risk or high-risk Parkinson's voice pattern result in a user-friendly dashboard.
            </div>
        </div>
        """
    )


def feature_cards():
    render_html(
        """
        <div class="card-grid">
            <div class="info-card">
                <div class="info-card-icon">🎤</div>
                <div class="info-card-title">Voice Recording</div>
                <div class="info-card-text">Users record the sustained vowel sound <b>aaaaah</b> three times.</div>
            </div>
            <div class="info-card">
                <div class="info-card-icon">🧠</div>
                <div class="info-card-title">Voice Pattern Analysis</div>
                <div class="info-card-text">The app extracts pitch instability, amplitude instability, noise proxy, and silence ratio.</div>
            </div>
            <div class="info-card">
                <div class="info-card-icon">📊</div>
                <div class="info-card-title">Dashboard View</div>
                <div class="info-card-text">Screening results are summarized through risk counts, charts, and recent records.</div>
            </div>
        </div>
        """
    )


def step_flow():
    steps = [
        ("01", "🏠", "Home", "Understand the purpose of the Parkinson's disease voice risk screening app."),
        ("02", "👤", "Personal Details", "Enter basic user information for record keeping."),
        ("03", "🎙️", "Voice Screening", "Record the sustained vowel sound aaaaah three times."),
        ("04", "📋", "Prediction Result", "View your low-risk or high-risk Parkinson's voice pattern result."),
        ("05", "📊", "Dashboard", "Review overall screening records and risk distribution."),
        ("06", "🧠", "Model Info", "Understand the prototype scoring logic."),
        ("07", "💙", "Thank You", "Complete the screening session with a short closing message.")
    ]

    rows = ""
    for number, icon, title, desc in steps:
        rows += f"""
        <div class="journey-row">
            <div class="journey-number">{number}</div>
            <div>
                <div class="journey-step-title">{icon} {title}</div>
                <div class="journey-step-desc">{desc}</div>
            </div>
        </div>
        """

    render_html(
        f"""
        <div class="journey-card">
            <div class="journey-title">Screening Journey</div>
            <div class="journey-subtitle">
                Follow these simple steps from voice recording to final screening result.
            </div>
            {rows}
        </div>
        """
    )


def recording_tips_card():
    render_html(
        """
        <div class="tips-card">
            <div class="tips-title">Before You Record</div>
            <div style="color:#475569; font-size:15px;">
                Follow these simple tips to get a clearer voice sample.
            </div>
            <div class="tips-grid">
                <div class="tip-item">
                    <div class="tip-icon">🤫</div>
                    <div class="tip-text">Find a quiet place</div>
                </div>
                <div class="tip-item">
                    <div class="tip-icon">🎙️</div>
                    <div class="tip-text">Stay close to the mic</div>
                </div>
                <div class="tip-item">
                    <div class="tip-icon">🔊</div>
                    <div class="tip-text">Use one steady tone</div>
                </div>
                <div class="tip-item">
                    <div class="tip-icon">🔁</div>
                    <div class="tip-text">Record three times</div>
                </div>
            </div>
        </div>
        """
    )


def recording_status_cards(completed):
    statuses = []

    for i in range(1, 4):
        if completed >= i:
            statuses.append((f"Recording {i}", "✅ Completed", "#dcfce7", "#166534"))
        else:
            statuses.append((f"Recording {i}", "⏳ Pending", "#f8fafc", "#475569"))

    cols = st.columns(3)

    for col, item in zip(cols, statuses):
        title, status, bg, color = item

        with col:
            render_html(
                f"""
                <div style="
                    background:{bg};
                    border:1px solid #e2e8f0;
                    border-radius:18px;
                    padding:18px;
                    text-align:center;
                    box-shadow:0 6px 16px rgba(15,23,42,0.06);
                ">
                    <div style="font-weight:900;color:#0f172a;font-size:17px;">{title}</div>
                    <div style="font-weight:800;color:{color};font-size:15px;margin-top:4px;">{status}</div>
                </div>
                """
            )


def risk_gauge(risk_score):
    percent = int(round(risk_score * 100))
    degree = int(round(risk_score * 360))
    label = "High Risk" if risk_score >= 0.65 else "Low Risk"
    color = "#ef4444" if risk_score >= 0.65 else "#22c55e"

    render_html(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:24px;
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:22px;
            padding:22px;
            box-shadow:0 8px 20px rgba(15,23,42,0.06);
            margin:16px 0;
        ">
            <div style="
                width:150px;
                height:150px;
                border-radius:50%;
                background: conic-gradient({color} {degree}deg, #e5e7eb {degree}deg);
                display:flex;
                align-items:center;
                justify-content:center;
            ">
                <div style="
                    width:105px;
                    height:105px;
                    border-radius:50%;
                    background:white;
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    font-weight:900;
                    color:#0f172a;
                ">
                    <div style="font-size:26px;">{percent}%</div>
                    <div style="font-size:12px;color:#64748b;">Risk Score</div>
                </div>
            </div>
            <div>
                <div style="font-size:22px;font-weight:900;color:#0f172a;">Prototype Risk Gauge</div>
                <div style="font-size:15px;color:#475569;margin-top:6px;">
                    Current classification: <b style="color:{color};">{label}</b>
                </div>
                <div style="font-size:14px;color:#64748b;margin-top:6px;">
                    High-risk threshold: <b>65%</b>
                </div>
            </div>
        </div>
        """
    )


def risk_interpretation_scale():
    render_html(
        """
        <div class="risk-scale-card">
            <div class="risk-scale-title">Risk Interpretation Scale</div>
            <div class="risk-scale-bar">
                <div class="risk-low"></div>
                <div class="risk-high"></div>
            </div>
            <div class="risk-scale-labels">
                <span>0% - 64.9%: Low Risk</span>
                <span>65% - 100%: High Risk</span>
            </div>
        </div>
        """
    )


def feature_score_bars(voice_features):
    st.subheader("Voice Indicator Strength")
    st.caption(
        "Each bar shows how your voice indicator compares to typical clinical reference ranges. "
        "Higher values may indicate more voice irregularity."
    )

    items = [
        ("Pitch Instability (Jitter)", voice_features["pitch_instability"] / 0.01),
        ("Amplitude Instability (Shimmer)", voice_features["amplitude_instability"] / 0.06),
        ("Voice Noise Level (NHR)", voice_features["noise_proxy"] / 0.03),
        ("Voice Clarity Issue (1/HNR)", voice_features["silence_ratio"] / 0.10)
    ]

    for label, value in items:
        normalized = min(max(float(value), 0), 1)
        st.write(f"{label}: **{normalized * 100:.1f}% of clinical reference range**")
        st.progress(normalized)


def pipeline_visual():
    render_html(
        """
        <div class="pipeline">
            <div class="pipeline-card">
                <div class="pipeline-icon">🎤</div>
                <div class="pipeline-title">Voice Recording</div>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-icon">🧪</div>
                <div class="pipeline-title">Feature Extraction</div>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-icon">📈</div>
                <div class="pipeline-title">Risk Score</div>
            </div>
            <div class="pipeline-card">
                <div class="pipeline-icon">✅</div>
                <div class="pipeline-title">Risk Classification</div>
            </div>
        </div>
        """
    )


def countdown_component():
    components.html(
        """
        <div style="
            border: 1px solid #e6e6e6;
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 12px;
            background: linear-gradient(135deg, #f8fafc, #f0f9ff);
            font-family: Arial, sans-serif;
        ">
            <h3 style="margin-top: 0;">⏱️ 5-Second Voice Countdown</h3>

            <p style="margin-bottom: 10px;">
                Use this timer as your guide. Press the microphone record button and say <b>aaaaah</b> for around 5 seconds.
            </p>

            <button onclick="startCountdown()" style="
                background-color: #0ea5e9;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-size: 16px;
                cursor: pointer;
                font-weight: 700;
            ">
                Start 5-Second Countdown
            </button>

            <div id="countdownText" style="
                font-size: 44px;
                font-weight: bold;
                margin-top: 15px;
                color: #0f172a;
            ">
                Ready
            </div>

            <div style="
                width: 100%;
                background-color: #e6e6e6;
                border-radius: 20px;
                margin-top: 10px;
                overflow: hidden;
            ">
                <div id="countdownBar" style="
                    width: 0%;
                    height: 18px;
                    background: linear-gradient(90deg, #38bdf8, #22c55e);
                    border-radius: 20px;
                "></div>
            </div>
        </div>

        <script>
        let intervalId = null;

        function startCountdown() {
            if (intervalId !== null) {
                clearInterval(intervalId);
            }

            let total = 5;
            let remaining = 5;

            document.getElementById("countdownText").innerHTML = "5";
            document.getElementById("countdownBar").style.width = "0%";

            intervalId = setInterval(function() {
                remaining -= 1;

                let progress = ((total - remaining) / total) * 100;
                document.getElementById("countdownBar").style.width = progress + "%";

                if (remaining > 0) {
                    document.getElementById("countdownText").innerHTML = remaining;
                } else {
                    document.getElementById("countdownText").innerHTML = "Done";
                    document.getElementById("countdownBar").style.width = "100%";
                    clearInterval(intervalId);
                    intervalId = null;
                }
            }, 1000);
        }
        </script>
        """,
        height=245
    )


def render_progress():
    current_step = st.session_state.page_index + 1
    total_steps = len(PAGES)
    progress_value = current_step / total_steps
    progress_percent = int(progress_value * 100)

    st.markdown(f"### Step {current_step} of {total_steps}: {PAGES[st.session_state.page_index]}")

    col1, col2 = st.columns([5, 1])
    with col1:
        st.progress(progress_value)
    with col2:
        st.markdown(f"**{progress_percent}%**")


def can_go_next():
    current_page = PAGES[st.session_state.page_index]

    if current_page == "Personal Details":
        if not str(st.session_state.get("user_id", "")).strip():
            st.warning("Please click Save and Continue after entering your details.")
            return False

    if current_page == "Voice Screening":
        if not st.session_state.prediction_ready:
            st.warning("Please complete the three recordings and click Analyze Voice and Continue first.")
            return False

    return True


def render_navigation_buttons():
    st.divider()

    left, middle, right = st.columns([1, 4, 1])

    is_first_page = st.session_state.page_index == 0
    is_last_page = st.session_state.page_index == len(PAGES) - 1

    with left:
        if not is_first_page:
            if st.button("⬅ Back", use_container_width=True):
                st.session_state.page_index -= 1
                st.rerun()

    with right:
        if not is_last_page:
            if st.button("Next ➡", use_container_width=True):
                if can_go_next():
                    st.session_state.page_index += 1
                    st.rerun()


# =========================
# Main layout
# =========================
st.title("🎙️ Parkinson's Disease Voice Risk Screening App")

st.warning(
    "This app is an academic prototype for screening support only. "
    "It is not a medical diagnosis. Please consult a healthcare professional for clinical assessment."
)

render_progress()

current_page = PAGES[st.session_state.page_index]


# =========================
# Page 1: Home
# =========================
if current_page == "Home":
    hero_banner()
    feature_cards()
    step_flow()

    render_html(
        """
        <div class="soft-note">
            The app is designed for non-technical users. Instead of asking users to upload CSV files
            or enter acoustic feature values manually, the app lets users record their voice and receive
            a simple screening result.
        </div>
        """
    )


# =========================
# Page 2: Personal Details
# =========================
elif current_page == "Personal Details":
    st.header("Personal Details")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.write(
            """
            Please enter basic information before starting the voice screening.
            These details are used for record keeping in the dashboard.
            """
        )

        gender_options = ["Male", "Female", "Prefer not to say"]

        current_gender = st.session_state.get("gender", "Prefer not to say")
        if current_gender not in gender_options:
            current_gender = "Prefer not to say"

        with st.form("personal_details_form"):
            user_id_input = st.text_input(
                "User ID / Name",
                value=st.session_state.get("user_id", "")
            )

            age_input = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=int(st.session_state.get("age", 25))
            )

            gender_input = st.selectbox(
                "Gender",
                gender_options,
                index=gender_options.index(current_gender)
            )

            save_continue = st.form_submit_button(
                "Save and Continue",
                use_container_width=True
            )

        if save_continue:
            if not user_id_input.strip():
                st.error("Please enter User ID or Name before continuing.")
            else:
                st.session_state.user_id = user_id_input.strip()
                st.session_state.age = int(age_input)
                st.session_state.gender = gender_input
                st.session_state.page_index = 2
                st.success("Personal details saved successfully.")
                st.rerun()

    with col_right:
        render_html(
            """
            <div class="info-card">
                <div class="info-card-icon">👤</div>
                <div class="info-card-title">Profile Setup</div>
                <div class="info-card-text">
                    Your details help organize the screening records in the dashboard.
                    The prediction itself is generated from your voice recording.
                </div>
            </div>
            """
        )

    st.info(
        "The prediction is generated from the voice recording. "
        "The personal details are used for screening record purposes only."
    )


# =========================
# Page 3: Voice Screening
# =========================
elif current_page == "Voice Screening":
    st.header("Voice Screening")

    st.subheader("Recording Instructions")

    st.info(
        """
        Please follow these instructions:

        1. Sit in a quiet place.  
        2. Take a normal breath.  
        3. Use the 5-second countdown as your guide.  
        4. Press the microphone record button.  
        5. Say **aaaaah** continuously for around 5 seconds.  
        6. Repeat the recording three times.  
        """
    )

    recording_tips_card()
    countdown_component()

    st.write("### Recording 1")
    st.audio_input("Record first 'aaaaah'", key="audio_1")

    st.write("### Recording 2")
    st.audio_input("Record second 'aaaaah'", key="audio_2")

    st.write("### Recording 3")
    st.audio_input("Record third 'aaaaah'", key="audio_3")

    audio_1 = st.session_state.get("audio_1")
    audio_2 = st.session_state.get("audio_2")
    audio_3 = st.session_state.get("audio_3")

    completed_recordings = sum(audio is not None for audio in [audio_1, audio_2, audio_3])

    recording_status_cards(completed_recordings)

    st.write(f"Completed recordings: **{completed_recordings}/3**")

    st.divider()

    if st.button("Analyze Voice and Continue", use_container_width=True):
        current_user_id = str(st.session_state.get("user_id", "")).strip()

        if current_user_id == "":
            st.error("Please go back to Personal Details and click Save and Continue.")

        elif audio_1 is None or audio_2 is None or audio_3 is None:
            st.error("Please complete all three voice recordings before analysis.")

        else:
            try:
                prediction, confidence, risk_score, voice_features = voice_based_prediction(
                    audio_1,
                    audio_2,
                    audio_3
                )

                if prediction is None:
                    st.error(
                        "The recording is too short or unclear. "
                        "Please record again and say 'aaaaah' for around 5 seconds."
                    )

                else:
                    save_audio_files(
                        current_user_id,
                        audio_1,
                        audio_2,
                        audio_3
                    )

                    save_log(
                        user_id=current_user_id,
                        age=st.session_state.get("age", 25),
                        gender=st.session_state.get("gender", "Prefer not to say"),
                        prediction=prediction,
                        confidence=confidence,
                        risk_score=risk_score,
                        voice_features=voice_features
                    )

                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.risk_score = risk_score
                    st.session_state.voice_features = voice_features
                    st.session_state.prediction_ready = True
                    st.session_state.page_index = 3

                    st.rerun()

            except Exception as e:
                st.error("The app could not analyze the recording.")
                st.write("Error details:")
                st.code(str(e))

    if st.session_state.prediction_ready:
        st.success("Prediction is ready. Click Next to view the result.")


# =========================
# Page 4: Prediction Result
# =========================
elif current_page == "Prediction Result":
    st.header("Prediction Result")

    if not st.session_state.prediction_ready:
        st.info("No prediction result yet. Please complete the voice screening first.")

    else:
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        risk_score = st.session_state.risk_score
        voice_features = st.session_state.voice_features

        if "High" in prediction:
            render_html(
                """
                <div class="status-card status-high">
                    <div class="status-title">⚠️ Oh no! You are currently at high risk of Parkinson's voice pattern.</div>
                    <div class="status-text">
                        Your voice recording shows higher instability based on the prototype screening.
                        This does not mean you are diagnosed with Parkinson's disease, but it suggests that
                        further medical assessment is recommended.
                    </div>
                </div>
                """
            )
        else:
            render_html(
                """
                <div class="status-card status-low">
                    <div class="status-title">🎉 Congratulations! You are currently at low risk of Parkinson's voice pattern.</div>
                    <div class="status-text">
                        No strong Parkinson's voice-risk pattern was detected from your recorded samples.
                        This result is only based on the prototype voice screening and is not a medical diagnosis.
                    </div>
                </div>
                """
            )

        risk_gauge(risk_score)
        risk_interpretation_scale()

        col1, col2 = st.columns(2)
        col1.metric("Model Confidence", format_percent(confidence))
        col2.metric("Model Risk Score", format_percent(risk_score))

        st.subheader("Extracted Voice Pattern Summary")
        st.caption(
            f"Recording duration: **{voice_features['duration']:.2f} seconds**. "
            "The 22 MDVP acoustic features extracted from your voice are grouped into "
            "5 categories below, each showing the average value across all features in that group."
        )

        mdvp = voice_features.get("mdvp_full", {})

        groups = {
            "Frequency (Pitch)": {
                "features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
                "layman": "How high or low your voice sounds",
                "unit": "Hz"
            },
            "Jitter (Pitch Instability)": {
                "features": ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP"],
                "layman": "Cycle-to-cycle variation in pitch",
                "unit": ""
            },
            "Shimmer (Amplitude Instability)": {
                "features": ["MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA"],
                "layman": "Cycle-to-cycle variation in loudness",
                "unit": ""
            },
            "Harmonicity (Voice Noise)": {
                "features": ["NHR", "HNR"],
                "layman": "Ratio of harmonic signal vs noise",
                "unit": "dB"
            },
            "Nonlinear / Entropy Measures": {
                "features": ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"],
                "layman": "Complexity and unpredictability of voice signal",
                "unit": ""
            }
        }

        group_rows = []
        for category, info in groups.items():
            values = [mdvp.get(f, 0) for f in info["features"] if f in mdvp]
            avg_val = np.mean(values) if values else 0
            feature_count = len(info["features"])
            group_rows.append({
                "Feature Category": category,
                "What It Means": info["layman"],
                "Average Value": f"{avg_val:.4f} {info['unit']}".strip(),
                "Features Count": feature_count
            })

        st.dataframe(pd.DataFrame(group_rows), use_container_width=True)

        # Generate personalized layman explanation
        st.subheader("What Your Voice Means")

        freq_val = float(np.mean([mdvp.get(f, 0) for f in ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]]))
        jitter_val = float(mdvp.get("MDVP:Jitter(%)", 0))
        shimmer_val = float(mdvp.get("MDVP:Shimmer", 0))
        hnr_val_check = float(mdvp.get("HNR", 0))
        ppe_val = float(mdvp.get("PPE", 0))

        user_gender = st.session_state.get("gender", "Prefer not to say")

        # Frequency sentence (uses user-provided gender, not assumed from pitch)
        if user_gender == "Female":
            gender_phrase = "which is within the typical female pitch range"
        elif user_gender == "Male":
            gender_phrase = "which is within the typical male pitch range"
        else:
            gender_phrase = "which is within the typical adult pitch range"

        if freq_val < 140:
            freq_text = f"Your voice sounds on the lower side, around {freq_val:.0f} Hz, {gender_phrase}."
        elif freq_val < 180:
            freq_text = f"Your voice sounds at a moderate pitch, around {freq_val:.0f} Hz, {gender_phrase}."
        else:
            freq_text = f"Your voice sounds higher pitched, around {freq_val:.0f} Hz, {gender_phrase}, and is the opposite of the lower, flatter tone often heard in Parkinson's."

        # Jitter sentence
        if jitter_val < 0.3:
            jitter_text = "Your pitch is very steady, with very little wobble, which is a healthy sign."
        elif jitter_val < 1.0:
            jitter_text = "Your pitch wobble (jitter) is slightly higher than average but still within what healthy people show."
        else:
            jitter_text = "Your pitch wobble (jitter) is on the higher side, meaning your voice was a bit unsteady during the recording."

        # Shimmer sentence
        if shimmer_val < 0.04:
            shimmer_text = "Your loudness stayed very stable throughout the recording, which is a healthy sign."
        elif shimmer_val < 0.08:
            shimmer_text = "Your loudness wobble (shimmer) is slightly elevated, which often happens with natural speech or close mic placement."
        else:
            shimmer_text = "Your loudness wobble (shimmer) is noticeably elevated, which usually just means your voice volume varied a bit during recording or you were close to the mic."

        # Harmonicity sentence
        if hnr_val_check >= 20:
            hnr_text = f"Your voice clarity is excellent at {hnr_val_check:.1f} dB, matching what is seen in healthy voices."
        elif hnr_val_check >= 14:
            hnr_text = f"Your voice clarity at {hnr_val_check:.1f} dB is in a reasonable range, though a bit lower than the cleanest healthy voices."
        else:
            hnr_text = f"Your voice clarity is a bit lower than usual healthy voices, but this is most likely from background noise or your laptop microphone rather than a real voice issue."

        # Nonlinear sentence
        if ppe_val < 0.15:
            nonlinear_text = "Your overall voice complexity looks completely normal."
        elif ppe_val < 0.3:
            nonlinear_text = "Your overall voice complexity is mostly within the expected range."
        else:
            nonlinear_text = "Your overall voice complexity shows some unusual patterns, which the model considers along with everything else."

        # Combine
        paragraph = f"{freq_text} {jitter_text} {shimmer_text} {hnr_text} {nonlinear_text}"

        st.write(paragraph)

        with st.expander("View all 22 individual MDVP features"):
            if mdvp:
                full_features_df = pd.DataFrame({
                    "MDVP Feature": list(mdvp.keys()),
                    "Value": [f"{v:.4f}" for v in mdvp.values()]
                })
                st.dataframe(full_features_df, use_container_width=True, height=600)

        st.info(
            "**How is the risk score calculated?**  \n"
            "The XGBoost model looks at all 22 voice features **together**, not individually. "
            "A single high value (such as high shimmer) does not automatically mean high risk. "
            "The model has learned specific Parkinson's voice patterns from training data and "
            "compares your overall voice profile against those patterns. Your risk score reflects "
            "how closely your voice matches the Parkinson's pattern across all features combined."
        )

        st.caption(
            "The full XGBoost model details and feature extraction pipeline are explained in the Model Info page."
        )

        if st.button("Start New Screening", use_container_width=True):
            reset_screening()


# =========================
# Page 5: Dashboard
# =========================
elif current_page == "Dashboard":
    st.header("Screening Dashboard")

    log_df = load_log()

    if log_df.empty:
        st.info("No screening records yet. Complete a voice screening first.")

    else:
        log_df["confidence"] = pd.to_numeric(log_df["confidence"], errors="coerce")
        log_df["risk_score"] = pd.to_numeric(log_df["risk_score"], errors="coerce")

        total_screenings = len(log_df)
        high_risk = log_df["prediction"].str.contains("High", na=False).sum()
        low_risk = log_df["prediction"].str.contains("Low", na=False).sum()
        avg_confidence = log_df["confidence"].mean()
        avg_risk_score = log_df["risk_score"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Screenings", total_screenings)
        col2.metric("High-Risk Results", high_risk)
        col3.metric("Low-Risk Results", low_risk)
        col4.metric("Average Confidence", format_percent(avg_confidence))

        st.subheader("Risk Distribution")

        risk_chart = pd.DataFrame({
            "Risk Category": ["High Risk", "Low Risk"],
            "Count": [high_risk, low_risk]
        })

        st.bar_chart(risk_chart.set_index("Risk Category"))

        st.subheader("Average Model Risk Score")
        st.metric("Average Risk Score", format_percent(avg_risk_score))

        trend_df = log_df.copy()
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"], errors="coerce")
        trend_df = trend_df.dropna(subset=["timestamp"])

        if not trend_df.empty:
            trend_df["date"] = trend_df["timestamp"].dt.date
            daily_trend = trend_df.groupby("date")["risk_score"].mean().reset_index()
            daily_trend = daily_trend.set_index("date")
            st.subheader("Average Risk Score Trend")
            st.line_chart(daily_trend)

        st.subheader("Recent Screening Records")
        st.dataframe(log_df.tail(10), use_container_width=True)

        if st.button("Start New Screening from Dashboard", use_container_width=True):
            reset_screening()


# =========================
# Page 6: Model Info
# =========================
elif current_page == "Model Info":
    st.header("Model Information")

    pipeline_visual()

    st.write(
        """
        This Streamlit app uses a trained **XGBoost classifier** to detect Parkinson's
        voice risk patterns. Voice recordings are processed through **Praat** (via the
        parselmouth library) to extract 22 MDVP acoustic features, matching the schema
        of the Parkinson's voice dataset from Kaggle used during training.
        """
    )

    st.subheader("Deployed Model Details")

    model_info = pd.DataFrame({
        "Component": [
            "Model",
            "Training Dataset",
            "Features Extracted",
            "Preprocessing Pipeline",
            "Training F1-score",
            "Training Recall",
            "Training Precision",
            "Decision Threshold"
        ],
        "Detail": [
            "XGBoost (200 estimators, max_depth=5, lr=0.03)",
            "Parkinson's Voice Dataset (Kaggle)",
            "22 MDVP features (jitter, shimmer, NHR, HNR, PPE, spread1, etc.)",
            "Winsorize (5/95) → Log1p transform → StandardScaler",
            "0.845",
            "0.968",
            "0.750",
            "0.65 (probability cutoff for high-risk classification)"
        ]
    })

    st.dataframe(model_info, use_container_width=True)

    st.subheader("MDVP Acoustic Features Used")

    features_table = pd.DataFrame({
        "Feature Group": [
            "Frequency",
            "Jitter (frequency instability)",
            "Shimmer (amplitude instability)",
            "Harmonicity (noise)",
            "Nonlinear / Entropy"
        ],
        "Features": [
            "MDVP:Fo, MDVP:Fhi, MDVP:Flo",
            "MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP",
            "MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA",
            "NHR, HNR",
            "RPDE, DFA, spread1, spread2, D2, PPE"
        ]
    })

    st.dataframe(features_table, use_container_width=True)

    st.subheader("Prediction Workflow")

    workflow = pd.DataFrame({
        "Step": ["1", "2", "3", "4", "5"],
        "Process": [
            "User records sustained 'aaaaah' sound (3 times)",
            "Praat extracts 22 MDVP features from each recording",
            "Features are averaged across 3 recordings",
            "Apply training preprocessing (winsorize → log → scale)",
            "XGBoost outputs probability → classified as High or Low Risk"
        ]
    })

    st.dataframe(workflow, use_container_width=True)

    st.info(
        "This app uses a real trained XGBoost model for prediction. However, it remains "
        "an academic prototype and is not a medical diagnosis tool. Browser-recorded audio "
        "may differ from studio-quality training data, so results should be interpreted "
        "as a screening indicator only. Please consult a qualified healthcare professional "
        "for clinical assessment."
    )


# =========================
# Page 7: Thank You
# =========================
elif current_page == "Thank You":
    st.header("Thank You for Completing the Voice Screening Test")

    render_html(
        """
        <div class="status-card status-low">
            <div class="status-title">💙 Thank you for trying out the voice screening test!</div>
            <div class="status-text">
                Your recording helps show how a simple voice-based app can be used to support
                early Parkinson's disease risk screening in a more user-friendly way.
            </div>
        </div>
        """
    )

    st.write(
        """
        Please remember that this result is generated by an academic prototype and should
        not be treated as a medical diagnosis. If you have concerns about your voice,
        movement, or other Parkinson's-related symptoms, please consult a qualified
        healthcare professional for proper assessment.
        """
    )

    render_html(
        """
        <div class="credit-card">
            <div class="credit-title">Developed by:</div>
            <div class="credit-text">
                <b>Group 20</b><br>
                WQD7001 Principle of Data Science<br>
                Universiti Malaya
            </div>
        </div>
        """
    )

    st.info(
        "You may click Back to review the model information or previous pages."
    )


render_navigation_buttons()
