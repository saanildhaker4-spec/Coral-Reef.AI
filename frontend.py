"""
Coral-Reef.AI - front end (Streamlit)

Run from the repo folder:
    pip install -r requirements.txt
    streamlit run frontend.py

Uses the team's reef_classifier.py directly (same features, same logistic
regression), trained the same way as
`python reef_classifier.py --train-csv metadata.csv --predict-csv unlabeled.csv`.
"""
import csv
import io
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.io import wavfile

import reef_classifier as rc

st.set_page_config(page_title="Coral-Reef.AI", page_icon="🪸", layout="wide")

ROOT = Path(__file__).parent
METADATA = ROOT / "metadata.csv"
PREDICTIONS = ROOT / "predictions.csv"
BUOY_IMAGE = ROOT / "assets" / "buoy.png"

# Same order reef_classifier.main() uses for the model's inputs
FEATURE_NAMES = [
    "rms_db", "centroid", "rolloff", "flatness", "spectral_entropy",
    "band_shrimp", "band_fish", "band_low", "band_high",
    "shrimp_fish_ratio", "aci", "zcr",
]

FEATURE_INFO = {
    "rms_db": "Overall loudness",
    "centroid": "Average pitch of the sound (centre of the spectrum)",
    "rolloff": "Frequency below which 85% of the energy sits",
    "flatness": "Hiss-like (flat) versus tonal (peaky) sound",
    "spectral_entropy": "How spread out the sound is across frequencies",
    "band_shrimp": "Energy at 2-10 kHz (snapping shrimp crackle)",
    "band_fish": "Energy at 100 Hz-2 kHz (fish calls, grunts)",
    "band_low": "Energy at 20-100 Hz (waves, boats, rumble)",
    "band_high": "Energy at 10-20 kHz",
    "shrimp_fish_ratio": "Shrimp energy divided by fish energy",
    "aci": "Acoustic Complexity Index: how much the sound changes over time",
    "zcr": "Zero-crossing rate: how often the waveform flips sign",
}

# ------------------------------------------------------------------
# Optional denoising hook. If a teammate adds denoise.py with
#   def denoise(x, sr) -> np.ndarray
# the app uses it. Otherwise noisereduce is used as a placeholder.
# ------------------------------------------------------------------
try:
    from denoise import denoise as _denoise
    DENOISE_SOURCE = "Team model (denoise.py)"
except ImportError:
    try:
        import noisereduce as nr

        def _denoise(x, sr):
            return nr.reduce_noise(y=x, sr=sr)

        DENOISE_SOURCE = "noisereduce (placeholder)"
    except ImportError:
        _denoise = None
        DENOISE_SOURCE = "Not available (pip install noisereduce)"


# ------------------------------ helpers ------------------------------
def save_temp_wav(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def write_wav_temp(x, sr) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    peak = np.max(np.abs(x)) or 1.0
    wavfile.write(path, sr, (x / peak * 0.9 * 32767).astype(np.int16))
    return path


def wav_bytes(x, sr) -> bytes:
    buf = io.BytesIO()
    peak = np.max(np.abs(x)) or 1.0
    wavfile.write(buf, sr, (x / peak * 0.9 * 32767).astype(np.int16))
    return buf.getvalue()


def safe_load(path):
    """rc.load_audio returns None on a bad file; handle that here."""
    out = rc.load_audio(path)
    return out if out is not None else (None, None)


def safe_features(path):
    try:
        return rc.extract_features(path)  # None if the file can't be read
    except Exception as e:  # unreadable or odd WAV format
        st.warning(f"Could not analyse {Path(path).name}: {e}")
        return None


def spectrogram_fig(x, sr, title):
    S, freqs = rc.stft(x, sr)
    S = S[:, ::max(1, S.shape[1] // 2000)]  # a plot only needs ~2000 time columns
    db = 20 * np.log10(S + 1e-9)
    db -= db.max()
    duration = len(x) / sr
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                   extent=[0, duration, freqs[0] / 1000, freqs[-1] / 1000])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def path_key(p):
    return os.path.normcase(os.path.normpath(p))


def to_matrix(feature_dicts):
    return np.array([[f[n] for n in FEATURE_NAMES] for f in feature_dicts], dtype=float)


@st.cache_data(show_spinner="Analysing labelled recordings...")
def load_dataset():
    """Labelled files from metadata.csv that exist on disk. Like reef_classifier's
    --feedback-csv, any verified_label in predictions.csv overrides or adds labels."""
    rows = {}
    if METADATA.exists():
        for row in rc.load_metadata(str(METADATA)):
            if row["file"]:
                rows[path_key(row["file"])] = row
    if PREDICTIONS.exists():
        for row in rc.load_metadata(str(PREDICTIONS)):
            verified = (row["verified_label"] or "").strip()
            if row["file"] and verified.casefold() in ("healthy", "degraded"):
                rows[path_key(row["file"])] = {**row, "label": verified}

    out = []
    for row in rows.values():
        label = (row["label"] or "").strip().casefold()
        if label not in ("healthy", "degraded") or not Path(row["file"]).exists():
            continue
        feat = safe_features(row["file"])
        if feat:
            out.append({**row, "label": label.capitalize(), "features": feat})
    return out


@st.cache_data(show_spinner="Training the reef model...")
def train_model(dataset):
    """Fit rc.LogisticRegression on all 12 features. Also returns leave-one-out
    probabilities: each file scored by a model trained without it."""
    X = to_matrix([d["features"] for d in dataset])
    y = np.array([d["label"] == "Healthy" for d in dataset], dtype=float)
    if len(set(y)) < 2:
        return None, None

    loo = np.full(len(X), np.nan)
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        if len(set(y[keep])) == 2:
            loo[i] = rc.LogisticRegression().fit(X[keep], y[keep]).predict_proba(X[i:i + 1])[0]
    return rc.LogisticRegression().fit(X, y), loo


def predict(model, feature_dicts):
    return model.predict_proba(to_matrix(feature_dicts))


def show_score(prob):
    c1, c2 = st.columns([1, 2])
    c1.metric("Chance of a healthy reef", f"{prob * 100:.0f}%")
    c2.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.5:
        c2.success("Sounds like the healthy reefs in our training data: a good source "
                   "for the playback library.")
    else:
        c2.warning("Sounds like the degraded reefs in our training data: possible "
                   "candidate site for restoration playback.")
    st.caption("Prediction from a logistic regression trained on a small labelled dataset. "
               "Not a validated ecological assessment.")


def feature_table(feat, model=None):
    """Feature values, plus how strongly the trained model leans on each one."""
    rows = []
    for i, n in enumerate(FEATURE_NAMES):
        row = {"Feature": n, "Value": f"{feat[n]:.4g}", "Meaning": FEATURE_INFO[n]}
        if model is not None:
            w = float(model.w[i])
            row["Model weight"] = round(w, 3)
            row["Higher value suggests"] = "Healthy" if w > 0 else "Degraded"
        rows.append(row)
    if model is not None:
        rows.sort(key=lambda r: -abs(r["Model weight"]))
    st.dataframe(rows, hide_index=True, width="stretch")


def score_chart(table):
    sites = sorted({t["Site"] for t in table})
    fig, ax = plt.subplots(figsize=(8, 3))
    order = sorted(range(len(table)), key=lambda k: table[k]["Score"])
    palette = {s: plt.cm.tab10(k) for k, s in enumerate(sites)}
    ax.barh([table[k]["File"] for k in order], [table[k]["Score"] for k in order],
            color=[palette[table[k]["Site"]] for k in order])
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability healthy")
    for s in sites:
        ax.barh([], [], color=palette[s], label=s)
    ax.legend(loc="lower right", fontsize=8)
    fig.set_figheight(max(3, 0.28 * len(table)))
    fig.tight_layout()
    st.pyplot(fig)


def site_metrics(table):
    sites = sorted({t["Site"] for t in table})
    for c, s in zip(st.columns(len(sites)), sites):
        ps = [t["Score"] for t in table if t["Site"] == s]
        c.metric(s, f"{np.mean(ps) * 100:.0f}% healthy", f"{len(ps)} recordings", delta_color="off")


# ------------------------------ layout ------------------------------
dataset = load_dataset()
model, loo = train_model(dataset) if dataset else (None, None)
n_healthy = sum(d["label"] == "Healthy" for d in dataset)

with st.sidebar:
    st.title("🪸 Coral-Reef.AI")
    st.caption("Listen. Clean. Restore.")
    st.markdown("**Status**")
    st.write(f"Labelled files found: {len(dataset)}")
    st.write(f"Denoising: {DENOISE_SOURCE}")
    if model is not None:
        st.write(f"Model: trained on {len(dataset)} recordings "
                 f"({n_healthy} Healthy, {len(dataset) - n_healthy} Degraded)")
    else:
        st.write("Model: not trained")
        st.info("Put the WAV files in a `data/` folder (paths from metadata.csv). "
                "The model needs readable recordings labelled both Healthy and Degraded.")
    st.divider()
    st.caption("SDG 14: Life Below Water")

st.title("Coral-Reef.AI")
st.subheader("Listening to reefs to help degraded ones recover")

tab_one, tab_sites, tab_buoy, tab_about = st.tabs(
    ["🎧 Analyse a recording", "📊 Reef sites", "🛟 Buoy concept", "ℹ️ About"])

# ------------------------------ analyse one ------------------------------
with tab_one:
    st.markdown("Upload a reef recording (.wav) to see its spectrogram, clean it, and score it.")
    up = st.file_uploader("Reef recording", type=["wav"], key="single")

    if up:
        raw_path = save_temp_wav(up.getvalue())
        sr, x = safe_load(raw_path)
        if sr is None:
            st.error("Couldn't read that WAV file. Try exporting it as 16-bit PCM WAV.")
        else:
            st.caption(f"{up.name} · {len(x) / sr:.1f} s after trimming silence · resampled to {sr} Hz")

            if st.session_state.get("file") != up.name:
                st.session_state.update(file=up.name, clean=None)

            if _denoise and st.button("✨ Clean with AI", type="primary"):
                with st.spinner("Removing boat noise and hiss..."):
                    st.session_state.clean = np.asarray(_denoise(x, sr), dtype=float)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Before")
                st.audio(wav_bytes(x, sr), format="audio/wav")
                st.pyplot(spectrogram_fig(x, sr, "Raw recording"))
            with col2:
                st.markdown("#### After")
                clean = st.session_state.clean
                if clean is not None:
                    b = wav_bytes(clean, sr)
                    st.audio(b, format="audio/wav")
                    st.pyplot(spectrogram_fig(clean, sr, "Cleaned recording"))
                    st.download_button("⬇️ Download cleaned audio", b,
                                       file_name=f"clean_{Path(up.name).stem}.wav")
                else:
                    st.info("Click **Clean with AI** to see the cleaned version.")

            st.divider()
            st.markdown("### Reef score")
            clean = st.session_state.clean
            target_path = write_wav_temp(clean, sr) if clean is not None else raw_path
            feat = safe_features(target_path)
            if feat:
                if model is not None:
                    show_score(float(predict(model, [feat])[0]))
                    st.caption(f"Model trained on {len(dataset)} labelled recordings from "
                               "metadata.csv. Most reliable for recordings made with the same "
                               "equipment and settings.")
                else:
                    st.info("Scoring needs the trained model, which needs the labelled WAV "
                            "files in `data/`.")
                feature_table(feat, model)

# ------------------------------ reef sites ------------------------------
with tab_sites:
    if model is None:
        st.info("No trained model yet. Put the labelled WAV files from metadata.csv in `data/`.")
        if PREDICTIONS.exists():
            st.markdown("Showing the last `predictions.csv` from the script:")
            with open(PREDICTIONS, newline="") as f:
                st.dataframe(list(csv.DictReader(f)), hide_index=True, width="stretch")
    else:
        st.markdown("#### How well does the model do?")
        st.markdown("Each training recording is scored by a model trained on **all the "
                    "others** (leave-one-out), so these are honest guesses, not memorised answers.")
        table = [{"File": d["filename"], "Site": d["site"], "True label": d["label"],
                  "Predicted": "Healthy" if p >= 0.5 else "Degraded", "Score": round(float(p), 3)}
                 for d, p in zip(dataset, loo) if not np.isnan(p)]
        if table:
            correct = sum(t["True label"] == t["Predicted"] for t in table)
            st.metric("Leave-one-out accuracy", f"{correct / len(table) * 100:.0f}%",
                      f"{correct} of {len(table)} correct", delta_color="off")
            site_metrics(table)
            score_chart(table)
            st.dataframe(table, hide_index=True, width="stretch")

        st.divider()
        st.markdown("#### Classify new recordings")
        ups = st.file_uploader("Upload WAV files to classify", type=["wav"],
                               accept_multiple_files=True, key="batch")
        items = []
        for u in ups or []:
            feat = safe_features(save_temp_wav(u.getvalue()))
            if feat:
                items.append({"filename": u.name, "site": "Uploaded", "features": feat})

        if items:
            probs = predict(model, [i["features"] for i in items])
            new = [{"File": i["filename"], "Site": i["site"],
                    "Predicted": "Healthy" if p >= 0.5 else "Degraded", "Score": round(float(p), 3)}
                   for i, p in zip(items, probs)]
            score_chart(new)
            st.dataframe(new, hide_index=True, width="stretch")

            # Same columns as reef_classifier.py's supervised output
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["file_path", "filename", "site", "predicted_label",
                        "probability_healthy", "verified_label"])
            for t in new:
                w.writerow([f"data/{t['File']}", t["File"], t["Site"], t["Predicted"],
                            f"{t['Score']:.3f}", ""])
            st.download_button("⬇️ Download predictions.csv", buf.getvalue(), file_name="predictions.csv")
            st.caption("To teach the model: copy the WAVs into `data/`, fill in `verified_label` "
                       "(Healthy or Degraded), and save the CSV as `predictions.csv` in the app "
                       "folder. The app retrains on them next time it starts.")

# ------------------------------ buoy ------------------------------
with tab_buoy:
    st.markdown("### Concept design: Coral-Reef.AI buoy")
    if BUOY_IMAGE.exists():
        st.image(str(BUOY_IMAGE), width="stretch")
    else:
        st.info("Add your Tinkercad screenshot as `assets/buoy.png` to show it here.")

    m1, m2 = st.columns(2)
    m1.markdown("**🌞 Listen mode (day)**\n\nRecords the reef and tracks its score over time.")
    m2.markdown("**🌙 Restore mode (night)**\n\nPlays cleaned healthy-reef sound, when many "
                "fish and coral larvae settle.")

    st.markdown("#### Estimated cost per unit (approximate)")
    st.dataframe({
        "Part": ["Hydrophone recorder", "Underwater speaker", "Raspberry Pi / ESP32", "Amplifier",
                 "Solar panel (~20 W)", "Battery (12 V LiFePO4)", "Housing, buoy, anchor"],
        "Role": ["Listens to the reef", "Plays cleaned sound", "Runs schedule, stores audio",
                 "Powers the speaker", "Charges battery", "Runs at night", "Floats and stays dry"],
        "Cost (₹)": ["10,000-15,000", "20,000-1,00,000+", "500-2,500", "300-800",
                     "1,500-2,500", "5,000-10,000", "5,000-15,000"],
    }, hide_index=True, width="stretch")

# ------------------------------ about ------------------------------
with tab_about:
    st.markdown("""
### The problem
Mass bleaching has hit reefs worldwide. Healthy reefs are noisy; degraded reefs go quiet,
and fish and coral larvae use sound to find a home.

### What we do differently
Research trials play a fixed recording and count fish later. Our buoy **listens, plays,
measures and adapts**: it tracks the reef's soundscape score and changes what it plays and
when. It's low-cost and autonomous, so restoration teams can use it across many sites.

### How the score works
We extract 12 bioacoustic features from each recording (loudness, pitch, snapping-shrimp
and fish frequency bands, acoustic complexity and more). A logistic regression model learns
from recordings labelled Healthy or Degraded which combination of features tells them apart,
then gives new recordings a probability of being healthy. Experts can correct its predictions,
and the model retrains on those corrections.

### Our honest position
Sound doesn't cure bleaching. It can help reefs recover faster by attracting fish and coral
larvae once conditions allow. The model is trained on only a few reefs, and we aim to
validate it with marine biologists.
""")
