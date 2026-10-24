"""
Coral-Reef.AI - front end (Streamlit)

Run from the repo folder:
    pip install -r requirements.txt
    streamlit run app.py

Uses the team's reef_classifier.py directly (same features, same
acoustic-richness scoring), so results match `python reef_classifier.py --auto-label`.
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

FEATURE_NAMES = [
    "rms_db", "centroid", "rolloff", "flatness", "spectral_entropy",
    "band_shrimp", "band_fish", "band_low", "band_high",
    "shrimp_fish_ratio", "aci", "zcr",
]
# Same four features reef_classifier.py uses for its richness score
RICHNESS = ["aci", "spectral_entropy", "band_shrimp", "band_fish"]

FEATURE_INFO = {
    "aci": "Acoustic Complexity Index: how much the sound changes over time",
    "spectral_entropy": "How spread out the sound is across frequencies",
    "band_shrimp": "Energy at 2-10 kHz (snapping shrimp crackle)",
    "band_fish": "Energy at 100 Hz-2 kHz (fish calls, grunts)",
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
        if rc.load_audio(path) is None:
            return None
        return rc.extract_features(path)
    except Exception as e:  # unreadable or odd WAV format
        st.warning(f"Could not analyse {Path(path).name}: {e}")
        return None


def spectrogram_fig(x, sr, title):
    S, freqs = rc.stft(x, sr)
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


def richness_scores(feature_dicts):
    """Exactly the scoring in reef_classifier.main(): z-score the four richness
    features across the batch, average them, threshold at the median."""
    X = np.array([[f[n] for n in RICHNESS] for f in feature_dicts], dtype=float)
    means, stds = X.mean(axis=0), X.std(axis=0)
    stds[stds == 0] = 1.0
    score = ((X - means) / stds).mean(axis=1)
    thr = float(np.median(score))
    prob = 1.0 / (1.0 + np.exp(-np.clip(score - thr, -50, 50)))
    return score, prob, score >= thr, (means, stds, thr)


def score_against_reference(feat, ref):
    means, stds, thr = ref
    v = np.array([feat[n] for n in RICHNESS], dtype=float)
    s = float(((v - means) / stds).mean())
    return float(1.0 / (1.0 + np.exp(-np.clip(s - thr, -50, 50))))


@st.cache_data(show_spinner="Analysing dataset recordings...")
def load_dataset():
    """Features for every file in metadata.csv that exists on disk."""
    if not METADATA.exists():
        return []
    out = []
    for row in rc.load_metadata(str(METADATA)):
        p = row["file"] and (ROOT / row["file"])
        if p and p.exists():
            feat = safe_features(str(p))
            if feat:
                out.append({**row, "path": str(p), "features": feat})
    return out


def show_score(prob):
    c1, c2 = st.columns([1, 2])
    c1.metric("Acoustic richness", f"{prob * 100:.0f} / 100")
    c2.progress(min(max(prob, 0.0), 1.0))
    if prob >= 0.5:
        c2.success("Rich soundscape: likely a good source for the playback library.")
    else:
        c2.warning("Quiet soundscape: possible candidate site for restoration playback.")
    st.caption("Exploratory score from acoustic-richness features, not a trained or "
               "validated ecological assessment.")


def feature_table(feat):
    rows = [{"Feature": n, "Value": f"{feat[n]:.4g}", "Meaning": FEATURE_INFO[n]} for n in RICHNESS]
    st.dataframe(rows, hide_index=True, width="stretch")
    with st.expander("All 12 features"):
        st.dataframe([{"Feature": n, "Value": f"{feat[n]:.4g}"} for n in FEATURE_NAMES],
                     hide_index=True, width="stretch")


# ------------------------------ layout ------------------------------
dataset = load_dataset()
ref = richness_scores([d["features"] for d in dataset])[3] if len(dataset) >= 2 else None

with st.sidebar:
    st.title("🪸 Coral-Reef.AI")
    st.caption("Listen. Clean. Restore.")
    st.markdown("**Status**")
    st.write(f"Dataset files found: {len(dataset)}")
    st.write(f"Denoising: {DENOISE_SOURCE}")
    st.write("Scoring: reef_classifier.py")
    if not dataset:
        st.info("Put the WAV files in a `data/` folder (paths from metadata.csv) "
                "to enable the dataset view and single-file scoring.")
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
                if ref is not None:
                    show_score(score_against_reference(feat, ref))
                    st.caption(f"Compared against {len(dataset)} recordings in metadata.csv.")
                else:
                    st.info("The score is relative to other recordings, so it needs the "
                            "dataset in `data/`. Meanwhile, use the **Reef sites** tab to "
                            "upload several files and compare them.")
                feature_table(feat)

# ------------------------------ reef sites ------------------------------
with tab_sites:
    st.markdown("Scores every recording in a batch, the same way as "
                "`python reef_classifier.py --auto-label`.")

    if dataset:
        items = [{"filename": d["filename"], "site": d["site"], "features": d["features"]}
                 for d in dataset]
        st.caption(f"Using {len(items)} recordings from metadata.csv.")
    else:
        ups = st.file_uploader("No dataset found. Upload 2 or more WAV files to compare",
                               type=["wav"], accept_multiple_files=True, key="batch")
        items = []
        for u in ups or []:
            feat = safe_features(save_temp_wav(u.getvalue()))
            if feat:
                items.append({"filename": u.name, "site": "Uploaded", "features": feat})

    if len(items) >= 2:
        score, prob, healthy, _ = richness_scores([i["features"] for i in items])
        table = [{"File": i["filename"], "Site": i["site"],
                  "Label": "Rich" if h else "Quiet", "Score": round(float(p), 3)}
                 for i, p, h in zip(items, prob, healthy)]

        sites = sorted({i["site"] for i in items})
        cols = st.columns(len(sites))
        for c, s in zip(cols, sites):
            ps = [t["Score"] for t in table if t["Site"] == s]
            c.metric(s, f"{np.mean(ps) * 100:.0f} / 100", f"{len(ps)} recordings", delta_color="off")

        fig, ax = plt.subplots(figsize=(8, 3))
        order = sorted(range(len(table)), key=lambda k: table[k]["Score"])
        palette = {s: plt.cm.tab10(k) for k, s in enumerate(sites)}
        ax.barh([table[k]["File"] for k in order], [table[k]["Score"] for k in order],
                color=[palette[table[k]["Site"]] for k in order])
        ax.axvline(0.5, color="grey", ls="--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Acoustic richness score")
        for s in sites:
            ax.barh([], [], color=palette[s], label=s)
        ax.legend(loc="lower right", fontsize=8)
        fig.set_figheight(max(3, 0.28 * len(table)))
        fig.tight_layout()
        st.pyplot(fig)

        st.dataframe(table, hide_index=True, width="stretch")

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["filename", "predicted_label", "probability_healthy"])
        for t, h in zip(table, healthy):
            w.writerow([t["File"], "Healthy" if h else "Degraded", f"{t['Score']:.3f}"])
        st.download_button("⬇️ Download predictions.csv", buf.getvalue(), file_name="predictions.csv")
        st.caption("Labels split the batch at its median, so about half are always 'Rich'. "
                   "Exploratory only.")
    elif PREDICTIONS.exists():
        st.markdown("Showing the last `predictions.csv` from the script:")
        with open(PREDICTIONS, newline="") as f:
            st.dataframe(list(csv.DictReader(f)), hide_index=True, width="stretch")

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
We extract 12 bioacoustic features from each recording. Four of them (acoustic complexity,
spectral entropy, snapping-shrimp band and fish band) are combined into an acoustic-richness
score, compared across recordings.

### Our honest position
Sound doesn't cure bleaching. It can help reefs recover faster by attracting fish and coral
larvae once conditions allow. The score is exploratory, and we aim to validate it with
marine biologists.
""")