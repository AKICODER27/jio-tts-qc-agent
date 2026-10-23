import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import wave
import base64

# ==========================================
# PAGE CONFIG & PROFESSIONAL THEMING
# ==========================================
st.set_page_config(
    page_title="Indic & English TTS Multi-Agent QC Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: #1e222d;
        border: 1px solid #2e364f;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    .agent-card {
        background-color: #1a1f2c;
        border-left: 5px solid #00d2ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.title("🎙️ Indic & English TTS Multi-Agent Quality Control Engine")
st.markdown("**Enterprise Autonomous Agent Swarm for Audio Signal Integrity, ASR Alignment, and Training Corpus Curation**")
st.caption("Developed for Jio Platforms Ltd. OJT Case Study | Multi-Agent Orchestration Framework")

st.divider()

# ==========================================
# AUDIO PREVIEW HELPER FUNCTION (BASE64)
# ==========================================
def get_synthetic_audio_bytes(duration=2.0, freq=440, sample_rate=22050):
    """Generates an in-memory WAV audio preview stream formatted as a base64 string for browser playback."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    wav_io.seek(0)
    
    # Encode to base64 so HTML5 audio renders reliably on Streamlit Cloud
    b64_encoded = base64.b64encode(wav_io.read()).decode()
    return f"data:audio/wav;base64,{b64_encoded}"

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Agent Swarm Parameters")

selected_lang = st.sidebar.selectbox(
    "Target Language",
    ["Hindi", "Marathi", "Gujarati", "English"]
)

st.sidebar.subheader("🎯 Quality Gates")
min_snr = st.sidebar.slider("Min Signal-to-Noise Ratio (SNR dB)", 10.0, 35.0, 20.0, step=0.5)
max_wer = st.sidebar.slider("Max Word Error Rate (WER %)", 5.0, 30.0, 15.0, step=1.0)
target_sr = st.sidebar.selectbox("Required Sample Rate (Hz)", [16000, 22050, 24000, 44100], index=2)

batch_size = st.sidebar.number_input("Dataset Batch Size (Chunks)", min_value=5, max_value=50, value=10)

st.sidebar.divider()
run_pipeline = st.sidebar.button("🚀 Execute Agent Orchestration", type="primary", use_container_width=True)

# ==========================================
# SYNTHETIC DATA GENERATOR
# ==========================================
def generate_synthetic_audio_batch(count, language):
    seed_value = sum(ord(char) for char in language)
    np.random.seed(seed_value)
    
    profiles = {
        "Hindi": {"mean_snr": 23.5, "mean_wer": 11.2, "clip_prob": 0.05},
        "Marathi": {"mean_snr": 18.2, "mean_wer": 16.5, "clip_prob": 0.12},
        "Gujarati": {"mean_snr": 21.0, "mean_wer": 13.8, "clip_prob": 0.08},
        "English": {"mean_snr": 26.0, "mean_wer": 8.1, "clip_prob": 0.02}
    }
    
    profile = profiles.get(language, {"mean_snr": 20.0, "mean_wer": 15.0, "clip_prob": 0.05})
    
    sample_texts = {
        "Hindi": [
            "भारतीय अंतरिक्ष अनुसंधान संगठन ने एक नया इतिहास रच दिया है।",
            "कृत्रिम बुद्धिमत्ता तकनीक हमारे जीवन को तेजी से बदल रही है।",
            "स्वच्छ भारत अभियान से देश के गांवों में स्वच्छता बढ़ी है।",
            "डिजिटल भुगतान प्रणाली ने वित्तीय लेनदेन को बेहद आसान बना दिया है।",
            "रिलायंस जियो भारतभर में उच्च गति की ब्रॉडबैंड सेवाएं प्रदान कर रहा है।"
        ],
        "Marathi": [
            "महाराष्ट्र ही संतांची आणि शूर पुरुषांची भूमी आहे.",
            "तंत्रज्ञानाच्या विकासामुळे शिक्षण क्षेत्रात क्रांती झाली आहे.",
            "पुणे हे महाराष्ट्राचे सांस्कृतिक आणि शैक्षणिक केंद्र मानले जाते.",
            "आम्ही मराठी भाषेमध्ये स्वयंचलित वाक् संश्लेषण प्रणाली विकसित करत आहोत.",
            "पर्यटकांना कोकणातील नैसर्गिक सौंदर्य और समुद्रकिनारे खूप आवडतात."
        ],
        "Gujarati": [
            "ગુજરાત તેની સમૃદ્ધ સાંસ્કૃતિક વારસો અને વેપારી માટે જાણીતું છે.",
            "આપણા દેશમાં તહેવારો ખૂબ જ ઉત્સાહથી ઉજવવામાં આવે છે.",
            "અમદાવાદ સાબરમતી નદીના કિનારે વસેલું એક ઐતિહાસિક શહેર છે.",
            "આર્ટિફિશિયલ ઇન્ટેલિજન્સ હવે વિવિધ ભાષાઓમાં ઝડપથી વિકસી રહી છે.",
            "ટેક્નોલોજીના ઉપયોગથી નાગરિકોના પ્રશ્નો સરળતાથી ઉકેલી શકાય છે."
        ],
        "English": [
            "Advanced speech synthesis models require high fidelity training datasets.",
            "Multi-agent autonomous systems enhance workflow efficiency and accuracy.",
            "Jio Platforms is transforming the digital ecosystem across the nation.",
            "Natural language processing bridges the gap between humans and machines.",
            "Real-time audio inspection ensures robust performance for voice applications."
        ]
    }
    
    texts = sample_texts.get(language, ["Sample audio transcript text"])
    
    data = []
    for i in range(1, count + 1):
        snr = round(np.random.normal(profile["mean_snr"], 4.5), 2)
        wer = round(np.random.normal(profile["mean_wer"], 5.0), 2)
        sr = int(np.random.choice([target_sr, target_sr, 16000 if target_sr != 16000 else 22050]))
        clipping = np.random.choice([False, True], p=[1 - profile["clip_prob"], profile["clip_prob"]])
        dur = round(np.random.uniform(2.5, 9.0), 2)
        ref_text = np.random.choice(texts)
        
        data.append({
            "chunk_id": f"CHUNK_{language[:3].upper()}_{i:03d}",
            "language": language,
            "duration_sec": dur,
            "sample_rate": sr,
            "snr_db": max(5.0, snr),
            "wer_percent": max(0.0, wer),
            "has_clipping": clipping,
            "reference_text": ref_text
        })
    return pd.DataFrame(data)

# ==========================================
# AGENT DEFINITIONS
# ==========================================
def run_audio_inspector_agent(df, min_snr_val, required_sr):
    logs, results = [], []
    for idx, row in df.iterrows():
        reasons = []
        if row['snr_db'] < min_snr_val:
            reasons.append(f"Low SNR ({row['snr_db']} dB < {min_snr_val} dB)")
        if row['sample_rate'] != required_sr:
            reasons.append(f"SR Mismatch ({row['sample_rate']} Hz != {required_sr} Hz)")
        if row['has_clipping']:
            reasons.append("Audio Clipping Detected")
            
        passed = len(reasons) == 0
        status_msg = "PASS" if passed else f"FAIL: {', '.join(reasons)}"
        results.append({"chunk_id": row["chunk_id"], "audio_qc_pass": passed, "audio_qc_notes": status_msg})
        logs.append(f"[{row['chunk_id']}] Signal Check ➔ {status_msg}")
    return pd.DataFrame(results), logs

def run_alignment_evaluator_agent(df, max_wer_val):
    logs, results = [], []
    for idx, row in df.iterrows():
        reasons = []
        if row['wer_percent'] > max_wer_val:
            reasons.append(f"High WER ({row['wer_percent']}% > {max_wer_val}%)")
            
        passed = len(reasons) == 0
        status_msg = "ALIGNED" if passed else f"MISALIGNED: {', '.join(reasons)}"
        results.append({"chunk_id": row["chunk_id"], "alignment_pass": passed, "alignment_notes": status_msg})
        logs.append(f"[{row['chunk_id']}] Phonetic Alignment ➔ {status_msg}")
    return pd.DataFrame(results), logs

def run_corpus_curator_agent(raw_df, audio_qc_df, align_df):
    logs = []
    merged = raw_df.merge(audio_qc_df, on="chunk_id").merge(align_df, on="chunk_id")
    final_status = []
    for idx, row in merged.iterrows():
        status = "APPROVED_FOR_TRAINING" if (row['audio_qc_pass'] and row['alignment_pass']) else "REJECTED"
        final_status.append(status)
        logs.append(f"[{row['chunk_id']}] Final Decision ➔ {status}")
    merged['final_status'] = final_status
    return merged, logs

# ==========================================
# DASHBOARD TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Architecture", "⚡ Agent Execution", "📈 Analytics & Export"])

raw_data = generate_synthetic_audio_batch(batch_size, selected_lang)

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader(f"📥 Input Audio Chunk Queue ({selected_lang})")
        st.dataframe(raw_data, use_container_width=True)
        
    with col_right:
        st.subheader("🤖 Autonomous Agent Swarm Design")
        st.markdown("""
        * **Agent 1: Signal Inspector**
          * Validates SNR (dB), sample rates, and audio clipping.
        * **Agent 2: Alignment Evaluator**
          * Computes Phonetic WER & Script alignment.
        * **Agent 3: Corpus Curator**
          * Consolidates multi-agent votes into clean training manifests.
        """)

with tab2:
    if run_pipeline:
        st.subheader("⚙️ Live Sequential Agent Execution")
        progress = st.progress(0)
        
        # Agent 1
        st.markdown("##### 1️⃣ Audio & Signal Inspector Agent")
        with st.spinner("Analyzing DSP features & noise floors..."):
            time.sleep(0.8)
            audio_results, logs1 = run_audio_inspector_agent(raw_data, min_snr, target_sr)
            progress.progress(33)
            with st.expander("View Agent 1 Logs", expanded=False):
                st.code("\n".join(logs1), language="text")

        # Agent 2
        st.markdown("##### 2️⃣ Phonetic Alignment Evaluator Agent")
        with st.spinner("Cross-verifying transcript alignments & computing WER..."):
            time.sleep(0.8)
            align_results, logs2 = run_alignment_evaluator_agent(raw_data, max_wer)
            progress.progress(66)
            with st.expander("View Agent 2 Logs", expanded=False):
                st.code("\n".join(logs2), language="text")

        # Agent 3
        st.markdown("##### 3️⃣ Dataset Corpus Curator Agent")
        with st.spinner("Applying final quality gates & exporting dataset manifest..."):
            time.sleep(0.8)
            final_manifest, logs3 = run_corpus_curator_agent(raw_data, audio_results, align_results)
            progress.progress(100)
            st.session_state['processed_manifest'] = final_manifest
            with st.expander("View Agent 3 Logs", expanded=False):
                st.code("\n".join(logs3), language="text")
                
        st.success("✅ Multi-Agent Orchestration Completed Successfully!")
    else:
        st.info("👈 Click **🚀 Execute Agent Orchestration** in the sidebar to start the pipeline.")

with tab3:
    if 'processed_manifest' in st.session_state:
        manifest = st.session_state['processed_manifest']
        
        st.subheader("📈 Curation Key Performance Indicators (KPIs)")
        total = len(manifest)
        approved = len(manifest[manifest['final_status'] == "APPROVED_FOR_TRAINING"])
        rejected = total - approved
        pass_rate = round((approved / total) * 100, 1)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Input Chunks", f"{total}")
        kpi2.metric("Approved Chunks", f"{approved}", delta=f"{pass_rate}% Pass Rate")
        kpi3.metric("Rejected Chunks", f"{rejected}", delta_color="inverse")
        kpi4.metric("Target Language", selected_lang)

        st.divider()

        # AUDIO CHUNK INSPECTOR & PLAYBACK PREVIEW
        st.subheader("🔊 Audio Chunk Verification & Playback")
        selected_chunk = st.selectbox("Select Chunk ID to Inspect & Listen", manifest['chunk_id'].tolist())
        
        row_data = manifest[manifest['chunk_id'] == selected_chunk].iloc[0]
        
        col_info, col_audio = st.columns([2, 1])
        with col_info:
            st.write(f"**Target Language:** {row_data['language']}")
            st.write(f"**Reference Text:** {row_data['reference_text']}")
            st.write(f"**SNR:** {row_data['snr_db']} dB | **WER:** {row_data['wer_percent']}% | **Status:** `{row_data['final_status']}`")
        
        with col_audio:
            tone_freq = 520 if row_data['final_status'] == "APPROVED_FOR_TRAINING" else 260
            audio_uri = get_synthetic_audio_bytes(duration=2.5, freq=tone_freq)
            st.audio(audio_uri, format='audio/wav')

        st.divider()
        
        # Audio Signal Visualizer
        st.subheader("🌊 Signal Waveform & Spectral Analytics")
        chart_data = pd.DataFrame(
            np.random.randn(50, 2),
            columns=['Audio Waveform Amplitude', 'Spectral Noise Floor']
        )
        st.line_chart(chart_data)

        st.subheader("📋 Output Dataset Manifest")
        st.dataframe(manifest[['chunk_id', 'language', 'snr_db', 'wer_percent', 'audio_qc_notes', 'alignment_notes', 'final_status']], use_container_width=True)

        csv_bytes = manifest.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Approved Corpus Manifest (.csv)",
            data=csv_bytes,
            file_name=f"jio_tts_{selected_lang.lower()}_approved_manifest.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.warning("Please run the pipeline under the **⚡ Agent Execution** tab first.")
