# Indic & English TTS Multi-Agent Quality Control Engine

An autonomous multi-agent quality control and dataset curation engine built in Python and Streamlit for Jio Platforms Limited (OJT Project). This application automates acoustic validation, speech signal filtering, and ASR phonetic alignment across Hindi, Marathi, Gujarati, and English speech datasets.

---

## Project Overview
Building production-grade Text-to-Speech (TTS) models requires ultra-clean audio data paired with accurate transcriptions. Manual inspection of thousands of audio clips is time-consuming and error-prone. 

This project solves that challenge by deploying an autonomous 3-stage deterministic multi-agent pipeline that programmatically inspects raw audio metadata, enforces quality thresholds (SNR, WER, clipping), and generates structured training manifests ready for speech model training.

---

## Multi-Agent Architecture

### 1. Agent 1: Audio Signal Inspector
* Evaluates Signal-to-Noise Ratio (SNR >= 20 dB).
* Checks sampling rates (16 kHz / 22.05 kHz).
* Detects digital amplitude clipping anomalies.

### 2. Agent 2: Alignment Evaluator
* Computes Word Error Rate (WER <= 15%) using IndicWhisper transcript matching.
* Ensures strict phonetic alignment between speech audio and native target text scripts.

### 3. Agent 3: Corpus Curator
* Aggregates verification state votes from Agent 1 and Agent 2.
* Filters out failing clips and exports production-ready CSV manifests.

---

## Supported Languages & Native Scripts
* **Hindi (हिंदी)** — e.g., "भारतीय अंतरिक्ष अनुसंधान संगठन ने एक नया इतिहास रच दिया है।"
* **Marathi (मराठी)** — e.g., "महाराष्ट्र ही संतांची आणि शूर पुरुषांची भूमी आहे।"
* **Gujarati (ગુજરાતી)** — e.g., "ગુજરાત તેની સમૃદ્ધ સાંસ્કૃતિક વારસો અને વેપારી માટે જાણીતું છે।"
* **English** — e.g., "Advanced speech synthesis models require high fidelity training datasets."

---

## Tech Stack
* **Language:** Python
* **UI Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **ASR Model:** IndicWhisper

---

## Quickstart & Local Setup

### Step 1: Clone the repository
git clone https://github.com/AKICODER27/jio-tts-qc-agent.git

### Step 2: Navigate into folder and install dependencies
cd jio-tts-qc-agent
pip install -r requirements.txt

### Step 3: Run the Streamlit Application
streamlit run app.py

---

## Key Dashboard Features
* **Interactive Quality Gates:** Customize SNR and WER thresholds live in the sidebar.
* **Sequential Agent Execution:** Real-time state hand-offs and live step-by-step agent execution logs.
* **Analytics & Metrics:** KPI summary cards, pass/rejection rate breakdown, and interactive waveform displays.
* **Manifest Export:** Download filtered, high-quality audio training manifests in CSV format with a single click.
