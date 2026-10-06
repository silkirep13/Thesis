# Byzantine & Eastern Music Analysis (MIR Thesis)

A web application for analysing recordings of **Byzantine chant, Greek folk, Cypriot and Arabic maqam** music.

Upload an audio file and choose its tradition. The system then answers three questions:

1. **Which mode is it?**: detects the echos / maqam / scale and its tonic, at cent-level resolution
2. **Which instrument is playing?**: names the dominant melodic instrument with a trained classifier
3. **What does it look like written down?**: produces a MusicXML score with quarter-tone accidentals

The point of the project is that these traditions use intervals that do not exist in 12-tone equal
temperament. Standard MIR tooling rounds them away; this pipeline preserves and notates them.

## Features

- **Mode detection** across 4 traditions (19 scale templates), with 36-bin chroma at 33.3 ¢ resolution
- **Tonic resolution** anchored on the finalis, so rotationally-equivalent modes are told apart
- **Microtonal analysis**: cent deviation from equal temperament (p10/p25/p50/p75/p90)
- **Instrument recognition**: 10 traditional instruments, restricted to the declared tradition
- **Score generation**: MusicXML with quarter-tone accidentals and exact cent annotations
- **Pulse measurement**: free-rhythm material is notated unmeasured rather than forced into a metre
- **Pitch contour**: CREPE pitch track over time against the detected tonic

### Supported traditions

| Tradition | Modes / Maqamat |
|---|---|
| Byzantine | Octoechos, 8 modes (genus is a property of each mode, not a separate template) |
| Greek Folk | Greek Minor (Minore), Greek Hijaz |
| Cypriot | Cypriot Pentachord |
| Arabic Maqam | Rast, Bayati, Hijaz, Nahawand, Saba, Kurd, Ajam, Jiharkah |

### Recognised instruments

| Tradition | Instruments |
|---|---|
| Greek | bouzouki, laouto, violin, clarinet, santouri, guitar |
| Cypriot | laouto, violin, pithkiavlin, guitar |
| Arabic | oud, ney, kanun, violin |
| Byzantine | none, the tradition is vocal only |

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI |
| Pitch tracking | CREPE (CNN, TensorFlow) |
| Source separation | Demucs htdemucs (PyTorch) |
| Audio features | librosa, SciPy |
| Instrument classifier | scikit-learn Random Forest on YAMNet embeddings |
| Score generation | music21 → MusicXML |
| Frontend | React 19, Vite |
| Visualisation | Recharts, OpenSheetMusicDisplay |

## Project Structure

```
Thesis/
├── backend/
│   ├── main.py                     # FastAPI app & Pydantic response model
│   ├── analysis.py                 # Mode/tonic detection, pitch tracking, note segmentation
│   ├── notation.py                 # Pulse estimation & MusicXML score generation
│   ├── test_scale_templates.py     # Self-check on the scale template library
│   ├── requirements.txt
│   └── instrument_classifier/
│       ├── prepare_dataset.py      # Audio → feature vectors
│       ├── train.py                # Random Forest training & evaluation
│       ├── predict.py              # Tradition-aware inference
│       ├── prepare_mixtures.py     # Multi-label experiment (see Limitations)
│       ├── train_multi.py          # Multi-label experiment (see Limitations)
│       ├── models/                 # Trained model (committed)
│       └── data/raw/<instrument>/  # Training audio (not committed)
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   └── components/
    │       ├── TraditionSelector.jsx
    │       ├── Uploader.jsx
    │       ├── ResultsPanel.jsx
    │       ├── ModeInfoCard.jsx
    │       ├── PitchChart.jsx
    │       └── ScoreView.jsx       # OpenSheetMusicDisplay score viewer
    └── package.json
```

## Running the Project

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs on `http://localhost:8000`.

On the first analysis the YAMNet weights (~15 MB) are downloaded from TensorFlow Hub and cached,
so the first request takes longer than later ones.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:5173`.

### Analysis time

Roughly **1:1 with the recording length** on CPU: a 3-minute upload takes about 3 minutes, most of
it spent in Demucs source separation. Input is capped at the last 4 minutes of audio.

## Training the Instrument Classifier

The trained model is committed, so this is only needed to retrain it, for example after adding
recordings.

**1. Arrange the audio.** One subfolder per instrument, containing solo recordings of that
instrument only:

```
backend/instrument_classifier/data/raw/
├── bouzouki/   bouzouki_1.mp3  bouzouki_2.mp3  ...
├── oud/        oud_1.mp3       ...
└── ...
```

**2. Extract features.** Run from inside `backend/instrument_classifier`:

```bash
python prepare_dataset.py --raw-dir data/raw --out-dir data/processed --features yamnet
```

`--features yamnet` uses 1024-dim pretrained embeddings (what the deployed model uses).
`--features handcrafted` uses 56 hand-designed descriptors instead, kept for comparison.

**3. Train and evaluate:**

```bash
python train.py --data-dir data/processed --test-videos-per-class 1 --out-model models/instrument_model.joblib
```

This prints per-instrument precision, recall and F1, plus two confusion matrices, one unrestricted
and one restricted to instruments valid for each tradition, and saves the model.

> **Note on the split.** Whole recordings are held out for testing, never individual clips. Clips
> from one recording share microphone, room and player, so a random per-clip split would let the
> model recognise the recording rather than the instrument, reporting accuracy that does not hold up
> on new material.

## Analysis Pipeline

1. **Demucs** separates the audio into 4 stems (vocals / bass / drums / other)
2. **HPSS** isolates harmonic content for chroma analysis
3. **36-bin chroma joint search** (12 tonics × N templates, ending-weighted) detects mode and tonic
4. **CREPE** extracts cent-accurate pitch from the dominant stem
5. **Refinement passes** resolve ambiguity:
   - finalis-based tonic resolution for rotationally equivalent modes
   - Byzantine kyrios/plagal disambiguation
   - Arabic neutral-interval disambiguation (Kurd/Bayati, Ajam/Rast)
6. **Note segmentation**: pitch frames grouped into notes, split on pitch change or onset
7. **Instrument classification**: YAMNet embeddings over sliding 4 s windows, soft-voted across the
   whole recording, masked to the declared tradition
8. **Pulse estimation**: onset-to-beat concentration decides metrical vs unmeasured notation
9. **Score generation**: music21 builds the score and exports MusicXML\
