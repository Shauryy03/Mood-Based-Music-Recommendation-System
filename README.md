# 🎭 Moodify — Mood-Based Music Recommendation System

Moodify detects your emotion from a photo, your webcam, or your voice, then builds a custom playlist on your Spotify account that matches your mood.

## How it works

You give Moodify an input — a webcam shot, an uploaded photo, or a short voice recording — and it runs emotion detection on it. Once it has a mood (happy, sad, angry, surprised, fearful, disgusted, neutral, or calm), it pulls a matching set of tracks, searches for them on Spotify, and creates a new public playlist on your account with the results.

## Features

- **Three input methods**: webcam capture, image upload, or voice recording
- **Facial emotion detection** via [DeepFace](https://github.com/serengil/deepface), with a custom-trained CNN (`train_fer.py` / `best_fer.h5`) as an alternative pipeline using MediaPipe for face detection and alignment
- **Voice emotion detection**: a simple prosodic/acoustic analysis using `librosa` in `app.py`, or a more advanced ensemble of a pretrained Wav2Vec2 transformer plus prosodic analysis in `app2.py`
- **Automatic Spotify playlist creation** on the user's own account via the Spotify Web API (through `spotipy`)
- **Two app versions**:
  - `app.py` — a simpler version with the core webcam/upload/voice + playlist flow
  - `app2.py` — an enhanced version with Spotify connect/disconnect UI, tabs, custom styling, region-aware track search, and the Wav2Vec2-based voice model

## Tech stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| Face emotion detection | DeepFace, OpenCV, MediaPipe, TensorFlow/Keras (custom FER model) |
| Voice emotion detection | librosa, PyTorch, torchaudio, Hugging Face Transformers (Wav2Vec2) |
| Music | Spotipy (Spotify Web API) |
| Data/ML utilities | NumPy, pandas, scikit-learn |

## Project structure

```
.
├── app.py              # Main Streamlit app (DeepFace + librosa prosodic voice detection)
├── app2.py              # Enhanced Streamlit app (Wav2Vec2 voice model, Spotify session UI)
├── train_fer.py         # Trains a CNN emotion classifier on the FER2013 dataset
├── model_predict.py     # Standalone face detection/alignment + custom FER model inference
├── load.py               # Helper script for loading/testing the trained model
├── demo.py               # Standalone demo script for emotion detection
├── best_fer.h5           # Trained facial emotion recognition model weights
├── requirements.txt      # Python dependencies
├── apt.txt               # System-level packages (for deployment, e.g. Streamlit Cloud)
└── .devcontainer/         # Dev container configuration
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AdityaGupta23-git/Mood-Based-Music-Recommendation-System.git
cd Mood-Based-Music-Recommendation-System
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `app2.py` additionally requires `torch`, `torchaudio`, and `transformers` for the Wav2Vec2 voice model, which aren't currently listed in `requirements.txt`. Install them separately if you plan to run `app2.py`:
> ```bash
> pip install torch torchaudio transformers
> ```

If deploying somewhere that needs system-level packages (e.g. Streamlit Community Cloud), the packages listed in `apt.txt` will be installed automatically.

### 3. Create a Spotify Developer app

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) and create a new app.
2. Note your **Client ID** and **Client Secret**.
3. Add a redirect URI matching what's used in the code (e.g. `http://127.0.0.1:8888` for `app.py`, or `http://127.0.0.1:8888/callback` for `app2.py`).

### 4. Configure credentials securely

⚠️ **Important:** the current versions of `app.py` and `app2.py` have Spotify credentials hardcoded directly in the source. Don't commit real credentials to a public repo. Instead, set them as environment variables and update the code to read from `os.environ`, for example:

```bash
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
export SPOTIFY_REDIRECT_URI="http://127.0.0.1:8888"
```

```python
import os

CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
REDIRECT_URI = os.environ["SPOTIFY_REDIRECT_URI"]
```

Or use a `.env` file with `python-dotenv` and make sure `.env` is in `.gitignore`. If you've already pushed real credentials to GitHub, rotate them in the Spotify Dashboard — old commits still contain them even after you remove them from the latest version.

### 5. Run the app

```bash
streamlit run app.py
```

or, for the enhanced version:

```bash
streamlit run app2.py
```

The app will open in your browser. On first run, you'll be redirected to Spotify to authorize the app.

## Usage

1. Choose an input method: webcam, image upload, or voice recording.
2. Wait for emotion detection to finish — the detected mood is shown with an emoji and color.
3. Click **Create Playlist** to generate a 50-track playlist on your Spotify account matching the detected mood.
4. Open the playlist directly in Spotify via the link provided.

## Training the facial emotion model (optional)

`best_fer.h5` is already included as a pretrained model, but if you want to retrain it:

1. Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` in the project root.
2. Run:
   ```bash
   python train_fer.py
   ```
   This trains a CNN on 48x48 grayscale face images across 7 emotion classes, with data augmentation and early stopping, and saves the best-performing weights.
3. `model_predict.py` provides face detection/alignment (via MediaPipe) and inference utilities for using this custom model as an alternative to DeepFace.

## Limitations

- Emotion detection from a single image/frame is inherently approximate; lighting, face angle, and occlusion all affect accuracy.
- Voice emotion detection without a model file shipped (`app.py`'s prosodic heuristic, or `app2.py`'s lightweight fallback) is a simplified approximation, not a robust classifier.
- Spotify playlist creation depends on track availability and matching by name; some songs in the curated lists may not be found exactly.

## License

No license file is currently included in this repository. Consider adding one (e.g. MIT) if you want to clarify how others can use this code.

## Author

Created by **Aditya Gupta**
