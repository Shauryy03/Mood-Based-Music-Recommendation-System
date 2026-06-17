import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import librosa
import soundfile as sf
import tempfile

# 🎵 Spotify Setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="2ab9da702ae44024a7e2c62dd03fad98",
    client_secret="cf0558bbab3744b1ada28aa1a7826abe",
    redirect_uri="http://127.0.0.1:8888",
    scope="playlist-modify-public,playlist-read-private,user-read-private"
))

# 🎭 Facial Emotion Detection
def detect_emotion_from_image(img):
    try:
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(img, actions=['emotion'], detector_backend='retinaface', enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        return f"Error: {str(e)}"

# 🎙️ Voice Emotion Detection
def detect_emotion_from_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        pitch = librosa.yin(y, fmin=50, fmax=300)
        energy = np.mean(librosa.feature.rms(y=y))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        if energy > 0.05 and contrast > 20:
            return "happy"
        elif pitch.mean() < 100 and energy < 0.03:
            return "sad"
        elif contrast > 25 and pitch.mean() > 150:
            return "angry"
        else:
            return "neutral"
    except Exception as e:
        return f"Error: {str(e)}"

# 🔍 Search Playlist by Emotion
def search_playlist_by_emotion(emotion):
    try:
        query = f"{emotion} music"
        results = sp.search(q=query, type='playlist', limit=1)
        items = results['playlists']['items']
        if items:
            playlist = items[0]
            playlist_id = playlist['id']
            playlist_name = playlist['name']
            tracks_data = sp.playlist_tracks(playlist_id)['items']
            tracks = [f"{track['track']['name']} by {track['track']['artists'][0]['name']}" for track in tracks_data[:5]]
            track_uris = [track['track']['uri'] for track in tracks_data[:5]]
            return playlist_name, tracks, track_uris
    except Exception as e:
        return None, [], []
    return None, [], []

# 🛠️ Create Custom Playlist
def create_custom_playlist(emotion, track_uris):
    user_id = sp.current_user()['id']
    playlist_name = f"{emotion.capitalize()} Vibes by Moodify"
    playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=True, description=f"Custom playlist for {emotion} mood")
    sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)
    return playlist['external_urls']['spotify']

# 🌈 Streamlit GUI
st.set_page_config(page_title="Moodify 🎶", layout="centered")
st.title("🎭 Mood-Based Music Recommender 🎶 — Made By ~ Aditya Gupta")
st.markdown("Upload an image, record your voice, or use your webcam to detect your mood and get a playlist!")

input_method = st.radio("Choose input method:", ["📷 Webcam", "🖼 Upload Image", "🎙️ Voice Recording"])

img = None
emotion = None

if input_method == "📷 Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture).convert("RGB")
        img = np.array(img)
        st.image(img, caption="Captured Image", use_column_width=True)
        with st.spinner("Analyzing emotion..."):
            emotion = detect_emotion_from_image(img)
            st.success(f"Detected Emotion: **{emotion.capitalize()}**")

elif input_method == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img = np.array(img)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing emotion..."):
            emotion = detect_emotion_from_image(img)
            st.success(f"Detected Emotion: **{emotion.capitalize()}**")

elif input_method == "🎙️ Voice Recording":
    audio_file = st.file_uploader("Upload a voice recording (WAV format)", type=["wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Analyzing voice tone..."):
            emotion = detect_emotion_from_audio(tmp_path)
            st.success(f"Detected Emotion from Voice: **{emotion.capitalize()}**")

# 🎵 Playlist Recommendation
if emotion and not emotion.startswith("Error"):
    playlist_name, tracks, track_uris = search_playlist_by_emotion(emotion)
    if playlist_name:
        st.subheader(f"🎶 Recommended Playlist: {playlist_name}")
        for track in tracks:
            st.write(f"- {track}")

        if st.button("Create Custom Playlist in My Spotify"):
            playlist_url = create_custom_playlist(emotion, track_uris)
            st.success(f"✅ Playlist created! [Open in Spotify]({playlist_url})")
    else:
        st.warning("No playlist found for this emotion.")
elif emotion and emotion.startswith("Error"):
    st.error(emotion)
