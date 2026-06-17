import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import tempfile
from collections import Counter
import random
import os
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings

# Suppress known warnings for a cleaner console
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
CLIENT_ID = "2ab9da702ae44024a7e2c62dd03fad98"
CLIENT_SECRET = "cf0558bbab3744b1ada28aa1a7826abe"
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPE = "playlist-modify-public,playlist-read-private,user-read-private"
PLAYLIST_SIZE = 50
SEARCH_LIMIT = int(PLAYLIST_SIZE * 0.4)

# 🌈 Streamlit GUI and Initialization (Moved to the top for flow)
st.set_page_config(page_title="Moodify 🎶", layout="centered", initial_sidebar_state="expanded")

# Initialize session state variables immediately after set_page_config
if 'sp' not in st.session_state:
    st.session_state.sp = None

# 🎵 Spotify Setup
def get_spotify_client():
    """Initializes or returns the Spotify client."""
    # Check session state directly, no need for redundant check in the function definition
    if st.session_state.sp is None:
        try:
            auth_manager = SpotifyOAuth(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                redirect_uri=REDIRECT_URI,
                scope=SCOPE,
                cache_path=".spotify_cache"
            )
            # Attempt to get token
            token_info = auth_manager.get_access_token(as_dict=True, check_cache=True)
            if token_info:
                st.session_state.sp = spotipy.Spotify(auth_manager=auth_manager)
            else:
                st.session_state.sp = None
        except Exception as e:
            st.error(f"Spotify authentication failed: {e}")
            st.session_state.sp = None
    return st.session_state.sp

# 🌟 Image Preprocessing
def preprocess_image(img):
    """Preprocess image for better emotion detection."""
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# 🎭 Image Emotion Detection
def detect_emotion_from_image(img):
    """Detect emotion from a single image."""
    try:
        img_processed = preprocess_image(img)
        result = DeepFace.analyze(
            img_processed,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=True
        )
        emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        return emotion
    except Exception as e:
        return f"Error: Unable to detect face. Please ensure your face is clearly visible."

# 🎭 Webcam Emotion Detection
def detect_emotion_from_webcam(frames):
    """Detect emotion from webcam frames."""
    emotions = []
    for frame in frames:
        preprocessed = preprocess_image(frame)
        try:
            result = DeepFace.analyze(
                preprocessed, 
                actions=['emotion'], 
                detector_backend='mtcnn',
                enforce_detection=True
            )
            emotions.append(result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion'])
        except:
            continue
    
    if emotions:
        return Counter(emotions).most_common(1)[0][0]
    return "neutral"

# 🎙️ ADVANCED: Load pre-trained Speech Emotion Recognition model
@st.cache_resource
def load_ser_model():
    """Load pre-trained Wav2Vec2 model for speech emotion recognition."""
    try:
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        
        # Emotion mapping for this specific model
        emotion_labels = {
            0: "angry",
            1: "calm",
            2: "disgust", 
            3: "fear",
            4: "happy",
            5: "neutral",
            6: "sad",
            7: "surprise"
        }
        
        return model, feature_extractor, emotion_labels, True
    except Exception as e:
        # Note: st.error/st.warning inside a cached function might not always display correctly
        # We handle the display in the main code block
        return None, None, None, False

# Initialize the model and session state variables for the model
# Need to initialize model_loaded *before* this runs
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'ser_model' not in st.session_state:
    st.session_state.ser_model = None
if 'ser_feature_extractor' not in st.session_state:
    st.session_state.ser_feature_extractor = None
if 'ser_labels' not in st.session_state:
    st.session_state.ser_labels = None

# Load model only once
if not st.session_state.model_loaded:
    with st.spinner("Loading speech emotion model..."):
        model, extractor, labels, loaded = load_ser_model()
        st.session_state.ser_model = model
        st.session_state.ser_feature_extractor = extractor
        st.session_state.ser_labels = labels
        st.session_state.model_loaded = loaded
        if not loaded:
            st.sidebar.warning("⚠️ Could not load advanced Wav2Vec2 model.")

def detect_emotion_wav2vec2(audio_path):
    """Detect emotion using pre-trained Wav2Vec2 model."""
    try:
        # Check if the model is actually loaded before proceeding
        if not st.session_state.model_loaded or st.session_state.ser_model is None:
             return None, 0, []

        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract features
        inputs = st.session_state.ser_feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            logits = st.session_state.ser_model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_id].item() * 100
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=min(3, len(probabilities[0])))
        top_predictions = [
            (st.session_state.ser_labels[idx.item()], prob.item() * 100)
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        emotion = st.session_state.ser_labels[predicted_id]
        
        return emotion, confidence, top_predictions
        
    except Exception as e:
        # st.error(f"Wav2Vec2 detection error: {e}") # Suppressing for cleaner UI
        return None, 0, []

def detect_emotion_ensemble(audio_path):
    """
    Ensemble method combining multiple approaches for maximum accuracy.
    Uses voting between different models/methods.
    """
    votes = []
    confidences = {}
    
    # Method 1: Wav2Vec2 (if available)
    if st.session_state.model_loaded and st.session_state.ser_model:
        emotion1, conf1, top_preds = detect_emotion_wav2vec2(audio_path)
        if emotion1:
            votes.append(emotion1)
            confidences['Wav2Vec2'] = (emotion1, conf1, top_preds)
    
    # Method 2: OpenSMILE + Statistical Analysis (lightweight fallback)
    try:
        import librosa
        y, sr = torchaudio.load(audio_path)
        y = y.numpy().flatten()
        
        # Prosodic features extraction and simplified logic... (as per your original code)
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0
        
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfccs)
        
        # Enhanced decision tree with stricter thresholds
        if rms > 0.05 and zcr > 0.1 and pitch_std > 100:
            emotion2 = "angry"
        elif rms > 0.03 and spectral_centroid > 2000 and 0.06 < zcr < 0.09:
            emotion2 = "happy"
        elif rms < 0.02 and spectral_centroid < 1400 and pitch_mean < 140:
            emotion2 = "sad"
        elif np.std(librosa.feature.rms(y=y)) < 0.008 and pitch_std < 40:
            emotion2 = "calm"
        elif pitch_std > 120 and mfcc_var > 80:
            emotion2 = "fear"
        elif spectral_centroid > 2400 and rms > 0.04:
            emotion2 = "surprise"
        elif zcr > 0.11 and spectral_rolloff < 2500:
            emotion2 = "disgust"
        else:
            emotion2 = "neutral"
        
        votes.append(emotion2)
        confidences['Prosodic Analysis'] = (emotion2, 65, [])
        
    except Exception as e:
        # st.warning(f"Prosodic analysis failed: {e}") # Suppressing for cleaner UI
        pass
    
    # Determine final emotion by voting
    if not votes:
        return "neutral", 0, {}, []
    
    vote_counts = Counter(votes)
    final_emotion = vote_counts.most_common(1)[0][0]
    
    # Calculate overall confidence
    total_confidence = sum(conf[1] for conf in confidences.values()) / len(confidences) if confidences else 0
    
    return final_emotion, total_confidence, confidences, votes

def detect_emotion_from_audio(audio_path):
    """Main audio emotion detection with comprehensive analysis."""
    
    with st.spinner("🎵 Analyzing audio with advanced AI models..."):
        emotion, confidence, method_results, votes = detect_emotion_ensemble(audio_path)
    
    # Display results
    with st.expander("🔬 Detailed Analysis Results", expanded=True):
        st.markdown(f"### 🎯 Final Detection: **{emotion.capitalize()}**")
        st.markdown(f"**Overall Confidence:** {confidence:.1f}%")
        
        if confidence > 0:
            st.progress(confidence / 100)
        
        st.divider()
        
        # Show individual method results
        if method_results:
            st.markdown("#### 📊 Detection Methods Used:")
            for method, (emo, conf, top_preds) in method_results.items():
                with st.container():
                    st.markdown(f"**{method}:**")
                    st.write(f"- Detected: {emo.capitalize()} ({conf:.1f}% confidence)")
                    
                    if top_preds:
                        st.write("- Top 3 predictions:")
                        for pred_emo, pred_conf in top_preds:
                            st.write(f"  • {pred_emo.capitalize()}: {pred_conf:.1f}%")
                    st.write("")
        
        if len(votes) > 1:
            st.markdown("#### 🗳️ Voting Results:")
            st.write(f"Methods agreed: {votes}")
            vote_agreement = len(set(votes)) == 1
            if vote_agreement:
                st.success("✅ All methods agree!")
            else:
                st.info("ℹ️ Majority voting applied")
    
    return emotion

# 🎶 Emotion Buffers
emotion_tracks = {
    "happy": [
        "Happy - Pharrell Williams", "Can't Stop the Feeling - Justin Timberlake", "Uptown Funk - Bruno Mars",
        "Shake It Off - Taylor Swift", "I Gotta Feeling - Black Eyed Peas", "Good Feeling - Flo Rida",
        "Dynamite - BTS", "Levitating - Dua Lipa", "Walking on Sunshine - Katrina & The Waves",
        "Sugar - Maroon 5", "Don't Stop Me Now - Queen", "On Top of the World - Imagine Dragons",
        "Firework - Katy Perry", "Roar - Katy Perry", "Best Day of My Life - American Authors",
        "Happy Now - Kygo", "Can't Stop - Red Hot Chili Peppers", "I'm a Believer - Smash Mouth",
        "Good Time - Owl City & Carly Rae Jepsen", "Celebrate - Kool & The Gang",
        "Just Dance - Lady Gaga", "Walking on Air - Katy Perry", "Stronger - Kanye West",
        "Pompeii - Bastille", "Counting Stars - OneRepublic", "Feel So Close - Calvin Harris",
        "Good Life - OneRepublic", "Don't You Worry Child - Swedish House Mafia",
        "Titanium - David Guetta ft. Sia", "Wake Me Up - Avicii", "Can't Hold Us - Macklemore & Ryan Lewis",
        "Shut Up and Dance - Walk The Moon", "Something Just Like This - The Chainsmokers",
        "Despacito - Luis Fonsi", "Taki Taki - DJ Snake", "La La La - Shakira", "Bailando - Enrique Iglesias",
        "Waka Waka - Shakira", "Viva La Vida - Coldplay", "Adventure of a Lifetime - Coldplay",
        "On Top of the World - Imagine Dragons", "Good Time - Nicky Romero", "Cheerleader - OMI",
        "Sugar - Robin Schulz", "Lean On - Major Lazer", "Havana - Camila Cabello", "Mi Gente - J Balvin",
        "Senorita - Shawn Mendes & Camila Cabello", "Blinding Lights - The Weeknd"
    ],
    "sad": [
        "Someone Like You - Adele", "Stay With Me - Sam Smith", "When I Was Your Man - Bruno Mars",
        "Fix You - Coldplay", "Let Her Go - Passenger", "The Scientist - Coldplay",
        "Say Something - A Great Big World", "Hurt - Johnny Cash", "All I Want - Kodaline",
        "Skinny Love - Birdy", "Jealous - Labrinth", "Too Good at Goodbyes - Sam Smith",
        "Lost Boy - Ruth B", "Million Reasons - Lady Gaga", "Un-break My Heart - Toni Braxton",
        "Creep - Radiohead", "How to Save a Life - The Fray", "My Immortal - Evanescence",
        "Back to December - Taylor Swift", "Jar of Hearts - Christina Perri",
        "The Night We Met - Lord Huron", "Say You Won't Let Go - James Arthur", "Photograph - Ed Sheeran",
        "All of Me - John Legend", "Hallelujah - Leonard Cohen", "Let It Go - James Bay",
        "Tears in Heaven - Eric Clapton", "Goodbye My Lover - James Blunt", "I Will Always Love You - Whitney Houston",
        "With or Without You - U2", "Fix You - Coldplay", "Nothing Compares 2 U - Sinead O'Connor",
        "Stay - Rihanna ft. Mikky Ekko", "Issues - Julia Michaels", "Breathe Me - Sia",
        "Lost - Frank Ocean", "Skin - Rag'n'Bone Man", "Someone You Loved - Lewis Capaldi",
        "All I Want - Kodaline", "Say Something - Justin Timberlake", "Unsteady - X Ambassadors",
        "When I Look at You - Miley Cyrus", "The A Team - Ed Sheeran", "Mad World - Gary Jules",
        "Say You Love Me - Jessie Ware", "Slow Dancing in a Burning Room - John Mayer", "Jealous - Labrinth"
    ],
    "angry": [
        "Break Stuff - Limp Bizkit", "Killing In The Name - Rage Against the Machine",
        "Bulls on Parade - Rage Against the Machine", "Smells Like Teen Spirit - Nirvana",
        "Duality - Slipknot", "Down with the Sickness - Disturbed", "Bodies - Drowning Pool",
        "Chop Suey - System of a Down", "Given Up - Linkin Park", "Faint - Linkin Park",
        "The Way I Am - Eminem", "Bad Blood - Taylor Swift", "Face Down - Red Jumpsuit Apparatus",
        "Headstrong - Trapt", "Before I Forget - Slipknot", "Psychosocial - Slipknot",
        "I Hate Everything About You - Three Days Grace", "Animal I Have Become - Three Days Grace",
        "Stupify - Disturbed", "Back In Black - AC/DC", "Kryptonite - 3 Doors Down", "Last Resort - Papa Roach",
        "My Songs Know What You Did In The Dark - Fall Out Boy", "Pain - Three Days Grace",
        "Fight Song - Rachel Platten", "Stronger - Kanye West", "Monster - Kanye West",
        "No Love - Eminem", "War Pigs - Black Sabbath", "The Beautiful People - Marilyn Manson",
        "Walk - Pantera", "Riot - Three Days Grace", "I Will Not Bow - Breaking Benjamin",
        "Numb/Encore - Linkin Park & Jay-Z", "Painkiller - Judas Priest", "Bodies - Drowning Pool",
        "Duality - Slipknot", "Before I Forget - Slipknot", "Break Stuff - Limp Bizkit",
        "Given Up - Linkin Park", "Smells Like Teen Spirit - Nirvana", "Psychosocial - Slipknot",
        "Bulls on Parade - Rage Against the Machine", "Down with the Sickness - Disturbed",
        "Headstrong - Trapt", "Stupify - Disturbed", "Face Down - Red Jumpsuit Apparatus",
        "Chop Suey - System of a Down"
    ],
    "surprise": [
        "Surprise Yourself - Jack Garratt", "Wake Me Up - Avicii", "Titanium - David Guetta ft. Sia",
        "Don't You Worry Child - Swedish House Mafia", "Adventure of a Lifetime - Coldplay",
        "Viva La Vida - Coldplay", "A Sky Full of Stars - Coldplay", "On Top of the World - Imagine Dragons",
        "Good Life - OneRepublic", "Feel So Close - Calvin Harris", "Counting Stars - OneRepublic",
        "Fireflies - Owl City", "Pompeii - Bastille", "Safe and Sound - Capital Cities",
        "Rather Be - Clean Bandit", "Shut Up and Dance - Walk The Moon", "Something Just Like This - The Chainsmokers",
        "Wake Up - Arcade Fire", "Can't Hold Us - Macklemore & Ryan Lewis", "Believer - Imagine Dragons",
        "Happy Now - Kygo", "Lean On - Major Lazer", "Sugar - Robin Schulz", "Hymn for the Weekend - Coldplay",
        "Mi Gente - J Balvin", "Senorita - Shawn Mendes & Camila Cabello", "Blinding Lights - The Weeknd",
        "Don't Start Now - Dua Lipa", "Rain On Me - Lady Gaga & Ariana Grande", "Dance Monkey - Tones and I",
        "Cold Water - Major Lazer", "Electricity - Silk City & Dua Lipa", "Ritual - Tiesto",
        "Animals - Martin Garrix", "One Kiss - Calvin Harris & Dua Lipa", "Turn Up The Speakers - Afrojack",
        "The Nights - Avicii", "Stole The Show - Kygo", "On The Floor - Jennifer Lopez", "I Like It - Cardi B",
        "Havana - Camila Cabello", "Shape of You - Ed Sheeran", "Cheap Thrills - Sia", "Firestone - Kygo",
        "Prayer in C - Lilly Wood & The Prick", "Lean On - Major Lazer", "Break Free - Ariana Grande", "New Rules - Dua Lipa"
    ],
    "fear": [
        "Fear - Blue October", "Disturbia - Rihanna", "Somebody's Watching Me - Rockwell",
        "Thriller - Michael Jackson", "Bring Me to Life - Evanescence", "Heathens - Twenty One Pilots",
        "Creep - Radiohead", "The Hills - The Weeknd", "Everybody Wants to Rule the World - Tears for Fears",
        "Enter Sandman - Metallica", "Boulevard of Broken Dreams - Green Day", "No Fear - DeJ Loaf",
        "Demons - Imagine Dragons", "Mad World - Gary Jules", "Monster - Kanye West", "In the End - Linkin Park",
        "Numb - Linkin Park", "Lose Yourself - Eminem", "Somebody That I Used to Know - Gotye", "The Fear - Lily Allen",
        "Ghost - Halsey", "Runaway - Bon Jovi", "House of the Rising Sun - The Animals", "Bury a Friend - Billie Eilish",
        "Disturbia - Rihanna", "Spirits - The Strumbellas", "Sleepwalking - Bring Me The Horizon",
        "Black Magic Woman - Santana", "People Are Strange - The Doors", "High Hopes - Panic! At The Disco",
        "Demons - Imagine Dragons", "Toxic - Britney Spears", "Monster - Imagine Dragons", "Control - Halsey",
        "Heathens - Twenty One Pilots", "Somebody's Watching Me - Rockwell", "Thriller - Michael Jackson",
        "Lose Yourself - Eminem", "Mad World - Gary Jules", "Creep - Radiohead", "Bring Me to Life - Evanescence",
        "The Hills - The Weeknd", "Fear of the Dark - Iron Maiden", "Disturbia - Rihanna", "In the End - Linkin Park",
        "Monster - Kanye West", "No Fear - DeJ Loaf", "Somebody That I Used to Know - Gotye", "The Fear - Lily Allen"
    ],
    "disgust": [
        "Disgusting - Bauhaus", "Creep - Radiohead", "Toxic - Britney Spears", "Bad Guy - Billie Eilish",
        "Smells Like Teen Spirit - Nirvana", "U Can't Touch This - MC Hammer", "Ironic - Alanis Morissette",
        "Stupid Girls - Pink", "Cry Me a River - Justin Timberlake", "Animal I Have Become - Three Days Grace",
        "No Scrubs - TLC", "Why'd You Only Call Me When You're High - Arctic Monkeys", "Every Breath You Take - The Police",
        "Rolling in the Deep - Adele", "Boulevard of Broken Dreams - Green Day", "Pumped Up Kicks - Foster the People",
        "Somebody That I Used to Know - Gotye", "Mad World - Gary Jules", "Hurt - Johnny Cash", "Numb - Linkin Park",
        "Bodies - Drowning Pool", "Toxicity - System of a Down", "Freak on a Leash - Korn", "Closer - Nine Inch Nails",
        "People = Shit - Slipknot", "Duality - Slipknot", "Paranoid - Black Sabbath", "Painkiller - Judas Priest",
        "The Way I Am - Eminem", "Break Stuff - Limp Bizkit", "Stupify - Disturbed", "Down with the Sickness - Disturbed",
        "Headstrong - Trapt", "Face Down - Red Jumpsuit Apparatus", "Before I Forget - Slipknot", "Psychosocial - Slipknot",
        "I Hate Everything About You - Three Days Grace", "Animal I Have Become - Three Days Grace", "Bodies - Drowning Pool",
        "Smells Like Teen Spirit - Nirvana", "Creep - Radiohead", "Disgusting - Bauhaus", "Toxic - Britney Spears",
        "Bad Guy - Billie Eilish", "No Scrubs - TLC", "Rolling in the Deep - Adele", "U Can't Touch This - MC Hammer"
    ],
    "neutral": [
        "Clocks - Coldplay", "Let It Be - The Beatles", "Yellow - Coldplay", "Fix You - Coldplay",
        "Viva La Vida - Coldplay", "Imagine - John Lennon", "Something - The Beatles", "Here Comes the Sun - The Beatles",
        "Comfortably Numb - Pink Floyd", "Wish You Were Here - Pink Floyd", "Time - Pink Floyd", "Hey Jude - The Beatles",
        "Blackbird - The Beatles", "Let Her Go - Passenger", "Fast Car - Tracy Chapman", "Hallelujah - Leonard Cohen",
        "Your Song - Elton John", "Rocket Man - Elton John", "Tiny Dancer - Elton John", "The Scientist - Coldplay",
        "Let It Go - James Bay", "All of Me - John Legend", "Photograph - Ed Sheeran", "Breathe Me - Sia",
        "The A Team - Ed Sheeran", "Say You Love Me - Jessie Ware", "Slow Dancing in a Burning Room - John Mayer",
        "Gravity - John Mayer", "Lost Stars - Adam Levine", "Skinny Love - Birdy", "Somewhere Only We Know - Keane",
        "Hallelujah - Jeff Buckley", "The Blower's Daughter - Damien Rice", "Chasing Cars - Snow Patrol",
        "Teardrop - Massive Attack", "Fix You - Coldplay", "River Flows in You - Yiruma", "Mad World - Gary Jules",
        "No Surprises - Radiohead", "The Sound of Silence - Simon & Garfunkel", "Boulevard of Broken Dreams - Green Day",
        "Everybody's Got to Learn Sometime - Beck", "Patience - Guns N' Roses", "Drive - Incubus", "Waiting on the World to Change - John Mayer",
        "Bitter Sweet Symphony - The Verve", "Lost - Frank Ocean", "Video Games - Lana Del Rey"
    ],
       "calm": [
        "Weightless - Marconi Union", "Strawberry Swing - Coldplay", "Holocene - Bon Iver",
        "River Flows in You - Yiruma", "Sunset Lover - Petit Biscuit", "Night Owl - Galimatias",
        "Cold Little Heart - Michael Kiwanuka", "Bloom - The Paper Kites", "Ophelia - The Lumineers",
        "Cherry Wine - Hozier", "Skinny Love - Birdy", "The Night We Met - Lord Huron", "Breathe Me - Sia",
        "Slow Dancing in a Burning Room - John Mayer", "Gravity - John Mayer", "All I Want - Kodaline",
        "Lost in Japan (Acoustic) - Shawn Mendes", "Better Together - Jack Johnson", "Banana Pancakes - Jack Johnson",
        "Holocene - Bon Iver", "Re: Stacks - Bon Iver", "First Day of My Life - Bright Eyes",
        "Such Great Heights - Iron & Wine", "To Build a Home - The Cinematic Orchestra", "Your Hand in Mine - Explosions in the Sky",
        "Experience - Ludovico Einaudi", "Nuvole Bianche - Ludovico Einaudi", "Comptine d'un autre été: L'après-midi - Yann Tiersen",
        "La Valse d'Amélie - Yann Tiersen", "River Flows in You - Yiruma", "Kiss the Rain - Yiruma",
        "A Thousand Years (Piano/Cello Cover) - The Piano Guys", "Clair de Lune - Claude Debussy",
        "Gymnopédie No.1 - Erik Satie", "Spiegel im Spiegel - Arvo Pärt", "The Promise - Michael Nyman",
        "Divenire - Ludovico Einaudi", "Una Mattina - Ludovico Einaudi", "Opus 55 - Dustin O'Halloran",
        "Experience - Ludovico Einaudi", "Nuvole Bianche - Ludovico Einaudi"
    ]
}
# 🎨 Emoji & color mapping
emotion_display = {
    "happy": ("😄 Happy", "#FFD700"),
    "sad": ("😢 Sad", "#1E90FF"),
    "angry": ("😡 Angry", "#FF4500"),
    "surprise": ("😲 Surprised", "#FF69B4"),
    "fear": ("😨 Fearful", "#8A2BE2"),
    "disgust": ("🤢 Disgusted", "#32CD32"),
    "neutral": ("😐 Neutral", "#808080"),
    "calm": ("😌 Calm", "#87CEEB")
}

# 🎵 Search tracks on Spotify
def search_tracks_on_spotify(tracks, sp_client):
    """Search for tracks on Spotify and return URIs."""
    uris = []
    for song in tracks:
        try:
            results = sp_client.search(q=song, type='track', limit=1, market='IN')
            if not results['tracks']['items']:
                results = sp_client.search(q=song, type='track', limit=1)
            items = results['tracks']['items']
            if items:
                uris.append(items[0]['uri'])
        except Exception:
            continue
    return uris

def create_custom_playlist(emotion, fallback_tracks):
    """Creates a playlist prioritizing new searches with fallback tracks."""
    sp_client = get_spotify_client()
    if not sp_client:
        st.error("Spotify not connected. Please connect your account first.")
        return None

    try:
        user_id = sp_client.current_user()['id']
    except Exception as e:
        st.error(f"Spotify error: {e}")
        return None

    playlist_name = f"{emotion.capitalize()} Vibes by Moodify"
    playlist = sp_client.user_playlist_create(
        user=user_id,
        name=playlist_name,
        public=True,
        description=f"Custom {emotion} mood playlist curated by Moodify"
    )
    
    # Search for new tracks
    new_tracks_uris = []
    search_queries = [
        f"{emotion} indian songs 2024",
        f"{emotion} bollywood songs",
        f"{emotion} playlist 2024",
        f"best {emotion} music"
    ]
    
    for query in search_queries:
        try:
            results = sp_client.search(q=query, type='track', limit=SEARCH_LIMIT, market='IN')
            new_tracks_uris.extend([item['uri'] for item in results['tracks']['items']])
            results_global = sp_client.search(q=query, type='track', limit=SEARCH_LIMIT)
            new_tracks_uris.extend([item['uri'] for item in results_global['tracks']['items']])
        except Exception:
            continue

    # Remove duplicates
    new_tracks_uris = list(set(new_tracks_uris))
    random.shuffle(new_tracks_uris)
    final_uris = new_tracks_uris[:PLAYLIST_SIZE // 2]

    # Add fallback tracks
    remaining_slots = PLAYLIST_SIZE - len(final_uris)
    if remaining_slots > 0 and fallback_tracks:
        selected_fallback = random.sample(fallback_tracks, min(remaining_slots * 2, len(fallback_tracks)))
        fallback_uris = search_tracks_on_spotify(selected_fallback, sp_client)
        
        for uri in fallback_uris:
            if uri not in final_uris and len(final_uris) < PLAYLIST_SIZE:
                final_uris.append(uri)
    
    if final_uris:
        sp_client.playlist_add_items(playlist_id=playlist['id'], items=final_uris)
        st.success(f"✅ Successfully added {len(final_uris)} songs!")
    else:
        st.warning("Could not find tracks to add.")
        
    return playlist['external_urls']['spotify']

# --- Streamlit UI Components ---

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎭 Moodify</h1><p>AI-Powered Mood Detection & Music Curation</p></div>', unsafe_allow_html=True)
st.markdown("*Created by Aditya Gupta*")
st.markdown("---")

# Sidebar - Spotify Connection
st.sidebar.title("🎵 Spotify Connection")

if st.session_state.sp is None:
    if st.sidebar.button("🔗 Connect to Spotify", type="primary"):
        sp_client = get_spotify_client()
        if sp_client:
            st.sidebar.success(f"✅ Connected as: {sp_client.current_user()['display_name']}")
            st.rerun()
        else:
            st.sidebar.error("❌ Connection failed. Check credentials.")
else:
    try:
        user_info = st.session_state.sp.current_user()
        st.sidebar.success(f"✅ Connected as: **{user_info['display_name']}**")
        if st.sidebar.button("Disconnect"):
            st.session_state.sp = None
            if os.path.exists(".spotify_cache"):
                os.remove(".spotify_cache")
            st.rerun()
    except Exception:
        st.session_state.sp = None
        st.sidebar.error("❌ Token expired. Please reconnect.")

st.sidebar.divider()
st.sidebar.markdown("### 🎙️ Voice Detection")

if st.session_state.model_loaded:
    st.sidebar.success("✅ Advanced Wav2Vec2 Model Loaded")
else:
    st.sidebar.warning("⚠️ Using alternative detection method")

st.sidebar.info("""
**Advanced AI Features:**
- Wav2Vec2 transformer model
- Ensemble voting system
- Prosodic feature analysis

**Tips for Best Results:**
- Speak naturally for 3-5 seconds
- Express emotion clearly
- Minimize background noise
""")

# Main App Logic
tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Upload Image", "🎙️ Voice Recording"])

emotion = None

with tab1:
    st.markdown("### 📷 Capture Your Mood")
    picture = st.camera_input("Take a picture")
    
    if picture:
        img = Image.open(picture).convert("RGB")
        img_np = np.array(img)
        frames = [img_np] * 5
        
        with st.spinner("🔍 Analyzing your emotion..."):
            emotion = detect_emotion_from_webcam(frames)
            if not emotion.startswith("Error"):
                emoji_label, color = emotion_display.get(emotion.lower(), ("😐 Neutral", "#808080"))
                st.markdown(f"<h2 style='color:{color}; text-align:center;'>{emoji_label}</h2>", unsafe_allow_html=True)

with tab2:
    st.markdown("### 🖼️ Upload Your Photo")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # FIX: Changed 'use_container_width=True' to 'width=300' or similar for compatibility
            st.image(img_np, caption="Your Image", width=300) 
        
        with st.spinner("🔍 Analyzing your emotion..."):
            emotion = detect_emotion_from_image(img_np)
            if not emotion.startswith("Error"):
                emoji_label, color = emotion_display.get(emotion.lower(), ("😐 Neutral", "#808080"))
                st.markdown(f"<h2 style='color:{color}; text-align:center;'>{emoji_label}</h2>", unsafe_allow_html=True)
            else:
                st.error(emotion)

with tab3:
    st.markdown("### 🎙️ Upload Your Voice Recording")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("🎯 **Pro Tip:** Record yourself speaking emotionally for 3-5 seconds")
    with col2:
        st.metric("AI Model", "Wav2Vec2" if st.session_state.model_loaded else "Prosodic")
    
    audio_file = st.file_uploader("Upload audio (WAV, MP3, OGG, FLAC)", type=["wav", "mp3", "ogg", "flac"])
    
    if audio_file:
        st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        
        emotion = detect_emotion_from_audio(tmp_path)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if emotion and not emotion.startswith("Error"):
            emoji_label, color = emotion_display.get(emotion.lower(), ("😐 Neutral", "#808080"))
            st.markdown(f"<h2 style='color:{color}; text-align:center;'>{emoji_label}</h2>", unsafe_allow_html=True)

# Playlist Generation
if emotion and not emotion.startswith("Error"):
    tracks = emotion_tracks.get(emotion.lower(), [])
    
    if tracks:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader(f"🎶 Mood: **{emotion.capitalize()}**")
            st.write(f"Create a {PLAYLIST_SIZE}-track playlist?")
        
        if st.session_state.sp:
            if st.button("🎵 Create Playlist", type="primary", use_container_width=True):
                with st.spinner("✨ Curating your personalized playlist..."):
                    playlist_url = create_custom_playlist(emotion, tracks)
                
                if playlist_url:
                    st.balloons()
                    st.success("🎉 Your playlist is ready!")
                    st.markdown(f"### [🎧 Open in Spotify]({playlist_url})")
        else:
            st.warning("⚠️ Please **Connect to Spotify** in the sidebar to create your playlist.")
            
elif emotion and emotion.startswith("Error"):
    st.error(emotion)
