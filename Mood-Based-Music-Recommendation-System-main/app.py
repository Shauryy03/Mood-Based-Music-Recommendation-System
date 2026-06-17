import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import librosa
import tempfile
from collections import Counter
import random

# 🎵 Spotify Setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="2ab9da702ae44024a7e2c62dd03fad98",
    client_secret="cf0558bbab3744b1ada28aa1a7826abe",
    redirect_uri="http://127.0.0.1:8888",
    scope="playlist-modify-public,playlist-read-private,user-read-private",
    cache_path=".spotify_cache"
))

# 🌟 Image Preprocessing (no cv2)
def preprocess_image(img):
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Histogram equalization on Y channel
    img_yuv = img.convert("YCbCr")
    y, cb, cr = img_yuv.split()
    y_eq = ImageOps.equalize(y)  # Equalize luminance
    img_eq = Image.merge("YCbCr", (y_eq, cb, cr)).convert("RGB")
    return np.array(img_eq)

# 🎭 Image Emotion Detection
def detect_emotion_from_image(img_np):
    try:
        img_processed = preprocess_image(Image.fromarray(img_np))
        result = DeepFace.analyze(
            img_processed,
            actions=['emotion'],
            detector_backend='mtcnn',
            enforce_detection=True
        )
        emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        return emotion
    except Exception as e:
        return f"Error: {str(e)}"

# 🎭 Webcam Emotion Detection
def detect_emotion_from_webcam(frames):
    emotions = []
    for frame in frames:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='mtcnn', enforce_detection=True)
            emotions.append(result[0]['dominant_emotion'])
        except:
            continue
    if emotions:
        return Counter(emotions).most_common(1)[0][0]
    return "neutral"

# 🎙️ Voice Emotion Detection
def detect_emotion_from_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        try:
            pitch = librosa.yin(y, fmin=50, fmax=300)
            pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0
            pitch_std = np.std(pitch) if len(pitch) > 0 else 0
        except:
            pitch_mean = pitch_std = 0
        energy = np.mean(librosa.feature.rms(y=y))
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        if energy > 0.05 and contrast > 20:
            return "happy"
        elif pitch_mean < 100 and energy < 0.03:
            return "sad"
        elif contrast > 25 and pitch_mean > 150 and pitch_std > 20:
            return "angry"
        else:
            return "neutral"
    except Exception as e:
        return f"Error: {str(e)}"

# 🎶 Emotion Buffers (shortened here — keep your full 50 songs per emotion)
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
    ]
    # Continue with similar 50-song buffers for "surprise", "fear", "disgust", "neutral" with global variety
}


# 🎨 Emoji & color mapping
emotion_display = {
    "happy": ("😄 Happy", "#FFD700"),
    "sad": ("😢 Sad", "#1E90FF"),
    "angry": ("😡 Angry", "#FF4500"),
    "surprise": ("😲 Surprised", "#FF69B4"),
    "fear": ("😨 Fearful", "#8A2BE2"),
    "disgust": ("🤢 Disgusted", "#32CD32"),
    "neutral": ("😐 Neutral", "#808080")
}

# 🎵 Search tracks on Spotify
def search_tracks_on_spotify(tracks):
    uris = []
    for song in tracks:
        try:
            results = sp.search(q=song, type='track', limit=1)
            items = results['tracks']['items']
            if items:
                uris.append(items[0]['uri'])
        except:
            continue
    return uris

# 🛠️ Create custom playlist
def create_custom_playlist(emotion, tracks):
    try:
        user_id = sp.current_user()['id']
    except spotipy.exceptions.SpotifyException:
        st.error("Spotify token invalid. Please log in again.")
        return None
    playlist_name = f"{emotion.capitalize()} Vibes by Moodify"
    playlist = sp.user_playlist_create(
        user=user_id,
        name=playlist_name,
        public=True,
        description=f"Custom playlist for {emotion} mood"
    )
    selected_tracks = random.sample(tracks, min(50, len(tracks)))
    track_uris = search_tracks_on_spotify(selected_tracks)
    if track_uris:
        sp.playlist_add_items(playlist_id=playlist['id'], items=track_uris)
    return playlist['external_urls']['spotify']

# 🌈 Streamlit GUI
st.set_page_config(page_title="Moodify 🎶", layout="centered")
st.title("🎭 Mood-Based Music Recommender 🎶 — Made By ~ Aditya Gupta")
st.markdown("Upload an image, record your voice, or use your webcam to detect your mood and get a playlist!")

input_method = st.radio("Choose input method:", ["📷 Webcam", "🖼 Upload Image", "🎙️ Voice Recording"])
emotion = None

# 📷 Webcam
if input_method == "📷 Webcam":
    picture = st.camera_input("Take a picture", key="webcam_input")
    if picture:
        img = Image.open(picture).convert("RGB")
        img_np = np.array(img)
        frames = [img_np] * 5
        with st.spinner("Analyzing emotion from webcam..."):
            emotion = detect_emotion_from_webcam(frames)
            emoji_label, color = emotion_display.get(emotion.lower(), ("😐 Neutral", "#808080"))
            st.markdown(f"<h2 style='color:{color}'>{emoji_label}</h2>", unsafe_allow_html=True)

# 🖼 Upload Image
elif input_method == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img_np, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Analyzing emotion..."):
            emotion = detect_emotion_from_image(img_np)
            emoji_label, color = emotion_display.get(emotion.lower(), ("😐 Neutral", "#808080"))
            st.markdown(f"<h2 style='color:{color}'>{emoji_label}</h2>", unsafe_allow_html=True)

# 🎙 Voice Recording
elif input_method == "🎙️ Voice Recording":
    audio_file = st.file_uploader("Upload a voice recording (WAV format)", type=["wav"])
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Analyzing voice tone..."):
            emotion = detect_emotion_from_audio(tmp_path)
            emoji_label, color = emotion_display.get(emotion.lower(), ("😐 Neutral", "#808080"))
            st.markdown(f"<h2 style='color:{color}'>{emoji_label}</h2>", unsafe_allow_html=True)

# 🎵 Playlist Recommendation
if emotion and not emotion.startswith("Error"):
    tracks = emotion_tracks.get(emotion.lower(), [])
    if tracks:
        st.subheader(f"🎶 Detected Emotion: {emotion.capitalize()}")
        st.write("Creating a custom playlist with 50 tracks for your mood...")
        if st.button("Create Custom Playlist in My Spotify"):
            playlist_url = create_custom_playlist(emotion, tracks)
            if playlist_url:
                st.success(f"✅ Playlist created! [Open in Spotify]({playlist_url})")
elif emotion and emotion.startswith("Error"):
    st.error(emotion)
