import streamlit as st
import cv2
import numpy as np
import random
import time
from datetime import datetime
import os
import pickle
import pygame

# Page configuration
st.set_page_config(
    page_title="Emotion Detection Game",
    page_icon="😊",
    layout="wide"
)

# Initialize pygame for sound
pygame.init()
pygame.mixer.init()

# Load sound effects and music
point_sound = pygame.mixer.Sound('point.wav')
background_music = pygame.mixer.Sound('background.wav')
end_music = pygame.mixer.Sound('end.wav')

# Play background music
def play_background_music():
    pygame.mixer.Sound.play(background_music, loops=-1)

# Stop background music
def stop_background_music():
    pygame.mixer.Sound.stop(background_music)

# Play end music
def play_end_music():
    pygame.mixer.Sound.play(end_music)

# Initialize session state variables
if 'game_active' not in st.session_state:
    st.session_state.game_active = False
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'current_target' not in st.session_state:
    st.session_state.current_target = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'emotion_start_time' not in st.session_state:
    st.session_state.emotion_start_time = None
if 'current_play_folder' not in st.session_state:
    st.session_state.current_play_folder = None

# Constants
GAME_DURATION = 60  # Game duration in seconds
REQUIRED_DURATION = 0.2  # Time required to maintain emotion
emotion_dict = {
    0: "Happy", 1: "Neutral", 2: "Sad", 3: "Surprised"
}
emotion_targets = ["Happy", "Surprised", "Neutral", "Sad"]

# Load face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_model():
    try:
        with open(r'model\emotion_model_100.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def get_emotion_model():
    return load_model()

def preprocess_image(roi_gray):
    cropped_img = cv2.resize(roi_gray, (48, 48))
    cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
    cropped_img = cropped_img / 255.0
    return cropped_img

def weighted_choice(choices, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(choices, weights):
        if upto + w >= r:
            return c
        upto += w
    return choices[0]

def choose_next_target(previous_target):
    weights = [10 if emotion != previous_target else 1 for emotion in emotion_targets]
    return weighted_choice(emotion_targets, weights)

def create_play_folder():
    main_folder = "played_images"
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_play_folder = os.path.join(main_folder, f"play_{timestamp}")
    os.makedirs(st.session_state.current_play_folder)

def draw_emotion_bars(frame, x, y, w, h, emotion_probs):
    bar_x = x + w + 10  # Position for the bar chart
    bar_y = y  
    bar_height = 20
    bar_width = 150
    space_between_bars = 5

    for idx, (emotion, prob) in enumerate(zip(emotion_dict.values(), emotion_probs)):
        bar_length = int(bar_width * prob)
        color = (0, 255, 0) if emotion != "Sad" else (0, 0, 255)

        cv2.rectangle(frame, 
                      (bar_x, bar_y + idx * (bar_height + space_between_bars)), 
                      (bar_x + bar_length, bar_y + idx * (bar_height + space_between_bars) + bar_height), 
                      color, -1)

        text = f"{emotion}: {int(prob * 100)}%"
        cv2.putText(frame, text, 
                    (bar_x, bar_y + idx * (bar_height + space_between_bars) + bar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def start_game():
    st.session_state.game_active = True
    st.session_state.game_over = False
    st.session_state.score = 0
    st.session_state.current_target = random.choice(emotion_targets)
    st.session_state.start_time = time.time()
    st.session_state.emotion_start_time = None
    create_play_folder()
    play_background_music()

def end_game():
    st.session_state.game_active = False
    st.session_state.game_over = True
    st.session_state.emotion_start_time = None  # Clear emotion start time to prevent further scoring
    stop_background_music()  # Stop background music
    play_end_music()  # Play end music
    st.balloons()  # Celebrate with balloons
    st.success("🎉 Congratulations you have completed the game!") 


def main():
    st.title("😊 Emotion Detection Game")

    st.sidebar.title("Game Controls")
    st.sidebar.subheader("Game Info")
    st.sidebar.write("Make facial expressions to score points!")

    emotion_model = get_emotion_model()
    if emotion_model is None:
        st.error("Failed to load emotion detection model")
        return

    if st.sidebar.button("🎮 Start Game", key="start_button") and not st.session_state.game_active:
        start_game()

    st.sidebar.markdown("## 📜 Instructions", unsafe_allow_html=True)
    st.sidebar.info("""\
1. Click 'Start Game' to begin
2. Match your facial expression with the target emotion
3. Hold the expression briefly to score
4. Get as many points as possible within the time limit!
""")

    col1, col2 = st.columns([4, 1])

    with col1:
        video_placeholder = st.empty()

    with col2:
        status_placeholder = st.empty()
        score_placeholder = st.empty()
        target_placeholder = st.empty()
        timer_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5, minSize=(300, 300))

            if st.session_state.game_active:
                elapsed_time = time.time() - st.session_state.start_time
                time_left = max(0, GAME_DURATION - int(elapsed_time))
            
                status_placeholder.markdown("### Game Status: Active")
                score_placeholder.markdown(f"### Score: {st.session_state.score}")
                target_placeholder.markdown(f"### Target Emotion: {st.session_state.current_target}")
                timer_placeholder.markdown(f"### Time Left: {time_left}s")
                
                if elapsed_time >= GAME_DURATION:
                    end_game()
                    continue
            
            elif st.session_state.game_over:
                status_placeholder.markdown("### Game Over!")
                score_placeholder.markdown(f"### Final Score: {st.session_state.score}")
                target_placeholder.empty()
                timer_placeholder.empty()
            else:
                status_placeholder.markdown("### Press 'Start Game' to begin!")
                score_placeholder.empty()
                target_placeholder.empty()
                timer_placeholder.empty()          

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y + h, x:x + w]
                cropped_img = preprocess_image(roi_gray)

                try:
                    emotion_prediction = emotion_model.predict(cropped_img)
                    if isinstance(emotion_prediction, np.ndarray) and emotion_prediction.ndim > 1:
                        emotion_probs = emotion_prediction[0]
                    else:
                        emotion_probs = emotion_prediction

                    maxindex = int(np.argmax(emotion_probs))
                    emotion_text = emotion_dict.get(maxindex, "Unknown")
                    
                    color = (0, 255, 0) if emotion_text != "Sad" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    draw_emotion_bars(frame, x, y, w, h, emotion_probs)

                    if st.session_state.game_active and emotion_text == st.session_state.current_target:
                        if st.session_state.emotion_start_time is None:
                            st.session_state.emotion_start_time = time.time()
                        elif time.time() - st.session_state.emotion_start_time >= REQUIRED_DURATION:
                            # Capture and save the frame with game labels and remaining time
                            labeled_frame = frame.copy()
                            elapsed_time = time.time() - st.session_state.start_time
                            time_left = max(0, GAME_DURATION - int(elapsed_time))
                            cv2.putText(labeled_frame, f"Score: {st.session_state.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(labeled_frame, f"Target: {st.session_state.current_target}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.putText(labeled_frame, f"Time Left: {time_left}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            
                            # Save the image
                            image_filename = f'point_frame_{st.session_state.score}_time_left_{time_left}s.jpg'
                            image_path = os.path.join(st.session_state.current_play_folder, image_filename)
                            cv2.imwrite(image_path, labeled_frame)
                            
                            st.session_state.score += 1
                            st.session_state.current_target = choose_next_target(st.session_state.current_target)
                            st.session_state.emotion_start_time = None
                            point_sound.play()
                    else:
                        st.session_state.emotion_start_time = None
                except Exception as e:
                    st.error(f"Error processing frame: {e}")

            video_placeholder.image(frame, channels="BGR")

            # Add a small sleep to prevent excessive CPU usage
            time.sleep(0.03)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_background_music()  # Ensure music stops when the app is closed or interrupted

if __name__ == "__main__":
    main()
