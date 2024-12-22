import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from gtts import gTTS
import os
import subprocess
import threading


model = tf.keras.models.load_model('./models/asl_model.h5')
labels = ['goodbye', 'hello', 'I_Love_You', 'thanks', 'where', 'yes']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

image_size = (224, 224)

cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

last_prediction = None

# gTTS function
def gtts_tts(tts_text):
    tts = gTTS(tts_text, lang='ja')  
    
    temp_audio_file = 'temp_audio.mp3'
    tts.save(temp_audio_file)
    
    subprocess.run(["afplay", temp_audio_file])

def async_generate_voice(text):
   
    threading.Thread(target=gtts_tts, args=(text,)).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  

            hand_image = frame[y_min:y_max, x_min:x_max]
            hand_image_resized = cv2.resize(hand_image, image_size)
            hand_image_normalized = hand_image_resized / 255.0
            hand_image_normalized = np.expand_dims(hand_image_normalized, axis=0)

            predictions = model.predict(hand_image_normalized)
            predicted_label_index = np.argmax(predictions)
            predicted_label = labels[predicted_label_index]
            display_label = predicted_label.replace('_', ' ') 
            probability = predictions[0][predicted_label_index] * 100  

            if predicted_label != last_prediction:
                async_generate_voice(display_label) 
                last_prediction = predicted_label
 
            
            cv2.putText(frame, f"Prediction: {display_label}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  
            cv2.putText(frame, f"Confidence: {probability:.2f}%", (x_min, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('ASL Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
