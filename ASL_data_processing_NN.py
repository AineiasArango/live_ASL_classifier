import cv2
import numpy as np
import os
import mediapipe as mp

#%%

#hand detection using mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) #detection using mediapipe
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

#extracting coordinates of each hand landmark
def extract_hand_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hands = np.array([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z] for i in range(len(hand_landmarks.landmark))]).flatten()
    else:
        hands = np.zeros(21*3)        
    return hands

mp_hands = mp.solutions.hands

#%% set up training data

train_dir = "C:\\Users\\sakai\\OneDrive\\Documents\\Year4\\Coding_projects\\ASL_recognition\\asl_alphabet_train"
os.chdir(train_dir)

data = []
labels = []

#find hand landmarks for every image in the folder
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
    i = 0
    #iterate through files containing each letter
    for letter_file in os.listdir():
        os.chdir(train_dir + "\\" + letter_file)
        
        #iterate through each sample of the chosen letter
        for letter_frame in os.listdir():
            frame = cv2.imread(letter_frame)
            image, results = mediapipe_detection(frame, hands)
            data.append(extract_hand_keypoints(results))
            labels.append(i)
        
        i += 1
        
#%%

#convert lists to numpy arrays
dataarray = np.asarray(data)
labelarray = np.asarray(labels)

