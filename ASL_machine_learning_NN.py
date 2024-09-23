
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
import mediapipe as mp
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



#%%

#import data
os.chdir("C:\\Users\\sakai\\OneDrive\\Documents\\Year4\\Coding_projects\\ASL_recognition")
data_labels = np.load("ASL_processed_data.npz")
data = data_labels['arr_0']
labels = data_labels['arr_1']

#produce training and testing data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#convert the numpy arrays into torch tensors
x_train = torch.from_numpy(x_train).to(torch.float32)
y_train = torch.from_numpy(y_train).long()
x_test = torch.from_numpy(x_test).to(torch.float32)
y_test = torch.from_numpy(y_test).long()

#%%

#subclass to create the dataset for training
class Data(Dataset):
    def __init__(self):
        self.x = x_train
        self.y = y_train
        self.len = self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

#subclass creating the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 29),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

#%%

#load data
data_set = Data()
trainloader = DataLoader(dataset = data_set, batch_size=32)

#set learning rate
learning_rate = 0.001

#define model, optimizer, and loss function
model = Net()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

#train model
n_epochs = 3
loss_list = []

for epoch in range(n_epochs):
    for x, y in trainloader:
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data)
        
#test model        
z = model(x_test)
yhat = torch.max(z.data,1)
correct = torch.sum(yhat[1] == y_test).float()
percent_correct = 100*correct.numpy()/len(y_test)
print(str(percent_correct) + '%')

#%%

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

#mediapipe detection of hands
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image) #detection using mediapipe
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

#drawing hand landmarks and connections
def draw_hands(image, results):
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), 
                                  mp_drawing_styles.get_default_hand_connections_style())

#extract the coordinates of each landmark as an array
def extract_hand_keypoints(results):
    for hand_landmarks in results.multi_hand_landmarks:
        hands = np.asarray([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z] for i in range(len(hand_landmarks.landmark))]).flatten()
    return hands
    
#find the maximum coordinates for the landmarks to draw a box around the hand on the display
def extract_hands_min_max(results):
    x_coord = []
    y_coord = []
    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_coord.append(x)
            y_coord.append(y)
    x1 = min(x_coord)
    y1 = min(y_coord)
    x2 = max(x_coord)
    y2 = max(y_coord)
    
    return x1, y1, x2, y2

#%%

#labels dictionary
labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'del', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'nothing', 16:'O', 17:'P', 18:'Q', 19:'R', 20:'S', 21:'space', 22:'T', 23:'U', 24:'V', 25:'W', 26:'X', 27:'Y', 28:'Z'}

#live feed showing what letter you're holding up
cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3) as hands:
    while cap.isOpened():
        
        #Read feed
        ret, frame = cap.read()
        H, W, _ = frame.shape
        
        #make detections
        image, results = mediapipe_detection(frame, hands)     
        
        #Draw Landmarks
        if results.multi_hand_landmarks:
            draw_hands(image, results)
            
        #make data and prediction
            data = torch.from_numpy(extract_hand_keypoints(results)).to(torch.float32)
            z = model(data)
            yhat = torch.max(z.data,0)[1]
            predicted_character = labels_dict[int(yhat)]
            print(predicted_character)
        #Show to screen
            x1, y1, x2, y2 = extract_hands_min_max(results)
            x1 = int(x1*W) 
            y1 = int(y1*H) 
            x2 = int(x2*W)
            y2 = int(y2*H) 
            
            cv2.rectangle(image, (x1,y1), (x2, y2), (0,0,0), 4)
            cv2.putText(image, predicted_character, (x1,y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        #breaking gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()









