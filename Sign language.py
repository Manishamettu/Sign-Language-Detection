def mediapipe_detection(image, model):

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# COLOR CONVERSION BGR 2 RGB

image.flags.writeable = False

# Image is no Longer writeable

results model.process(image)

image.flags.writeable = True

# Make prediction

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Image is now writeable

return image, results

# COLOR COVERSION RGB 2 BGR

def draw_landmarks (image, results):

mp_drawing.draw_landmarks (image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) mp_drawing.draw_landmarks (image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

1011

# Draw face connections # Draw pose connections

mp_drawing.draw_landmarks (image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw Left hand connections mp_drawing.draw_landmarks (image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def extract_keypoints (results):

pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

1h = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) return np.concatenate ([pose, face, 1h, rh])
cap = cv2.VideoCapture(0)

#Set mediapipe model

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

#Loop through actions for action in actions:

#Loop through sequences aka videos

for sequence in range(1, no_sequences+1):

#Loop through video Length aka sequence Length for frame_num in range(sequence_length):

#Read feed

ret, frame cap.read()

if not ret: break

#Make detections image, results mediapipe_detection (frame, holistic)

#Draw Landmarks draw_styled_landmarks(image, results)

#walt Logic

if frame num

cv2.putText(image, STARTING COLLECTION', (120,200), CV2. FONT HERSHEY S HERSHEY _SIMPLEX, 1, (0,255, 0), 4, 1 CV2.LINE AA)

cv2.putText(image, Collecting frames for () Video Number ().format(action, sequence), (15,12), CV2 FONT HERSHEY SIMPLEX, 0.5, (0, 0, 255), 1, CV2.LINE_AA)

# show to screen cv2.imshow('OpenCV Feed", image)

cv2.waitKey(500)

else: cv2.putText(image, Collecting frames for () Video Number ()'.format(action, sequence), (15,12), CV2. FONT HERSHEY SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE AA)

#show to screen cv2.imshow("OpenCV Feed', image)

#NEW Export keypoints

keypoints extract keypoints(ultetion, npy pathos.path.join(DATA PATH, str(sequence), str(frame_num))

np.save(npy_path, keypoints)

gracefully #Break gracefu 1f cv2.waitKey(10) & OXFFord('q'):

break

cap.release() cv2.destroyAllwindows()
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))

model.add(LSTM(128, return_sequences=True, activation='relu'))

model.add(LSTM (64, return_sequences=False, activation='relu'))

model.add(Dense (64, activation='relu')) model.add(Dense (32, activation='relu'))

model.add(Dense(actions.shape[0], activation='softmax'))
sequence

8

sentence

predictions 11 threshold

cap cv2.videocapture(0)

with mp holistic.Holistic(min_detection_confidenceso.5, min_tracking confidencevo.5) as holistic:

while cap.isOpened():

ret, frame cap.read()

image, results mediapipe detection(frame, holistic)

print(results)

draw_styled_landmarks (image, results)

keypoints extract keypoints(results) sequence.append(keypoints)

sequence sequence[-301]

if len(sequence) 30: res model.predict(np.expand_dims(sequence, axis=0)) [0]

print(actions[op.argmax(res)])

predictions.append(np.argmax(res))

if np.unique(predictions[-10:])[0]==np.argmax(res): if res(np.argmax(res)] > thresholdi

if len(sentence) 0:

if actions[np.argmax(res)] Is sentence[-1]: sentence.append(actions(np.argmax(res)])

else: sentence.append(actions[np.argmax(res)])

if len(sentence) > 5: sentence sentence[-5:]

image prob viz(res, actions, image, colors)

cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1) cv2.putText(image, Join(sentence), (3,30),

2. FONT HERSHEY SIMPLEX, 1, (255, 255, 255), 2. cv2.LINE_AA)

#show to screen

cv2.imshow(OpenCV Feed', Image)

Break gracefully

if cv2.waitKey(10) & 0XFFord('q'):

break

cap.release()

cv2.destroyAllWindows()