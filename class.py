import cv2
import mediapipe as mp
import math
import numpy as np
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

min_radius = 2   # min rad for circle
max_radius = 250  # max rad for circle

# some distance bwtween two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)

# wavy fucking lines
def draw_wavy_lines(image, point1, point2, amplitude=4, frequency=1, thickness=3):
    x1, y1 = point1
    x2, y2 = point2
    length = calculate_distance(x1, y1, x2, y2)
    
    # some no. of segment
    num_segments = int(length / 5)  #so 5 per segments 
    for i in range(num_segments):
        t = i / num_segments
        x = int((1 - t) * x1 + t * x2)
        y = int((1 - t) * y1 + t * y2 + amplitude * math.sin(frequency * t * 2 * np.pi))
        
        # adjust the thickness and color (no black and fat)
        if i > 0:
            cv2.line(image, (prev_x, prev_y), (x, y), (255, 255, 255), thickness=2)
        prev_x, prev_y = x, y


previous_index_x = None
swipe_threshold = 40  # least movement for detection of ur hand

# Variables for X signs
x_positions = [(0, 0) for _ in range(4)]  # no. of x signs
move_interval = 2  # Frames between movements
move_counter = 100  # idk wht the fuck is this for 

while True:
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    results = hands.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    center_x, center_y, radius = None, None, None
    gesture_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 6, (255, 255, 255), 1)  # Small white circles 
            
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_landmark = hand_landmarks.landmark[start_idx]
                end_landmark = hand_landmarks.landmark[end_idx]
                
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                draw_wavy_lines(image, start_point, end_point, thickness=3)  # wavy lines with ur wish of fatness
            
            # Get the x-coordinate of the index finger tip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            index_x = int(index_tip.x * w)

            # Detect the hand moving left ad right
            if previous_index_x is not None:
                if index_x - previous_index_x > swipe_threshold:
                    gesture_text = "Swipe Right"
                elif previous_index_x - index_x > swipe_threshold:
                    gesture_text = "Swipe Left"

            previous_index_x = index_x
            
            # Get the coordinates of the base of index finger (landmark 6) and base of thumb (landmark 3)
            index_base = hand_landmarks.landmark[6]
            thumb_base = hand_landmarks.landmark[3]

            index_x_base, index_y_base = int(index_base.x * w), int(index_base.y * h)
            thumb_x, thumb_y = int(thumb_base.x * w), int(thumb_base.y * h)

            distance = calculate_distance(index_x_base, index_y_base, thumb_x, thumb_y)
            
            radius = int(min_radius + (distance / 3))  # Smaller scaling factor to minimize size increase
            radius = min(max(radius, min_radius), max_radius)  
            
            center_x = int((index_x_base + thumb_x) / 2)
            center_y = int((index_y_base + thumb_y) / 2)

            # landmarks of the E(x)'s
            landmark_0 = hand_landmarks.landmark[0]
            landmark_5 = hand_landmarks.landmark[5]
            landmark_9 = hand_landmarks.landmark[9]
            landmark_17 = hand_landmarks.landmark[17]
            landmark_0_x = int(landmark_0.x * w)
            landmark_0_y = int(landmark_0.y * h)
            landmark_5_x = int(landmark_5.x * w)
            landmark_5_y = int(landmark_5.y * h)
            landmark_9_x = int(landmark_9.x * w)
            landmark_9_y = int(landmark_9.y * h)
            landmark_17_x = int(landmark_17.x * w)
            landmark_17_y = int(landmark_17.y * h)

            # to change the position of the x so its gay and straight
            random_offset_0 = (random.randint(-10, 10), random.randint(-10, 10))
            random_offset_5 = (random.randint(-10, 10), random.randint(-10, 10))
            random_offset_9 = (random.randint(-10, 10), random.randint(-10, 10))
            random_offset_17 = (random.randint(-10, 10), random.randint(-10, 10))

            # Ssame but more gay (increase horizontal offset)
            x_positions[0] = (landmark_0_x + 50 + random_offset_0[0], landmark_0_y + random_offset_0[1])  # Increased offset
            x_positions[1] = (landmark_5_x + 100 + random_offset_5[0], landmark_5_y + random_offset_5[1])
            x_positions[2] = (landmark_9_x + 100 + random_offset_9[0], landmark_9_y + random_offset_9[1])
            x_positions[3] = (landmark_17_x + 50 + random_offset_17[0], landmark_17_y + random_offset_17[1])

            # Get the coordinates of landmarks 4, 8, 12, 16, and 20
            landmark_8 = hand_landmarks.landmark[8]
            landmark_12 = hand_landmarks.landmark[20]
            landmark_16 = hand_landmarks.landmark[12]
            landmark_20 = hand_landmarks.landmark[16]

            points = [
                (int(landmark_8.x * w), int(landmark_8.y * h)),
                (int(landmark_12.x * w), int(landmark_12.y * h)),
                (int(landmark_16.x * w), int(landmark_16.y * h)),
                (int(landmark_20.x * w), int(landmark_20.y * h)),
            ]

            # Connect wavy lines with less fatness
            for i in range(len(points) - 1):
                draw_wavy_lines(image, points[i], points[i + 1], amplitude=4, frequency=1, thickness=10)

    if center_x is not None and center_y is not None and radius is not None:
        circle_color = (255, 255, 255)  # Fixed white color (BGR format)
        cv2.circle(image, (center_x, center_y), radius, circle_color, 2)  # Draw the circle border with thickness 2

    cv2.putText(image, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for i, (x, y) in enumerate(x_positions):
        cv2.putText(image, "X", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
