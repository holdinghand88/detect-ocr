import cv2 
import time
import os

# Video processing....

cap = cv2.VideoCapture('./vid_samples/17.mp4')
# image number reference
frame_rate = 0.4
prev = 0
i = 72
while True:
    time_elapsed = time.time() - prev
    isTrue, frame = cap.read()
    
    cv2.imshow('Video', frame)
    if time_elapsed > 1./frame_rate:
        prev = time.time()                           
        saving_path = f'./img_samples/{i}.jpg'  # Image file directory
        cv2.imwrite(saving_path, frame)
        i += 1
        
    # Escape to terminate the processes...
    if cv2.waitKey(1) > -1:
        break
    

cap.release()
cv2.destroyAllWindows() 