from PIL import Image
import cv2
import numpy as np
import requests

video_file = cv2.VideoCapture('cars2.mp4')

total = 0
number = 0

# Loop through the frames of the video
while True:
    # Read the next frame from the video
    ret, frame = video_file.read()
    
    if not ret:
        break
    
    #output = "result/car" + str(number) + ".png"
    number += 1
    #cv2.imwrite(output, frame)

    # Check if we have reached the end of the video
    #if not ret:
        #break

    # Do some processing on the frame (e.g., detect vehicles)
    # ..
    image = Image.fromarray(frame)
    image = image.resize((450,250))
    image_arr = np.array(image)
    image

    grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
    Image.fromarray(grey)

    blur = cv2.GaussianBlur(grey,(5,5),0)
    Image.fromarray(blur)

    dilated = cv2.dilate(blur,np.ones((3,3)))
    Image.fromarray(dilated)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
    Image.fromarray(closing)

    car_cascade_src = 'cars.xml'
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(closing, 1.1, 20)
    cars

    cnt = 0
    for (x,y,w,h) in cars:
      cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
      cnt += 1
    print(cnt, " cars found")
    Image.fromarray(image_arr)
    total += cnt
	    
    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

	  
# Release the video file and close the window
print("total of " ,  total , " cars crossed form here")
video_file.release()
cv2.destroyAllWindows()

