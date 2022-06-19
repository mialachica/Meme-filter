import cv2, time
import numpy as np
import random

# Opens the camera 
video = cv2.VideoCapture(1)

# Creates a list of different meme captions
memes_titles = ["Me when I'm sad: ", "Me when I see cake: ", "silly and goofy", "My face when I get an A+:, ", "My Face when I failed my test:" ]

# Creates the ransom selection of which meme title
user_write = random.choice(memes_titles)

# setup text
font = cv2.FONT_HERSHEY_COMPLEX

# get boundary of this text
textsize = cv2.getTextSize(user_write, font, 1, 2)[0]

# Read logo and resize
logo = cv2.imread('Meme_logo.png')
size = 200
logo = cv2.resize(logo, (size, size))

# Create a mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
check, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

while True:
    check, frame = video.read()
    
    # Flips the camera
    camera = cv2.flip(frame, 1)

    # Inputs the facial detection to the code
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Sets the scale for the face dection to be able to read the camera screen
    faces =  face_cascade.detectMultiScale(frame,
                                            scaleFactor = 1.05,
                                            minNeighbors=5)
    
    # Sets different variables to the size of the faces decected
    for x, y, w, h in faces:
        
        # Puts the text onto the users face
        cv2.putText(camera, user_write, (x, y), font, 1, (255, 255, 255), 2)

    roi = camera[-size-10:-10, -size-10:-10]
    
    # Set an index of where the mask is
    roi[np.where(mask)] = 0
    roi += logo

    # Creates the name for the camera window and displays the program in use as a filter 
    cv2.imshow('Meme face', camera)

    # Allows for the 'q' key to close the program and for the 's' key to screenshot a picture of the user using the filter
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('s'):
            cv2.imwrite("MyMemeFace.jpg", camera)

# releases the programs access to the camera
video.release()

# Closes all the windows and shuts off the entire program
cv2.destroyAllWindows