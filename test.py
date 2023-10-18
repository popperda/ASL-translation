import cv2

# Open the video capture
cap = cv2.VideoCapture(0)

# Read a frame from the video
ret, frame = cap.read()

# Define the region of interest (ROI) coordinates
x = 100  # starting x-coordinate
y = 100  # starting y-coordinate
width = 200  # width of the region
height = 200  # height of the region

# Crop the frame using the ROI coordinates
cropped_frame = frame[y:y+height, x:x+width]

# Display the cropped frame
cv2.imshow('Cropped Frame', cropped_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture
cap.release()