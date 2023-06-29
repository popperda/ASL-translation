# import the opencv library
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# define a video capture object
video = cv2.VideoCapture(0)
print(video.isOpened())
print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


while(True):
	
	# Capture the video frame
	# by frame
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break    
    

	# Display the resulting frame
   
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
    red  = cv2.cvtColor(frame, cv2.COLOR_LBGR2Lab )
    #cv2.imshow('frame', frame)
    cv2.imshow('red', frame)
    #print(video.get(cv2.CAP_PROP_FPS))
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('1.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        video.set(cv2.CAP_PROP_FPS,10)
        print("HEY!")
    if cv2.waitKey(1) & 0xFF == ord('9'):
        video.set(cv2.CAP_PROP_FPS,90)
        
        

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()
