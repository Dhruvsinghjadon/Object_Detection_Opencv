#Dhruv Singh Jadon
# import the necessary packages
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile
import matplotlib.pyplot as plt

# QR code detector function
def detectQRcode(image):
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode:
        x, y, w, h =obDecoded.rect
        # cv.rectangle(image, (x,y), (x+w, y+h), ORANGE, 4)
        points = obDecoded.polygon
        if len(points) > 4:
            hull = cv.convexHull(
                np.array([points for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        return hull

ref_point = []
click = False
points =()
cap = cv.VideoCapture(0)
_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

lk_params = dict(winSize=(20, 20),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
 
cap = cv.VideoCapture(0)
point_selected = False
points = [()]
old_points = np.array([[]])
qr_detected= False
# stop_code=False

# Create a figure for plotting
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
line_pyzbar, = ax.plot([], [], label='Pyzbar Velocity')
line_optical_flow, = ax.plot([], [], label='Optical Flow Velocity')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Velocity')
ax.legend()

# Initialize variables for velocity calculation for Optical Flow
previous_position = None
previous_time = None
velocity = None

# Initialize variables for velocity calculation for Pyzbar
previous_pyzbar_points = None
previous_pyzbar_time = None
pyzbar_velocity = None

# Create lists to store velocities
pyzbar_velocities = []
optical_flow_velocities = []

frame_counter =0
starting_time =time.time()
# keep looping until the 'q' key is pressed
while True:
    frame_counter +=1
    ret, frame = cap.read()
    img = frame.copy()
    # img = cv.resize(img, None, fx=2, fy=2,interpolation=cv.INTER_CUBIC)
    cv.imshow('old frame ', old_gray)
    cv.imshow('img', img)

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # display the image and wait for a keypress
    clone = frame.copy()
    hull_points =detectQRcode(frame)
    # print(old_points.size)
    stop_code=False
    if hull_points:
        pt1, pt2, pt3, pt4 = hull_points
        qr_detected= True
        stop_code=True
        old_points = np.array((pt1, pt2, pt3, pt4), dtype=np.float32)
        frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.4)
        AiPhile.textBGoutline(frame, f'Detection: Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
        cv.circle(frame, pt1, 10, AiPhile.GREEN, 10)
        cv.circle(frame, pt2, 10, AiPhile.BLUE, 10)
        cv.circle(frame, pt3, 10,AiPhile.YELLOW, 10)
        cv.circle(frame, pt4, 10, AiPhile.RED, 10)
        # Display coordinates next to each circle for Pyzbar
        cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
        cv.circle(frame, pt2, 3, AiPhile.BLUE, 3)
        cv.circle(frame, pt3, 3,AiPhile.YELLOW, 3)
        cv.circle(frame, pt4, 3, AiPhile.RED, 3)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text_offset = 10  # Adjust this value for the distance between the circle and the text

        cv.putText(frame, f'({pt1[0]}, {pt1[1]})', (pt1[0], pt1[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
        cv.putText(frame, f'({pt2[0]}, {pt2[1]})', (pt2[0], pt2[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
        cv.putText(frame, f'({pt3[0]}, {pt3[1]})', (pt3[0], pt3[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
        cv.putText(frame, f'({pt4[0]}, {pt4[1]})', (pt4[0], pt4[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)

       # Calculate velocity for Pyzbar
        current_pyzbar_points = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
        current_pyzbar_time = time.time()

        if previous_pyzbar_points is not None and previous_pyzbar_time is not None:
            displacement = np.linalg.norm(current_pyzbar_points - previous_pyzbar_points)
            time_difference = current_pyzbar_time - previous_pyzbar_time

            # Assuming length of QR code is in centimeters
            qr_code_length_cm = 17.5

            # Convert displacement to centimeters
            displacement_cm = (displacement / frame.shape[1]) * qr_code_length_cm

            # Calculate velocity in cm/s for Pyzbar
            pyzbar_velocity = displacement_cm / time_difference

            # Display velocity for Pyzbar
            cv.putText(frame, f'Pyzbar Velocity: {round(pyzbar_velocity, 2)} cm/s', (30, 150),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        # Update previous position and time for Pyzbar for the next iteration
        previous_pyzbar_points = current_pyzbar_points
        previous_pyzbar_time = current_pyzbar_time                                    
        
    if qr_detected and stop_code==False:
        # print('detecting')
                                 
        
        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_points = new_points
        new_points=new_points.astype(int)
        n = (len(new_points))
        frame =AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.4)
        AiPhile.textBGoutline(frame, f'Detection: Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.GREEN)
        cv.circle(frame, tuple(new_points[0]), 3,AiPhile.GREEN, 2)
        
        # Display coordinates next to the circle for Optical Flow
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text_offset = 10

        cv.putText(frame, f'({new_points[0][0]}, {new_points[0][1]})', (new_points[0][0], new_points[0][1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)

        # Calculate velocity for Optical Flow
        current_position = new_points[0]
        current_time = time.time()

        if previous_position is not None and previous_time is not None:
            displacement = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            time_difference = current_time - previous_time

            # Assuming length of QR code is in centimeters
            qr_code_length_cm = 17.5

            # Convert displacement to centimeters
            displacement_cm = (displacement / frame.shape[1]) * qr_code_length_cm

            # Calculate velocity in cm/s
            velocity = displacement_cm / time_difference

            # Display velocity
            cv.putText(frame, f'Velocity: {round(velocity, 2)} cm/s', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        # Update previous position and time for the next iteration
        previous_position = current_position
        previous_time = current_time        
                # Calculate Pyzbar velocity
        pyzbar_velocity = np.linalg.norm(np.array(pt1) - np.array(old_points[0]))

        # Calculate optical flow velocity
        optical_flow_velocity = np.linalg.norm(new_points[0] - old_points[0])

        # Store velocities
        pyzbar_velocities.append(pyzbar_velocity)
        optical_flow_velocities.append(optical_flow_velocity)
        
              # Update the plot dynamically
        line_pyzbar.set_xdata(range(len(pyzbar_velocities)))
        line_pyzbar.set_ydata(pyzbar_velocities)
        line_optical_flow.set_xdata(range(len(optical_flow_velocities)))
        line_optical_flow.set_ydata(optical_flow_velocities)

        ax.relim()
        ax.autoscale_view()

        # Pause for a short duration to update the plot
        plt.pause(0.01)
    cv.imshow('image', frame)    
    old_gray = gray_frame.copy()
    # press 'r' to reset the window
    key = cv.waitKey(1)
    if key == ord("s"):
        cv.imwrite(f'reference_img/Ref_img{frame_counter}.png', img)

    # if the 'c' key is pressed, break from the loop
    if key == ord("q"):
        break
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)

    
# close all open windows
cv.destroyAllWindows()
cap.release()

# Disable interactive mode before exiting
plt.ioff()

# Plotting velocities
plt.figure()
plt.plot(pyzbar_velocities, label='Pyzbar Velocity')
plt.plot(optical_flow_velocities, label='Optical Flow Velocity')
plt.xlabel('Frame Number')
plt.ylabel('Velocity')
plt.legend()
plt.show()
