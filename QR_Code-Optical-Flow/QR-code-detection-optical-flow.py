# Dhruv Singh Jadon
# import the necessary packages
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
#import time
import AiPhile
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk

# Physical size of the QR code in centimeters
qr_code_length_cm = 25

# Create a separate window for displaying velocities
cv.namedWindow('Velocities', cv.WINDOW_NORMAL)
cv.resizeWindow('Velocities', 400, 200)  # Adjust the size as needed


# Initialize and store variables for ploting

velocities_x = [] # actual calculated velocity
velocities_y = [] # actual calculated velocity
velocities_z = [] # actual calculated velocity

max_slew_rate = 5.0  # Set your desired maximum slew rate in cm/s
filtered_velocities_x = [0]  # Initial velocity value for slew rate
filtered_velocities_y = [0]  # Initial velocity value for slew rate
filtered_velocities_z = [0]  # Initial velocity value for slew rate

smoothed_velocities_x = [] # averaging velocity value for ploting
smoothed_velocities_y = [] # averaging velocity value for ploting
smoothed_velocities_z = [] # averaging velocity value for ploting
timestamps = []

# Initialize variables for velocity (n and n-1)

#prev_time = time.time()


prev_coordinates_pyzbar = None
prev_coordinates_optical_flow = None
prev_distance = None

# Smoothing window size for smoothing/averaging
window_size = 5

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

# Optical flow intialization
lk_params = dict(winSize=(20, 20),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
 
cap = cv.VideoCapture(0)
point_selected = False
points = [()]
old_points = np.array([[]])
qr_detected= False
# stop_code=False



# Frame counter and time initialization
frame_counter =0
#starting_time =time.time()

# keep looping until the 'q' key is pressed
while True:
    frame_counter +=1
    ret, frame = cap.read()
    img = frame.copy()
    # img = cv.resize(img, None, fx=2, fy=2,interpolation=cv.INTER_CUBIC)       
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # display the image and wait for a keypress
    clone = frame.copy()
    
    # QR code detectionq
    hull_points =detectQRcode(frame)
    
    # pyzbar detection
    stop_code=False
    if hull_points:
        pt1, pt2, pt3, pt4 = hull_points
        qr_detected= True
        stop_code=True
        old_points = np.array((pt1, pt2, pt3, pt4), dtype=np.float32)
        frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.4)
        #AiPhile.textBGoutline(frame, f'Detection using Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
        
        
        cw_p = pt4[0]-pt1[0]
        
        # Finding the Focal Length
        W2 = 25
        
        # Calculate distance based on apparent width
        width = np.sqrt((old_points[-1][0] - old_points[0][0])**2 + (old_points[-1][1] - old_points[0][1])**2)
        # focal lenght
        f = 620
        new_distance = (W2 * f) / width
        
        d2 =(W2*f)/cw_p
        d2=d2*0.01
       
        # Display coordinates next to each circle for Pyzbar
        cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
        cv.circle(frame, pt2, 3, AiPhile.BLUE, 3)
        cv.circle(frame, pt3, 3,AiPhile.YELLOW, 3)
        cv.circle(frame, pt4, 3, AiPhile.RED, 3)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text_offset = 7  # Adjust this value for the distance between the circle and the text
        
        
        # In 1 pixels how much cm it cover in pyzbar
        conversion_factor_px_to_cm_pyzbar = qr_code_length_cm/(pt4[0]-pt1[0])
        cw_p = pt4[0]-pt1[0]
        
        # Finding the Focal Length
        W = 14.5
        #d = 100
        #f_p=(cw_p*d)/W
        #print("Focal lenght at pyzbar")    
        #print(f_p)  
            
        # Distance at Z-axis
        f = 840
        d =(W*f)/cw_p
        d=d*0.01
        d= -0.02573*(d)**2  + (1.48168)*d - 0.30833
        
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0,0,255)  
        font_thickness = 2

        focal_length_text = f"Distance Extimation: {d:.2f} m"
        cv.putText(frame, focal_length_text, (10, 30), font, font_scale, font_color, font_thickness)
        
        focal_length_text = f"Distance Extimation: {d2:.2f} m"
        cv.putText(frame, focal_length_text, (10, 60), font, font_scale, font_color, font_thickness)
        
        if prev_coordinates_pyzbar is not None and prev_distance is not None :
            
            # error
            b_a_x = 0.5*640*(new_distance/prev_distance-1)
            b_a_y = 0.5*480*(new_distance/prev_distance-1)
            
            #apply to all x
            old_points[0][0]=old_points[0][0]-b_a_x
            old_points[1][0]=old_points[1][0]-b_a_x
            old_points[2][0]=old_points[2][0]-b_a_x
            old_points[3][0]=old_points[3][0]-b_a_x
            
            #apply to all y
            old_points[0][1]=old_points[0][1]-b_a_y
            old_points[1][1]=old_points[1][1]-b_a_y
            old_points[2][1]=old_points[2][1]-b_a_y
            old_points[3][1]=old_points[3][1]-b_a_y

            
                    
            # Displaying coordinates value
            cv.putText(frame, f'({pt1[0]}, {pt1[1]})', (pt1[0], pt1[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
            cv.putText(frame, f'({pt2[0]}, {pt2[1]})', (pt2[0], pt2[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
            cv.putText(frame, f'({pt3[0]}, {pt3[1]})', (pt3[0], pt3[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
            cv.putText(frame, f'({pt4[0]}, {pt4[1]})', (pt4[0], pt4[1] - text_offset),
                   font, font_scale, (0,0,0), font_thickness)
            
            change_in_distance = new_distance - prev_distance
            change_in_time = 1/30
            velocities_Z = change_in_distance / change_in_time
            
            displacement_X_pyzbar = old_points[0][0] - prev_coordinates_pyzbar[0][0]
            displacement_X_pyzbar=displacement_X_pyzbar*conversion_factor_px_to_cm_pyzbar
            time_difference_pyzbar = 1/30
            
            displacement_Y_pyzbar = old_points[0][1] - prev_coordinates_pyzbar[0][1]
            displacement_Y_pyzbar=displacement_Y_pyzbar*conversion_factor_px_to_cm_pyzbar
            
            change_in_distance = new_distance - prev_distance
            change_in_time = 1/30
            velocities_Z = change_in_distance / change_in_time
            
            
            velocities_X = (displacement_X_pyzbar / time_difference_pyzbar)
            velocities_Y = (displacement_Y_pyzbar / time_difference_pyzbar)
            #print("This is Pyzbar")
            #print(velocities_X)
            #print(velocities_Y)
            
            #cw = old_points[0][0] - prev_coordinates_pyzbar[0][0]            
         
            # Append data for plotting
            timestamps.append(frame_counter/30)
            velocities_x.append(velocities_X)
            velocities_y.append(velocities_Y)
            velocities_z.append(velocities_Z)
            
            
            # Display velocities in a separate window
            velocities_text1 = f'Velocities X (cm/s) = {velocities_X:.2f}'
            velocities_text2 = f'Velocities Y (cm/s) = {velocities_Y:.2f}'
            velocities_text3 = f'Velocities Z (cm/s) = {velocities_Z:.2f}'
            
            
            # Display velocities in the separate 'Velocities' window
            velocities_window = np.ones((200, 400, 3), dtype=np.uint8) * 0  # White background
            cv.putText(velocities_window, velocities_text1, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv.putText(velocities_window, velocities_text2, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)    
            cv.putText(velocities_window, velocities_text3, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv.imshow('Velocities', velocities_window)
            
            # Slew rate limiting
            delta_t = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 0.0
            
            filtered_velocity_y = filtered_velocities_y[-1] + np.clip(velocities_Y - filtered_velocities_y[-1], -max_slew_rate * delta_t, max_slew_rate * delta_t)
            filtered_velocity_x = filtered_velocities_x[-1] + np.clip(velocities_X - filtered_velocities_x[-1], -max_slew_rate * delta_t, max_slew_rate * delta_t)
            filtered_velocity_z = filtered_velocities_z[-1] + np.clip(velocities_Z - filtered_velocities_z[-1], -max_slew_rate * delta_t, max_slew_rate * delta_t)
            #print(filtered_velocity_x)
            
             
            filtered_velocities_x.append(filtered_velocity_x)
            filtered_velocities_y.append(filtered_velocity_y)
            filtered_velocities_z.append(filtered_velocity_z)
            #timestamps.append(1/30)
            
            
            
            # Apply moving average smoothing and Plotting smoothed velocity
            if len(filtered_velocities_x) >= window_size:
                smoothed_x = np.mean(filtered_velocities_x[-window_size:])
                smoothed_y = np.mean(filtered_velocities_y[-window_size:])
                smoothed_z = np.mean(filtered_velocities_z[-window_size:])
                smoothed_velocities_x.append(smoothed_x)
                smoothed_velocities_y.append(smoothed_y)
                smoothed_velocities_z.append(smoothed_z)
                
        
        
        prev_distance = new_distance
        prev_time=frame_counter/30
        prev_coordinates_pyzbar = old_points
                                   
        
    if qr_detected and stop_code==False:
        # print('detecting')
                                 

        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_points = new_points
        new_points=new_points.astype(int)
        n = (len(new_points))
        frame =AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.4)
        #AiPhile.textBGoutline(frame, f'Detection using Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.GREEN)

        # Calculate distance based on apparent width
        width = np.sqrt((new_points[-1][0] - new_points[0][0])**2 + (new_points[-1][1] - new_points[0][1])**2)
        f = 840
        
        W2 = 25
        f2=630
        
        new_distance = (W2 * f2) / width

        # Calculate the apparent width based on the distance between the first and last points
        if len(new_points) >= 2:
            width = np.sqrt((new_points[-1][0] - new_points[0][0])**2 + (new_points[-1][1] - new_points[0][1])**2)
            #print(f'Apparent Width: {width}')
            
            W2 = 25
            f2=620
            
            f = 840
            d =(W*f)/width
            d=d*0.01
            d= -0.02573*(d)**2  + (1.48168)*d - 0.30833
                  
            d2=(W2*f2)/width
            
            conversion_factor_px_to_cm_optical = qr_code_length_cm/(width)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0,0,255)  
            font_thickness = 2

            focal_length_text = f"Distance Extimation: {d:.2f} m"
            cv.putText(frame, focal_length_text, (10, 30), font, font_scale, font_color, font_thickness)
            
            focal_length_text = f"Distance Extimation: {d2:.2f} m"
            cv.putText(frame, focal_length_text, (10, 60), font, font_scale, font_color, font_thickness)
        
        if prev_coordinates_optical_flow is not None and prev_distance is not None :
            
            
            # error
            #b_a=int(new_distance/prev_distance) - 1
            #new_points = new_points #+ b_a
            #b_a_x = 0.5*1920*(new_distance/prev_distance-1)
            #b_a_y = 0.5*1080*(new_distance/prev_distance-1)
            
            
            #apply to all x
            old_points[0][0]=old_points[0][0]-b_a_x
            old_points[1][0]=old_points[1][0]-b_a_x
            old_points[2][0]=old_points[2][0]-b_a_x
            old_points[3][0]=old_points[3][0]-b_a_x
            
            #apply to all y
            old_points[0][1]=old_points[0][1]-b_a_y
            old_points[1][1]=old_points[1][1]-b_a_y
            old_points[2][1]=old_points[2][1]-b_a_y
            old_points[3][1]=old_points[3][1]-b_a_y


            
            
            displacement_X_optical_flow = old_points[0][0] - prev_coordinates_optical_flow[0][0]
            displacement_X_optical_flow=displacement_X_optical_flow*conversion_factor_px_to_cm_optical
            time_difference_optical_flow = 1/30
            
            
            
            displacement_Y_optical_flow = old_points[0][1] - prev_coordinates_optical_flow[0][1]
            displacement_Y_optical_flow=displacement_Y_optical_flow*conversion_factor_px_to_cm_optical
            
            change_in_distance = new_distance - prev_distance
            change_in_time = 1/30
            
            velocities_X = (displacement_X_optical_flow / time_difference_optical_flow)
            velocities_Y = (displacement_Y_optical_flow / time_difference_optical_flow)
            
            velocities_Z = change_in_distance / change_in_time
            
            # Append data for plotting
            timestamps.append(frame_counter/30)   
            velocities_x.append(velocities_X)
            velocities_y.append(velocities_Y)
            velocities_z.append(velocities_Z)
            
            # Display velocities in a separate window
            velocities_text1 = f'Velocities X (cm/s) = {velocities_X:.2f}'
            velocities_text2 = f'Velocities Y (cm/s) = {velocities_Y:.2f}'
            velocities_text3 = f'Velocities Z (cm/s) = {velocities_Z:.2f}'
            
            
            # Display velocities in the separate 'Velocities' window
            velocities_window = np.ones((200, 400, 3), dtype=np.uint8) * 0  # White background
            cv.putText(velocities_window, velocities_text1, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv.putText(velocities_window, velocities_text2, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)    
            cv.putText(velocities_window, velocities_text3, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv.imshow('Velocities', velocities_window)
            
            
            # Slew rate limiting
            delta_t = timestamps[-1] - timestamps[-2] if len(timestamps) > 1 else 0.0
            filtered_velocity_x = filtered_velocities_x[-1] + np.clip(velocities_X - filtered_velocities_x[-1], -max_slew_rate * delta_t, max_slew_rate * delta_t)
            filtered_velocity_y = filtered_velocities_y[-1] + np.clip(velocities_Y - filtered_velocities_y[-1], -max_slew_rate * delta_t, max_slew_rate * delta_t)
            filtered_velocity_z = filtered_velocities_z[-1] + np.clip(velocities_Z - filtered_velocities_z[-1], -max_slew_rate * delta_t, max_slew_rate * delta_t)
            
            filtered_velocities_x.append(filtered_velocity_x)
            filtered_velocities_y.append(filtered_velocity_y)
            filtered_velocities_z.append(filtered_velocity_z)
            #timestamps.append(1/30)
            
            # Apply moving average smoothing and Plotting smoothed velocity
            if len(filtered_velocities_x) >= window_size:
                smoothed_x = np.mean(filtered_velocities_x[-window_size:])
                smoothed_y = np.mean(filtered_velocities_y[-window_size:])
                smoothed_z = np.mean(filtered_velocities_z[-window_size:])
                smoothed_velocities_x.append(smoothed_x)
                smoothed_velocities_y.append(smoothed_y)
                smoothed_velocities_z.append(smoothed_z)
                
        prev_distance = new_distance    
        prev_time=frame_counter/30
        prev_coordinates_optical_flow = old_points
            
             
        
        for i, new_point in enumerate(new_points):
            cv.circle(frame, tuple(new_point), 10, (0,0,255), 10)
        
        # Display coordinates next to the circle for Optical Flow
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2

            cv.putText(frame, f'({new_point[0]}, {new_point[1]})', (new_point[0], new_point[1] - text_offset),
                       font, font_scale, (0, 0, 0), font_thickness)       

    cv.imshow('QR Code Detecting', frame)  
    

    old_gray = gray_frame.copy()
    # press 'r' to reset the window
    key = cv.waitKey(1)
    if key == ord("s"):
                    cv.imwrite(f'reference_img/Ref_img{frame_counter}.png', img)

    # if the 'c' key is pressed, break from the loop
    if key == ord("q"):
        break
    fps = frame_counter/(30)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("QR Code Detecting", frame)
    
# close all open windows
cv.destroyAllWindows()

cap.release()

#Plotting velocity data
plt.figure(figsize=(10, 6))
#plt.plot(timestamps, velocities_x, label='Vx', alpha=0.5)
#plt.plot(timestamps, velocities_y, label='Vy', alpha=0.5).11
#plt.plot(timestamps, velocities_y, label='Vz', alpha=0.5)

plt.grid(True)
#Smoothed velocities
timestamps = timestamps[:len(smoothed_velocities_x)]
plt.plot(timestamps, smoothed_velocities_x, label=f'Vx')
plt.plot(timestamps, smoothed_velocities_y, label=f'Vy')
plt.plot(timestamps, smoothed_velocities_z, label=f'Vz')


#plt.plot(timestamps, filtered_velocities_x, label='Vx Slew Rate Limiting')
#plt.plot(timestamps, filtered_velocities_y, label='Vy Slew Rate Limiting')
#plt.plot(timestamps, filtered_velocities_z, label='Vz Slew Rate Limiting')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (cm/s)')
plt.title('Velocity XYZ Over Time')
plt.legend()
plt.show()
