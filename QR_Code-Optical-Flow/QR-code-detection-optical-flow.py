# import the necessary packages
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile
import matplotlib.pyplot as plt

# Physical size of the QR code in centimeters
qr_code_length_cm = 17.5

# Lists to store velocity data for plotting
timestamps = []
velocities_x = []
velocities_y = []


# Initialize variables for velocity calculation
prev_time = time.time()
prev_coordinates_pyzbar = None
prev_coordinates_optical_flow = None


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
starting_time =time.time()

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
    # print(old_points)
    stop_code=False
    if hull_points:
        pt1, pt2, pt3, pt4 = hull_points
        qr_detected= True
        stop_code=True
        old_points = np.array((pt1, pt2, pt3, pt4), dtype=np.float32)
        frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.4)
        AiPhile.textBGoutline(frame, f'Detection using Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
        
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
        # In 1 pixels how much cm we cover
        conversion_factor_px_to_cm_pyzbar = qr_code_length_cm/(pt4[0]-pt1[0])
        #print(pt4[0]-pt1[0])
        
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
        
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0,0,255)  
        font_thickness = 2

        focal_length_text = f"Distance Extimation: {d:.2f} cm"
        cv.putText(frame, focal_length_text, (10, 30), font, font_scale, font_color, font_thickness)
        
        if prev_coordinates_pyzbar is not None:
            displacement_X_pyzbar = old_points[0][0] - prev_coordinates_pyzbar[0][0]
            displacement_X_pyzbar=displacement_X_pyzbar*conversion_factor_px_to_cm_pyzbar
            time_difference_pyzbar = (time.time() - prev_time)
            
            displacement_Y_pyzbar = old_points[0][1] - prev_coordinates_pyzbar[0][1]
            displacement_Y_pyzbar=displacement_Y_pyzbar*conversion_factor_px_to_cm_pyzbar
            
            
            
            velocities_X = (displacement_X_pyzbar / time_difference_pyzbar)
            velocities_Y = (displacement_Y_pyzbar / time_difference_pyzbar)
            #print("This is Pyzbar")
            #print(velocities_X)
            #print(velocities_Y)
            
            #cw = old_points[0][0] - prev_coordinates_pyzbar[0][0]
        
             
            
            # Append data for plotting
            timestamps.append(time.time() - starting_time)
            velocities_x.append(velocities_X)
            velocities_y.append(velocities_Y)
            
        prev_time=time.time()
        prev_coordinates_pyzbar = old_points
                                   
        
    if qr_detected and stop_code==False:
        # print('detecting')
                                 

        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_points = new_points
        new_points=new_points.astype(int)
        n = (len(new_points))
        frame =AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.4)
        AiPhile.textBGoutline(frame, f'Detection using Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.GREEN)
        
        # Calculate the apparent width based on the distance between the first and last points
        if len(new_points) >= 2:
            width = np.sqrt((new_points[-1][0] - new_points[0][0])**2 + (new_points[-1][1] - new_points[0][1])**2)
            #print(f'Apparent Width: {width}')
            f = 840
            d =(W*f)/width
        
            conversion_factor_px_to_cm_optical = qr_code_length_cm/(width)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0,0,255)  
            font_thickness = 2

            focal_length_text = f"Distance Extimation: {d:.2f} cm"
            cv.putText(frame, focal_length_text, (10, 30), font, font_scale, font_color, font_thickness)
        
        
        #cw_o = pt4[0]-pt1[0]
        # Finding the Focal Length
        #W = 14.5
        #d = 150
        #f_o=(cw_p*d)/W
        
        
        # Distance at Z-axis
       
        
        #print(d)
        
        #print("Focal lenght at optical flow")
        #print(f_o)  
        
        if prev_coordinates_optical_flow is not None:
            displacement_X_optical_flow = old_points[0][0] - prev_coordinates_optical_flow[0][0]
            displacement_X_optical_flow=displacement_X_optical_flow*conversion_factor_px_to_cm_optical
            time_difference_optical_flow = (time.time() - prev_time)
            velocities_X = (displacement_X_optical_flow / time_difference_optical_flow)
            
            
            displacement_Y_optical_flow = old_points[0][1] - prev_coordinates_optical_flow[0][1]
            displacement_Y_optical_flow=displacement_Y_optical_flow*conversion_factor_px_to_cm_optical
            
            
            velocities_Y = (displacement_Y_optical_flow / time_difference_optical_flow)
            #print("This is optical flow")
            #print(velocities_X)
            #print(velocities_Y)
            
            # Append data for plotting
            timestamps.append(time.time() - starting_time)   
            velocities_x.append(velocities_X)
            velocities_y.append(velocities_Y)
            
            
            
        prev_time=time.time()
        prev_coordinates_optical_flow = old_points
            
             
        
        for i, new_point in enumerate(new_points):
            cv.circle(frame, tuple(new_point), 10, (0,0,255), 10)
        
        # Display coordinates next to the circle for Optical Flow
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_offset = 10

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
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("QR Code Detecting", frame)

    
# close all open windows
cv.destroyAllWindows()
cap.release()


#Plotting velocity data
plt.figure(figsize=(10, 6))
plt.plot(timestamps, velocities_x, label='Vx')
plt.plot(timestamps, velocities_y, label='Vy')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity (cm/s)')
plt.title('Velocity XY Over Time')
plt.legend()
plt.show()

#ploting real time data smoothing filter
