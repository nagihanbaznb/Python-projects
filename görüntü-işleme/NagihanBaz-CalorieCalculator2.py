#171805024-Nagihan Baz

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 00:50:56 2023

@author: USER
"""

import cv2

def detect_fruit(cap):
    
    hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    edges = cv2.Canny(hsv, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)
    if area > 1000:
        shape = "big"
    elif area > 500:
        shape = "medium"
    else:
        shape = "small"

    mean_color = cv2.mean(cap)
    color = None
    if mean_color[2] < 20 or mean_color[2] > 170:
        color = "red"
    elif mean_color[1] > 30 and mean_color[1] < 90:
        color = "green"
    else:
        color = "yellow"
        
        """
    for contour in cv2.findContours(hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]:
        #contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #(x, y, w, h) = cv2.boundingRect(contour.astype(np.int))
        # Approximate the contour and calculate its bounding box
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # Check if the contour has the shape of a banana (convex shape)
        if cv2.isContourConvex(approx):
            shapef = "banana"
        elif aspect_ratio >= 0.98 and aspect_ratio <= 1.02:
            shapef = "apple"
        else:
            shapef = "orange"
                """
 
    calories = {
        "red_apple": 64,
        "green_apple": 95,
        "banana": 88,
        "orange": 47,
        "strawberry": 6,
        "grape": 67,
        #muz
        "yellow_big":88,
        "yellow_small":88,
        #portakal ve elma
        "red_small":47,
    }
   
    fruit_type = color + "_" + shape
    if fruit_type.lower() in calories:
        calorie = calories.get(fruit_type.lower())
    else:
        calorie = "Calorie count not found"
        
       
                
    return shape,color,calorie #,shapef


cap = cv2.VideoCapture(0)
while True:
   
    ret, frame = cap.read()
    shape, color, calorie = detect_fruit(frame)
    cv2.putText(frame, "Shape: " + shape, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Color: " + color, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Calorie: " + str(calorie), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Fruit Calorie Calculator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#print("Fruit: ",shapef)
print("Shape: ",shape)
print("Color: ",color)
print("Calorie: ",calorie)

#img = cv2.resize(img, (400, 600))
#cv2.imshow("Fruit", img)
#cv2.waitKey(0)