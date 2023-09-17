# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 01:40:31 2022

@author: USER
"""

import cv2
import numpy as np
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import messagebox
from tkinter.ttk import Combobox

form = tk.Tk()
form.geometry('500x500')
form.title('Resistor Colour Code Calculator')
pic = ImageTk.PhotoImage(Image.open(r'C:\Users\USER\Desktop\görüntü\resistor.jpg'))
RGB = cv2.imread(r'C:\Users\USER\Desktop\görüntü\resistor.jpg')
RGB_copy = RGB.copy()
label = tk.Label(form, image=pic)
label.pack()

def nothing(x):
    pass



def bwareaopen(image):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 100  
    
    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
            
    return img2


hsv = cv2.cvtColor(RGB, cv2.COLOR_BGR2HSV)
#cv2.namedWindow('T')
#cv2.createTrackbar('thmin', 'T', 0, 255, nothing)
#cv2.createTrackbar('thmax', 'T', 0, 255, nothing)

I_gray = cv2.cvtColor(RGB, cv2.COLOR_BGR2GRAY)
cv2.imshow('T1', I_gray)

#while True:
threshmin =  80 #cv2.getTrackbarPos('thmin', 'T') #80
threshmax =  67 #cv2.getTrackbarPos('thmax', 'T') #67
#print(thresh)

ret1, I_thresh = cv2.threshold(I_gray, threshmin, threshmax, cv2.THRESH_BINARY)
cv2.imshow('T', I_thresh)

I_bw = bwareaopen(I_thresh)
I_bn = I_bw.astype(np.uint8)

cv2.imshow('T3', I_bn)

I_contours,_ = cv2.findContours(I_bn, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


for cnts in I_contours:
  # if the contour is not sufficiently large, ignore it
  if cv2.contourArea(cnts) < 505:
    continue
  # compute the rotated bounding box of the contour
  
  M = cv2.moments(cnts)
  cX = int((M["m10"] / M["m00"]))
  cY = int((M["m01"] / M["m00"]))
  
  b,g,r = cv2.split(hsv)

  b_mean = b[cY][cX]
  g_mean = g[cY][cX]
  r_mean = r[cY][cX]
  
  print(cX, cY, b_mean,g_mean,r_mean)
  
  text1= "brown"
  text2= "green"
  text3= "orange"
  text4= "gold"
  
  values= {"brown" :1, "green":5, "orange": 1000, "gold": 5} 
  

  if b_mean >= 7 and b_mean <= 9 and g_mean >= 77 and g_mean <= 170 and r_mean >=80 and r_mean <= 90:
      cv2.putText(RGB_copy, text1 , (cX -60, cY - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 2)
  elif b_mean >= 70 and b_mean <= 80 and g_mean >= 110 and g_mean <= 130 and r_mean >= 50 and r_mean <= 60:
      cv2.putText(RGB_copy, text2, (cX -10, cY + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 2)
  elif b_mean >= 8 and b_mean <= 20 and g_mean >= 120 and g_mean <= 130 and r_mean >= 150 and r_mean <= 180:
    cv2.putText(RGB_copy, text3, (cX -20, cY - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 2)
  else:
      cv2.putText(RGB_copy, text4, (cX + 20, cY + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5, (0, 0, 0), 2)
      
  

  I_drawcountours = cv2.drawContours(RGB_copy, cnts, -1, (0, 255, 0), 1)
  cv2.imshow('T5', RGB_copy)

      
  #ohm = text1.values + values[1] * values[2], "%" ,values[3]
  #RGB_copy= cv2.putText(RGB_copy, ohm, (cX - 50, cY + 150), cv2.FONT_HERSHEY_SIMPLEX ,1, (255, 0, 0), 2) 
  
  if(cv2.waitKey(1) == 27):
    cv2.destroyAllWindows()



label_4 = tk.Label(form, text='Resistor Colour Code Calculator', bg='black', fg='white', font='verdana 10 bold')
label_4.place(x=10, y=50)

liste=['Black', 'Brown', 'Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'Gray', 'White']
box1= Combobox(form, values=liste)
box1.place(x=10, y=90, width=80)

listem=['Black', 'Brown', 'Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Purple', 'Gray', 'White']
box2= Combobox(form, values=listem)
box2.place(x=100, y=90, width=80)

liste2=['Black', 'Brown', 'Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Gold', 'Silver']
box3= Combobox(form, values=liste2)
box3.place(x=190, y=90, width=80)

liste3=['Brown', 'Red', 'Gold', 'Silver']
box4= Combobox(form, values=liste3)
box4.place(x=280, y=90, width=80)

def Calculate():
    if len(box1.get())==0 or len(box2.get())==0 or len(box3.get())==0 or len(box4.get())==0:
        messagebox.showinfo('Please dont leave blank.', message='Fill in the blanks.')
    else:
        dic1={'Black':0, 'Brown':1, 'Red':2, 'Orange':3, 'Yellow':4, 'Green':5, 'Blue':6, 'Purple':7, 'Gray':8, 'White':9}
        dic2={'Black':1, 'Brown':10, 'Red':100, 'Orange':1000, 'Yellow':10000, 'Green':100000, 'Blue':1000000, 'Gold':0.1, 'Silver':0.01}
        dic3={'Brown':1, 'Red':2, 'Gold':5, 'Silver':10}
        value1 = box1.get()
        value2 = box2.get()
        value3 = box3.get()
        value4 = box4.get()
        
        ohm1 = str(dic1[value1]) + str(dic1[value2]) + 'x' + str(dic2[value3]) + '=' + str(dic1[value1]) + str(dic1[value2]) + str(dic2[value3])[1:] + ' Ohm'
        tolerance1 = '%' + str(dic3[value4])
        
        ohm = tk.Label(form, text='Ohm')
        ohm.place(x=10, y=120)
        label1 = tk.Label(form, text=ohm1)
        label1.place(x=100, y=120)
        
        tolerance = tk.Label(form, text='Tolerance')
        tolerance.place(x=10, y=150)
        label = tk.Label(form, text=tolerance1)
        label.place(x=100, y=150)
        
button1 = tk.Button(form, text='Calculate', command=Calculate)
button1.place(x=380, y=87)

form.mainloop()