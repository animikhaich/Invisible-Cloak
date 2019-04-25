# Harry Potter's Invisible Cloak

This is simply a program that uses color segmentation and a bit of image processing to replace the foreground object of interest with the background.

You can take a unicolor bedsheet and apply color segmentation on it. After that take a picture of the background and place the bedsheet over yourself to virtually disappear, or make some part of your body disappear.

<img align="center" src="Assets/output.png" alt="Invisible Cloak">

## Dependencies:
  1. Python 3
  2. Numpy
  3. OpenCV (Python)
 
## Installation for Python 3
  Visit: https://www.python.org/downloads/ and download the version suitable for your System
  
## Installation for Numpy
  Open Command Prompt (Windows) or Terminal (Linux) and type in the following command:
     
     pip install numpy
  
  Visit: https://scipy.org/install.html for details.
  
 ## Installation for OpenCV
  Open Command Prompt (Windows) or Terminal (Linux) and type in the following command:
     
     pip install opencv-python
  
  Visit: https://pypi.org/project/opencv-python/ for details.

## Algorithm
 - Convert RGB to HSV and select a range, which is used to segment the color
 - Create a mask with the segmented HSV color
 - Apply Morphological Operations (Closing) to fill out holes in the mask. Make sure you use appropiate kernel/filter size. 10 works for my case. 
 - Find all the contours using OpenCV function and then arrange them in descending order in terms of Area. 
 - Select the largest contour and create a complete mask of that using polynomial fill function. 
 - With that mask, create object and background masks
 - Bitwise OR the background and object masks in order to replace the object of interest with the background
 - Repeat this for every frame
 - Voila!

## How to use?
  1. Run the ColorSegmentation.py (While Running it, make sure nobody is infront of the webcam)
  2. An window showing the background image will pop up. 
  3. Press any key while being on that window to start real time frame-by-frame Color segmentation. 

The HSV values are calculated for the following cloth color. But you may have to re-define your values to make it work for other colors. 

![Cloth Used](Assets/Cloth.jpg)