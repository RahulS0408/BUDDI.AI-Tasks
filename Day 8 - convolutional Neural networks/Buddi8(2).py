# Import necessary libraries or modules
import cv2
import numpy as np

# Load the images
image1 = cv2.imread('/home/rahul/anaconda3/Documents/Buddi/Day 8 - convolutional Neural networks/Group2.jpg')
image2 = cv2.imread('/home/rahul/anaconda3/Documents/Buddi/Day 8 - convolutional Neural networks/Crop2.jpg')

# Convert the images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Convert the grayscale images to float64
float_image1 = gray_image1.astype(np.float64)
float_image2 = gray_image2.astype(np.float64)

# Perform the convolution
result = cv2.filter2D(float_image1, -1, float_image2)

# Find the location where the second image matches the first image
(min_convolve, max_convolve, min_coordinate, max_coordinate) = cv2.minMaxLoc(result)

# Print the result
print(f"Min and Max Convolution value:{min_convolve, max_convolve}")
print(f"Min and Max coordinates location: {min_coordinate, max_coordinate}")

# Visualize the match location on the original image
top_left = max_coordinate
h, w = gray_image2.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image1, top_left, bottom_right, (255, 0, 0), 2)

# Display the original image with the match location
cv2.imshow('crop image',image2)
cv2.imshow('Matched part in full image', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()