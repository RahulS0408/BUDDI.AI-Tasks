# Buddi8.py

## Overview
This assignment demonstrates a simple convolution operation on two strings, counting exact and partial matches between the words of the strings. The results are plotted using `matplotlib` to visualize the match positions.

## Prerequisites
Ensure you have the following Python packages installed:
- `matplotlib`

You can install the necessary package using pip:
```bash
pip install matplotlib
```

## Code Description
The script is composed of the following sections:

1. **Importing Libraries**:
   - `matplotlib.pyplot` for plotting the results.

2. **Function Definition**:
   - `convolution(s1, s2)`: This function takes two strings as input and computes the exact and partial matches between them.
     - It splits the strings into lists of words.
     - It initializes lists to store exact and partial match counts.
     - It performs two nested loops to count exact and partial matches for each word in the strings.
     - It prints the match lists and the positions for visualization.
     - It returns the positions, exact matches, and partial matches for plotting.

3. **Example Strings**:
   - `s1` and `s2` are two example strings provided for convolution.

4. **Performing Convolution**:
   - Calls the `convolution` function with the example strings.

5. **Plotting Results**:
   - Plots the exact and partial match counts against their positions using `matplotlib`.

## Usage
To use the script, you can modify the strings `s1` and `s2` with your own strings and run the script. The plot will visualize the exact and partial matches between the two strings.


## Explanation of the Plot
- **X-axis (Position)**: Positions where the convolution was calculated.
- **Y-axis (Matches)**: Number of exact or partial matches.
- **Exact Matches**: Number of words that exactly match between the two strings at each position.
- **Partial Matches**: Number of words from the second string that are present in the first string at each position.

The plot helps visualize where the two strings have more exact and partial matches, with peaks indicating higher matches.


## Results:
![Convolution of two strings](Buddi8.png)


## Conclusion
This script provides a basic example of string convolution, counting exact and partial word matches, and visualizing the results using a line plot. Modify the example strings to see how different inputs affect the convolution results.

# Buddi8(2).py

## Overview
This project demonstrates how to perform image convolution to find the location where a smaller image (template) matches a larger image. It uses OpenCV for image processing and visualization.

## Prerequisites
Ensure you have the following Python packages installed:
- `opencv-python`
- `numpy`

You can install the necessary packages using pip:
```bash
pip install opencv-python numpy
```

## Code Description
The script performs the following steps:
1. **Importing Libraries**:
   - `cv2` for image processing.
   - `numpy` for numerical operations.

2. **Loading Images**:
   - Reads two images from specified file paths.

3. **Converting Images to Grayscale**:
   - Converts both images to grayscale for easier processing.

4. **Converting Grayscale Images to Float64**:
   - Converts the grayscale images to float64 format to prepare for convolution.

5. **Performing Convolution**:
   - Applies a convolution operation using the smaller image as the kernel on the larger image.

6. **Finding Match Location**:
   - Identifies the location with the highest convolution value, indicating the best match.

7. **Visualizing Match Location**:
   - Draws a rectangle on the original larger image to visualize the location of the best match.

8. **Displaying Images**:
   - Displays the original smaller image and the larger image with the match highlighted.

## Usage
To use the script, ensure your images are located at the specified paths or update the paths accordingly. Then run the script to see the matching result.


### Explanation of Variables
- **image1**: The larger image where the template will be searched.
- **image2**: The smaller template image.
- **gray_image1**: Grayscale version of the larger image.
- **gray_image2**: Grayscale version of the template image.
- **float_image1**: Larger image converted to float64.
- **float_image2**: Template image converted to float64.
- **result**: Result of the convolution operation.
- **min_convolve, max_convolve**: Minimum and maximum values of the convolution result.
- **min_coordinate, max_coordinate**: Coordinates of the minimum and maximum values in the convolution result.
- **top_left**: Top-left coordinate of the matching region.
- **bottom_right**: Bottom-right coordinate of the matching region.
- **h, w**: Height and width of the template image.

## Results:
![Convolution of Images](Buddi8(2).png)


## Conclusion
This script provides a method to perform image convolution to find and visualize the matching region of a template in a larger image using OpenCV. Modify the file paths to your own images to test the script with different inputs.
