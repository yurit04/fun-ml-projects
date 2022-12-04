# **Finding Lane Lines on the Road**

[image1]: ./test_images/solidYellowCurve.jpg
[image2]: ./test_images_output/solidYellowCurve.jpg


### 1. Description of the pipeline and the *draw_lines()* function. 

My image processing pipeline consists of the following steps. 
* Convert images to gray scale using *grayscale()* helper function. 
* Apply Gaussian smoothing via *gaussian_blur()* function to smooth out the noise.
* Apply Canny edge detection algorithm to identify objects/lane edges on an image. 
* Define a trapezoidal "region of interest", which most likely contains the lanes. 
* Use Hough transform to identify lane segments. 
* Connect all lane segments via *draw_lines()* function. 

Below is an example of an input image and an output image. 

![alt text][image1]

![alt text][image2]

The output videos can be found in the */test\_videos_output* folder.

In order to draw a single line on the left and right lanes, I modified the *draw_lines()* function to perform the following calculation steps. 
 * Collect all the coordinates (x1,y1,x2,y2) defining a lane segment from the left half of the "region of interest" into two arrays x_left, y_left, provided the slope of the line satisfies the condition 0.25 < slope < 0.8. 
 * Collect all the coordinates (x1,y1,x2,y2) defining a lane segment from the right half of the "region of interest" into two arrays x_right, y_right, provided the slope of the line satisfies the condition -0.8 < slope < -0.25. 
 * The slope requirement was needed to improve stability of the function output on solidYellowLeft.mp4 test video. 
 * To identify the *slope* and *intercept* parameters of the left solid line, a linear regression was performed on (x_left,y_left) pair. 
 * A regression was also performed on the (x_right,y_right) pair to find the *slope* and *intercept* of the right solid line. 
 

### 2. Potential shortcomings with the current pipeline/approach

The pipeline is very sensitive to the choice of the hyperparameters. Even though the presented approach works well on the specific videos of this project, most likely it will not generalize well to other videos. This can happen for a variety of reasons such as obstacles on the road of white and/or yellow color, different color or fading lanes, and if lanes are not straight. 


### 3. Possible improvements 

* We could define a flexible region of interest that could fit curved lanes as well as straight ones. We might be able to make it work if we allow for the algorithm to use the lane information from the previous frames, i.e. to make it a time-dependent approach. 
* If we could automate the hyperparameter tunning process, it would definitely help a lot.
