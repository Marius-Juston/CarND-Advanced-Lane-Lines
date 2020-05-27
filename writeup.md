## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./report_images/calibration_example.jpg "Calibration Example"
[image2]: ./report_images/undistorted_calibration.png "Undistorted Calibration Example"
[image3]: ./report_images/initial.jpg "Initial"
[image4]: ./report_images/undistorted.png "Undistorted"
[image5]: ./report_images/grads.png "Applied Combined Gradient Thresholding"
[image6]: ./report_images/colors.png "Applied Color Thresholding"
[image7]: ./report_images/Combined.png "Combined Thresholds"
[image8]: ./report_images/Outline.png "The transform outline points"
[image9]: ./report_images/Perspective.png "Perspective Transform"
[image10]: ./report_images/Sliding%20Lanes.png "Finding Lanes through Sliding Means"
[image11]: ./report_images/lane_histogram.png "Histogram of pixel density"
[image12]: ./report_images/Reverse%20Transform.png "Reverse Transform the Lane"
[image13]: ./report_images/output.jpg "Final output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function `undistort` and `calibrate_camera` in the `pipeline.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

I then saved the matrix into a pickle file for later use so that it does not need to be computed every time

Distorted             |  Undistorted
:-------------------------:|:-------------------------:
![alt text][image1] | ![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Distorted             |  Undistorted
:-------------------------:|:-------------------------:
![alt text][image3] | ![alt text][image4]

I start by looking at if I have already computed the required matrices for undistorting an image by looking at the parameter dict (`params`) and seeing if the file directed to by ```params['undistort']``` exists. If it does not it will run the calibration sequence in `callibrate_camera` and compute the required matrices and save them to the asked file. From there I use the matrices computed to call `cv2.undistort()`

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image.
I started by creating a OpenCV GUI in order to be able to fine tune the gradient thresholder which comprises of Sobel x, Sobel y, Sobel Magnitude and Sobel direction. The combination of these  in the form `(Sx & Sy) | (Smag & Sdir)` could be controlled and fined tuned by the GUI. The GUI controlled every aspect of the images such as the individual kernels and the all the thresholds for the Sobel derivatives. 

As a result of the fine tuning I managed to achieve a result of this:

![alt text][image5]

I then proceeded to fine tuning the color spaces. I only used the HLS color space in order to fine tune my image, with a little trial and error and tweaking I achieved this.

![alt text][image6]

A threshold that removed a lot of the noise.

I then proceeded to combine both thresholds in order to get the most useful information out of the image. and ths is the result

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `find_perspective_lines`, which appears in lines 325 and 450 in the file `pipeline.py` (scripts/pipeline.py). The `find_perspective_lines` function takes as inputs an image (`image`), as well as the pipeline parameters (`params`) and a boolean flag to show the vertices or not (`show_plots`).  I chose the hardcode the source and destination points in the following manner:

```python
ground_offset = 35
x_bottom_width = 835
x_top_width = 115
y_height = height * 5 / 8

vertices = np.array([[width // 2 - x_bottom_width // 2, height - ground_offset],
                     [width // 2 - x_top_width // 2, y_height],
                     [width // 2 + x_top_width // 2, y_height],
                     [width // 2 + x_bottom_width // 2, height - ground_offset]], dtype=np.float32)

dist = np.array([[width // 2 - x_bottom_width // 2, height],
                 [width // 2 - x_bottom_width // 2, 0],
                 [width // 2 + x_bottom_width // 2, 0],
                 [width // 2 + x_bottom_width // 2, height]], dtype=np.float32)
```

I chose to elevate the trapezoid due to fact that the bottom of the image was getting cut of by the car's front.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 223, 685      | 223, 720        | 
| 583, 450      | 223, 0      |
| 697, 450     | 1057, 0      |
| 1057, 685      | 1057, 720        |

I verified that my perspective transform was working as expected by drawing the `vertices` and `dist` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. This could be triggered to be on or off given the `show_plots` parameter

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using the perspective transformed image through `find_perspective_lines` which returned this:
![alt text][image9]

I then proceeded to use a moving average with a subdivision of 18 windows and I used a histogram to figure out where the peaks in the vertical sum of the images where so that I could know where to start the first windows. This function is called `fit_polynomial` and is implemented in lines 328 and 455 of `pipeline.py`.

After finding each peak each window would increment by its' height and within the margin find the highest concentration of "on" values thus shifting the starting x for the next window.

This is an example of a histogram, you can clearly see two peaks for the right and the left lane.

![alt text][image11]

The values that where within the margins of the window (its width) are considered part of the lane and are appended to a list of points. These points are then used to use a polynomial of order 2 fit using numpy's `np.polyfit`

This is the result of the sliding mean and the poly fit line:

![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In order to calculate the radius of the curvature I used the the two lines of best fit calculated previously. I then used the formula for calculate the curvature of the lane. At the same time I scaled the curvature in order to reflect real world units, in this case meters.

This process was done through the function `calculate_curvature` seen in `pipeline.py` and implemented in lines 331-332 and 468-489

In order to calculate the offset from the center of the image I use the fitted lane lines and from that I got the lowest point possible and then I took the average x value of points. This average should represent the center of the lane. From that I proceeded tp substract that pixel value with half the width of the image in order to know the difference. From this difference I converted the value into real world values.

This process was done through the function `get_offset` seen in `pipeline.py` and implemented in lines 334 and 471

This is the output:

![alt text][image13]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 328 and 465 in the function `draw_lines` in `pipeline.py`. In order to implement this change instead of drawing the lines in the perspective changed image I decided to transform the polyfit line points instead into the orginal perspective. In order to implement this I had to reshape the points in the format (-1, 1, 2) in order for the algorithm to work.

I then utilized OpenCV's `cv2.perspectiveTransform` and passed in the points as the initial parameters and the inverse matrix of the perspective transform matrix.

I then drew the points unto the image and the created the fill polygon as an overlay.

This is the result:

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

###### Issues

Some issues that rose while working on the pipeline is that if there was too large of a curve then the sliding mean window:
1. Started too off center thus causing the next window to be offset and miss some values
2. If there was too large of a turn the margin would be able to catch all the data
3. If there was a sharp turn and some erroneous lanes appeared then it was possible for the window to catch that fake lane a pursue that instead
 
Another issue that I noticed was when there were sharp edges due to shadows the Sobel thresholder managed to identify it as an edge; however, this caused the sliding window to misinterpret it as a a lane

Another issue was when the car was too off center then the it was possible that the the lane would not appear in the perspective

###### Improvements

In order to fix some of the issues I could fine tune even more and use even more thresholds such as other color thresholds. 

For creating a better perspective warp I could try to find the vanishing point in order to find the horizon and be able to create a better figure that best represent the flat road.

For the sliding mean I could make the find adaptive so it goes higher the window gets larger and wider due to the corrected warped perspective that increases the image that was before further away  