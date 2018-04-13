
[TOC]

## Writeup
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

[orig_chessboard]: ./output_images/orig.png "original"
[undistort_chessboard]: ./output_images/undis.png  "Undistorted"
[distort_road]: ./output_images/distorted_img.png "Road Transformed"
[thresh]: ./output_images/thresh_img.png "Binary Example"
[undistort_road]: ./output_images/undistorted_img.png
[warp_img]: ./output_images/warped_img.jpg "Warp Example"
[image5]: ./output_images/out_img.jpg "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


The code for this step is contained in the function definition get_camera_calibation() in the file called `helpFunctions.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:  
 
![alt text][orig_chessboard]
![alt text][undistort_chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:  
![alt text][distort_road]  
with the distortion corrected resulting   
![alt text][undistort_road]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 97 through 106 in `helpFunctions.py`).  Here's an example of my output for this step.

The V-channel and S-channel from HSV and HLS colorspaces are used. The Sobel filter along the X axis has been applied as well. finally the mangitude-thresholding and directional-threstholding is applied. the result of combining the previous function found in the `helpFunctions.py` in lines 51 to 95

![alt text][thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform which appears in lines 34 through 49 in the file `helpFunctions.py` function `c2.warpPerspective()`. `get_prespective()`function takes as inputs an image (`img`), where source (`src`) and destination (`dst`) points are pre-defined.  I chose the hardcode the source and destination points in the following manner:

```python
h, w = 720, 1280
src = np.float32([(525,464),
                  (630,464),
                  (300,682),
                  (700,682)])

dst = np.float32([(460,0),
                  (w-476,0),
                  (460,h),
                  (w-460,h)])
```
The ouput transformation matrix of the function is then pass over to `c2.warpPerspective()` at line 30 to warp the imag. 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 525, 464      | 460, 0        | 
| 630, 464      | w-476, 0      |
| 300, 682      | 460, h        |
| 700, 682      | w-460, h      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][warp_img]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once I identified the lane line pixels using histogram peaks to identify the starting point followed by sliding window operation , I used np.polyfit function to fit the 2nd order polynomial function to the detected lane lines . I followed the same formula as shown during the lectures the result of sliding window is shown below for one of the test images. The code for the same is in script main.py from line (44-152)


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

There are two functions in helpFunctions.py from line 109-114 called lane_curvature
and dist_from_center from line 116-120 to compute the lane curvature and and dist of vechile from the center of the lane respectively.
The radius of curvature was computed with formula
Rcurve=∣2A∣(1+(2Ay+B)2)3/2 for both left and right lane respectively.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 123 through 150 in my code in `main.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging part was fine-tuning the warping correctly as well the thresholding paramters

The system would fail if the vehicle would attempt changing lanes as the historgam results presumes only two lans present in the image field but in case of more than two lanes present the i woudl expect the system to fail.