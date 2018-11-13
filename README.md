
# Advanced Lane Finding Project

---

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

[image0]: ./output_images/draw_corners/calibration2.jpg "Draw Corners"
[image1]: ./output_images/undistorted_test.jpg "Undistorted"
[image2]: ./output_images/thresholds/load_images.jpg "Test Images"
[image3]: ./output_images/thresholds/HLS.jpg 
[image4]: ./output_images/thresholds/abs_sobelx_thresh.jpg
[image5]: ./output_images/thresholds/mag_thresh.jpg 
[image6]: ./output_images/thresholds/dir_threshold.jpg 
[image7]: ./output_images/thresholds/combine_gradient.jpg
[image8]: ./output_images/thresholds/color_grad.jpg 
[image9]: ./output_images/thresholds/thresholds.jpg 
[image10]: ./output_images/src&dst_drawn.jpg
[image11]: ./output_images/warped_binary_image.jpg
[image12]: ./output_images/window_fitting_result.jpg
[image13]: ./output_images/find_ploty.jpg
[image14]: ./output_images/draw_lane_area.jpg
[image15]: ./output_images/draw_lane_line.jpg
[image16]: ./output_images/process_image.jpg
[image17]: ./output_images/src&dst_drawn_1st_submission.jpg
[image18]: ./output_images/thresholds/RGB.jpg
[image19]: ./output_images/thresholds/combine_color.jpg
[image20]: ./output_images/thresholds/abs_sobely_thresh.jpg
[image21]: ./output_images/better_fitting_result.jpg



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

**Catalog**

1. Calibrate the Camera
2. Correct for Image Distortion
3. Implement a Color & Gradient Threshold
4. Warp the Image Using Perspective Transform
5. Decide Which Pixels are Lane Line Pixels
6. Determine the Line Shape and Position
7. Output
8. Discussion


## 1. Calibrate the Camera
*The code for this step is contained in __cell [1] to [4]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Chessboard images with corners drawn are saved in file "./output_images/draw_corners/" as well as arrays `objpoints` and `imgpoints` in "./output_images/".

Here is an example of chessboard images with corners drawn:

![alt text][image0]

---

## 2. Correct for Image Distortion
*The code for this step is contained in __cell [5] to [8]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---

## 3. Implement a Color & Gradient Threshold
*The code for this step is contained in __cell [9] to [31]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image9]

### 3.1 Load and display original test images

Those test images are used to:

- check my threshold functions 
- tune and get the best parameter values for my threshold functions

#### 3.1.1 Import snapshots of project_video.mp4 as test images

#### 3.1.2 Load all test images

![alt text][image2]

### 3.2 Color Thresholds

#### 3.2.1 HLS threshold
The `hls_select()` fucntion applies threshold on the S-channel of HLS.

I tune the parameter as `hls_select(img, thresh=(180, 255))`

![alt text][image3]

#### 3.2.2 RGB threshold
The `rgb_select()` function applies threshold on the R-channel of RGB.

I tune the parameter as `rgb_select(img, thresh=(50, 255))`

![alt text][image18]

#### 3.2.3 Combine HLS and RGB threshold

To summarize, I tune all color threshold parameters as below:
- `hls_select(img, thresh=(80, 255))`
- `rgb_select(img, thresh=(50, 255))`

The `combine_color` function combines S_channel and R_channel to set color thresholds.

![alt text][image19]

### 3.3 Gradient Threshold

#### 3.3.1 Absolute Value of the Gradient
The `abs_sobel_thresh()` function applies Sobel x and y, then takes an absolute value and applies a threshold.

I tune the parameter as `abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=200)`

![alt text][image4]

and `abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=200)`

![alt text][image20]

#### 3.3.2 Magnitude of the Gradient
The `mag_thresh()` function returns the magnitude of the gradient for a given sobel kernel size and threshold values.

I tune the parameter as `mag_thresh(img, sobel_kernel=3, thresh=(30, 100))`

![alt text][image5]

#### 3.3.3 Direction of the Gradient
The `dir_threshold` function applies Sobel x and y, then compute the direction of the gradient and applies a threshold.

I tune the parameter as `dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.2))`

![alt text][image6]

#### 3.3.4 Combing Gradient Thresholds
To summarize, I tune all threshold parameters as below:

- `abs_sobel_thresh(img, orient='x', thresh=(20, 200))`
- `abs_sobel_thresh(img, orient='y', thresh=(20, 200)=)`
- `mag_thresh(img, sobel_kernel=3, thresh=(30, 100))`
- `dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.2))`

The `combine_gradient` function combines gradient, magnitude of gradient and direction of gradient thresholds. 

![alt text][image7]

### 3.4 Combining Color and Gradient Thresholds

Finally I combine color and all gradient thresholds together to in `color_grad()` function.

![alt text][image8]

Here is a comparison of a test after the implemetion of color & gradient thresholds.

![alt text][image9]

---

## 4. Warp the Image Using Perspective Transform
*The code for this step is contained in __cell [32] to [39]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
img_size = (1280, 720) 

src = np.int32(
    [[(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100],
    [(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]]])
dst = np.int32(
    [[(img_size[0] * 5 / 6), img_size[1]],
    [(img_size[0] * 5 / 6), 0],
    [(img_size[0] / 6), 0],
    [(img_size[0] / 6), img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1126, 720     | 1066, 720      | 
| 705, 460      | 1066, 0        |
| 580, 460      | 213, 0        |
| 320, 720      | 213, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

The `src` and `dst` seemed to be good so I applied them in my `warper()` function. The output of the `warper()` function is as below when taken the eight test images as input:

![alt text][image11]

---

## 5. Decide Which Pixels are Lane Line Pixels
*The code for this step is contained in __cell [40] to [48]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

### 5.1 Window Fitting

The `window_mask` function is used to draw window areas.

#### 5.1.1 Unidirectional Window Fitting
The `find_window_centroids` function finds all the left and right window centroids for each level in the given binary image. __Note__: the output centroids are ordered from the bottom to the top of the image.

The `mark_centroids()` function find and mark left and right centroids.

![alt text][image12]

#### 5.1.2 Bidirectional Window Fitting
As the output images shown above, the `find_window_centroids` works well in the front half part of the levels but tends to be out of order in the back half part of the levels. 

Thus I decide to apply the `find_window_centroids` in both directions (from bottom to top as well as from top to bottom) and choose the better left and right centroids for each level repectively.

`np.flipud()` is used twice to upside down the binary warped image as an input, as well as the `find_window_centroids()` output to make the orders of the level in the same direction

The `better_window_centroids()` function find window centroids in bidirection (bottom to top & top to bottom) and then selects better left and right centroids for each level.


![alt text][image21]

### 5.2 Find and Plot the Track Line

The `find_ploty()` function extract left and right line pixel positions to fit a second order polynomial to both, and generate x and y for plotting.

![alt text][image13]

---

## 6. Determine the Line Shape and Position
*The code for this step is contained in __cell [49] to [58]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

### 6.1 Measuring Curvature

To begin with I calculated the curve rad of the left and right lane lines on the bottom of the image. If both curve rads are valid I took the average of them as the lane curvature. If one curve rad was invalid I took the valid one as the lane curvature.

__Note__: the change `xm_per_pix = (3.7/700)*(3/4)` is described in details in Part 8.3.1

My `measure_curve()` function calculated the curvature of the eight test images as the table below:

|Image Name| Left| Right| Curvature|
|:--------:|:---:|:-----|:---------:|
|snapshot1|21186.25(m)|4318.20(m)|12752.22(m)|
|snapshot2|854.74(m)|929.56(m)|892.15(m)|
|snapshot3|815490.54(m)|1145.41(m)|8317.79(m)|
|snapshot4|940.45(m)|951.98(m)|946.22(m)|
|straight_lines1|2415.82(m)|3615.28(m)|3015.55(m)|
|straight_lines2|2503.34(m)|9275.88(m)|5889.61(m)|
|test1|5688.72(m)|1815.23(m)|3751.97m)|
|test2|726.83(m)|632.95(m)|679.89(m)|
|test3|1395.48(m)|1013.39(m)|1204.44(m)|
|test4|17898.69(m)|3441.74(m)|10670.22(m)|
|test5|1049.39(m)|1062.38(m)|1055.89(m)|
|test6|2182.19(m)|697.80(m)|1440.00(m)|

### 6.2 Determine Vehicle Position

|Image Name| Position |
|:---:|:---:|
|snapshot1|Vehicle is 0.15m left of center|
|snapshot2|Vehicle is 0.15m left of center|
|snapshot3|Vehicle is 0.07m left of center|
|snapshot4|Vehicle is 0.08m left of center|
|straight_lines1|Vehicle is 0.03m right of center
|straight_lines2|Vehicle is 0.01m right of center
|test1|Vehicle is 0.12m left of center
|test2|Vehicle is 0.20m left of center
|test3|Vehicle is 0.08m left of center
|test4|Vehicle is 0.24m left of center
|test5|Vehicle is 0.04m right of center
|test6|Vehicle is 0.22m left of center

### 6.3 Drawing Lane

My `drawing()` function marks the area betweern the detected left and right lanes with green as below:

![alt text][image14]

My `drawing_lane()` function marks the detected left and right lane lines in red and blue respectively.

![alt text][image15]

---

## 7. Output
*The code for this step is contained in __cell [59] to [65]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

The `process_image` function:

1. undistorts the original camera image,
2. converts the undistorted image into a binary form via applying color & gradient thresholds,
3. warps the binary image into birds-eye perspective,
4. detects lane curvature and position via the warped binary image,
5. displays the curvature and position on the undistorted image,
6. draws/marks the detected lane on the undistorted image.

![alt text][image16]

###Pipeline(video)

My final output video lacated in "./test\_videos\_output/project\_video.mp4". 

---

## 8. Discussion

I discussed how I made improvements according to my reviewer's advice.

### 8.1 Choose line to calculate curvature

Get rid of nan value in curvature calculation.

1st submission

```
def measure_curve(ploty, fitx):
	...
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, fitx[0]*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, fitx[1]*xm_per_pix, 2)
	
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1]**2)**1.5) / np.absolute(2*right_fit_cr[0]))
	...
```

2nd submission

```
def measure_curve(ploty, lane):
	...
	# Fit a second order polynomial to each lane
	A_left, B_left, C_left = np.polyfit(lane[0]*ym_per_pix, lane[1]*xm_per_pix, 2)
	A_right, B_right, C_right = np.polyfit(lane[2]*ym_per_pix, lane[3]*xm_per_pix, 2)
	    
	# Calculate the radius of each lane
	left_curverad = calculate_curve(A_left, B_left, y_eval*ym_per_pix)
	right_curverad = calculate_curve(A_right, B_right, y_eval*ym_per_pix)
	...
```

I used `lane` to substitute `fitx` as depedent variable of function `measure_curve()`. Both `lane` and `fitx` are the output of function `find_ploty` 

```
def find_ploty():
	...
	# Pack all x and y coordinates to cut down the numbers of output
	lane = np.array([lefty, leftx, righty, rightx])
	fitx = np.array([left_fitx, right_fitx])
	...
	return ploty, lane, fitx
```
`lane` is a set of nonzero pixel coordinates within each window area while `fitx` is a set of points coordinates for plotting. `fitx` is calculated via `lane` so used `lane` instead of `fitx` could decrease error.

### 8.2 Determine Vehicle Position

Change the 'right' and 'left' in 2nd submission:

```
def car_position(fitx, x_median):
xm_per_pix = 3.7/700 # meters per pixel in x dimension
car_center = x_median * xm_per_pix  # take the image center in x direction as the center of the vehicle
    
# Take the bottom x positions of the left and right lanes
	x_left = fitx[0][-1] * xm_per_pix
	x_right = fitx[1][-1] * xm_per_pix
	# Take the average of the x_left and x_right as the center of the lane
	lane_center = (x_left + x_right)/2
	
	if car_center+0.005 < lane_center:
	    T_position = 'Vehicle is {:0.2f}m left of center'.format(lane_center-car_center)
	    return T_position
	elif car_center-0.005 > lane_center:
	    T_position = 'Vehicle is {:0.2f}m right of center'.format(car_center-lane_center)
	    return T_position
	else:
	    return 'Vehicl is on the center
```  
 

### 8.3 Output Video Improvement

####8.3.1 Take the snapshot of the problematic frames and analyse them.

*The code for this step is contained in __cell [5]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*


**try to map the lane to a slightly wider part of the birds-eye view image, so that the passing black car does not disturb**

before:

```
dst = np.int32(
    [[(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0],
    [(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]]])
```
```
dst = array([[960, 720],
       [960,   0],
       [320,   0],
       [320, 720]], dtype=int32)    
```
![alt text][image17]  
after:

```
dst = np.int32(
    [[(img_size[0] * 5 / 6), img_size[1]],
    [(img_size[0] * 5 / 6), 0],
    [(img_size[0] / 6), 0],
    [(img_size[0] / 6), img_size[1]]])
```
```
array([[1066,  720],
       [1066,    0],
       [ 213,    0],
       [ 213,  720]], dtype=int32)
```    

I also changed `xm_per_pix = 3.7/700` to `xm_per_pix = (3.7/700)*(3/4)` since the ratio of the same horizontal distance between lane lines were __(3/4 - 1/4):(5/6 - 1/6) = (1/2) : (2/3) =  3/4__

#### 8.3.1 Use bidirectional window fitting instead of Unidirectional Window Fitting.

See more details in __Part 5.1__
