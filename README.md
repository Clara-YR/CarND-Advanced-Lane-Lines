
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
[image4]: ./output_images/thresholds/abs_sobel_thresh.jpg
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




## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

**Catalog**

1. Calibrate the Camera
2. Correct for Image Distortion
3. Implement a Color & Gradient Threshold
 - HLS Threshold
 - Gradient Threshold
 - Combine Color and Graident Thresholds
4. Warp the Image Using Perspective Transform
5. Decide Which Pixels are Lane Line Pixels
6. Determine the Line Shape and Position
7. Output


##1. Calibrate the Camera
*The code for this step is contained in __cell [1] to [4]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Chessboard images with corners drawn are saved in file "./output_images/draw_corners/" as well as arrays `objpoints` and `imgpoints` in "./output_images/".

Here is an example of chessboard images with corners drawn:

![alt text][image0]

---

##2. Correct for Image Distortion
*The code for this step is contained in __cell [5] to [8]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

---

##3. Implement a Color & Gradient Threshold
*The code for this step is contained in __cell [9] to [24]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image9]

### 3.1 Load and display original test images

Those test images are used to:

- check my threshold functions 
- tune and get the best parameter values for my threshold functions

![alt text][image2]

### 3.2 HLS threshold
The `hls_select()` fucntion applies threshold on the S-channel of HLS.

I tune the parameter as `hls_select(img, thresh=(180, 255))`

![alt text][image3]



### 3.3 Gradient Threshold

#### 3.3.1 Absolute Value of the Gradient
The `abs_sobel_thresh()` function applies Sobel x and y, then takes an absolute value and applies a threshold.

I tune the parameter as `abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)`

![alt text][image4]

#### 3.3.2 Magnitude of the Gradient
The `mag_thresh()` function returns the magnitude of the gradient for a given sobel kernel size and threshold values.

I tune the parameter as `mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))`

![alt text][image5]

#### 3.3.3 Direction of the Gradient
The `dir_threshold` function applies Sobel x and y, then compute the direction of the gradient and applies a threshold.

I tune the parameter as `dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.2))`

![alt text][image6]

#### 3.3.4 Combing Gradient Thresholds
To summarize, I tune all threshold parameters as below:

- `hls_select(img, thresh=(180, 255))`
- `abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)`
- `mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))`
- `dir_threshold(img, sobel_kernel=9, thresh=(0.7, 1.2))`

The `combine_gradient` function combines gradient, magnitude of gradient and direction of gradient thresholds. 

![alt text][image7]

### 3.4 Combining Color and Gradient Thresholds

Finally I combine color and all gradient thresholds together to in `color_grad()` function.

![alt text][image8]

Here is a comparison of a test after the implemetion of color & gradient thresholds.

![alt text][image9]

---

##4. Warp the Image Using Perspective Transform
*The code for this step is contained in __cell [25] to [32]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
img_size = (1280, 720) 

src = np.int32(
    [[(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100],
    [(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]]])
dst = np.int32(
    [[(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0],
    [(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1126, 720     | 960, 720      | 
| 705, 460      | 960, 0        |
| 580, 460      | 320, 0        |
| 320, 720      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

The `src` and `dst` seemed to be good so I applied them in my `warper()` function. The output of the `warper()` function is as below when taken the eight test images as input:

![alt text][image11]

---

##5. Decide Which Pixels are Lane Line Pixels
*The code for this step is contained in __cell [33] to [39]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

The `window_mask` function is used to draw window areas.
The `find_window_centroids` function finds all the left and right window centroids for each level in the given binary image.
The `mark_centroids()` function find and mark left and right centroids.

![alt text][image12]

The `find_ploty()` function extract left and right line pixel positions to fit a second order polynomial to both, and generate x and y for plotting.

![alt text][image13]

---

##6. Determine the Line Shape and Position
*The code for this step is contained in __cell [40] to [48]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

###6.1 Measuring Curvature

To begin with I calculated the curve rad of the left and right lane lines on the bottom of the image. If both curve rads are valid I took the average of them as the lane curvature. If one curve rad was invalid I took the valid one as the lane curvature.

My `measure_curve()` function calculated the curvature of the eight test images as the table below:

|Image Name| Left| Right| Curvature|
|:--------:|:---:|:-----|:---------:|
|straight_lines1|3741.42(m)|3851.54(m)|3796.48(m)|
|straight_lines2|210573.06(m)|nan|210573.06(m)|
|test1|707.45(m)|1420.46(m)|1063.96(m)|
|test2|709.11(m)|nan|709.11(m)|
|test3|1625.19(m)|960.96(m)|1293.07(m)|
|test4|2424.32(m)|9230.60(m)|5827.46(m)|
|test5|590.13(m)|1051.76(m)|820.94(m)|
|test6|1443.10(m)|599.76(m)|1021.43(m)

###6.2 Determine Vehicle Position

|Image Name| Position |
|:---:|:---:|
|straight_lines1|Vehicle is 0.06m left of center
|straight_lines2|Vehicle is 0.08m left of center
|test1|Vehicle is 0.13m right of center
|test2|Vehicle is 0.14m right of center
|test3|Vehicle is 0.05m right of center
|test4|Vehicle is 0.22m right of center
|test5|Vehicle is 0.06m left of center
|test6|Vehicle is 0.19m right of center

###6.3 Drawing Lane

My `drawing()` function marks the area betweern the detected left and right lanes with green as below:

![alt text][image14]

My `drawing_lane()` function marks the detected left and right lane lines in red and blue respectively.

![alt text][image15]

---

##7. Output
*The code for this step is contained in __cell [49] to [4]__ of the IPython notebook located in "./Advanced Lane Finding.ipynb".*

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