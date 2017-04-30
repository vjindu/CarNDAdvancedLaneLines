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

#### The main code and execution can be seen in  'AdvancedLaneFindingPJT.ipynb'  notebook



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Camera Calibration

#### Q 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

It's normal for cameras to have distortion problems. In order to make sure we are getting an undistorted image for processing the image/ video stream we need to calibrate the camera. 

The camera calibration process needs a number of chessboard images taken by the same camera in different angles. A good starting point would be 20 images. The more the number of images the better the calibrated distortion points. 

The chessboard images used in this project are taken from the provided project repository. The provided images consists of 9x6 chessboard corners. These images are converted into grayscale first using 'cv2.COLOR_RGB2GRAY' filter.

Then Opencv library cv2.findChessboardCorners was used to find the chessboard corners in the gray images. The corner points found using cv2.findChessboardCorners function and camera object matrix are fead to cv2.calibrateCamera function. The cv2.calibrateCamera function provides with camera calibration and distortion coefficients.

These coefficients along with the distorted image(image from camera) are passed into the cv2.undistort Opencv function to obtain undistorted images.

The code for camera calibration can be found in the first code cell of the iPython notebook 'AdvancedLaneFindingPJT.ipynb' 

![png](Out_images/output_6_0.png)

###### The distorted input image and undistorted output image after passing detecting the distortion matrix can be seen in the above image. 

![png](Out_images/output_6_0.png)

###### The distorted and undistorted images of the road seen above.


#### Q 2. Use color transforms, gradients, etc., to create a thresholded binary image . Apply a perspective transform to rectify binary image. 

It was observed HLS filter extracts more information with regard to color gradient brightness with different lighter or darker colors. This allows us to better filter the necessary features in an image. Thus instead of converting directly form RGB to Gray images were converted to HLS. There are also other gradient methods like PLS. But we chose to use HLS filter in this project. HLS was used to obtain an image in three layers. The three layers are weighted combined to obtain a grayscale image. The function code used can be observed below. The lower layer was given a lower weightage since the lower layer highlites parts with lower gradient.

``` python
def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    gray_2 = ( 0.20*gray[:,:,0] + 0.4*gray[:,:,1] + 0.4*gray[:,:,2]).astype(np.uint8) # 
    return gray_2
``` python
![png](Out_images/color_gray.png)

##### The color and grayscale images with HLS thresholding can be seen in the image above.

Gray scale image is then passed into a binary filter with an upper and lower threshold of 200 and 120. To obtain the following gray scale image. Code for the function can be observed below.
```python
def grayToBinary(gray_img, thresh_min = 50, thresh_max = 150):
    
    #binary_img = np.zeros_like(gray_img)
    _, binary_img = cv2.threshold(gray_img.astype('uint8'), thresh_min, thresh_max, cv2.THRESH_BINARY)
    #binary_img[(gray_img >= thresh_min) & (gray_img <= thresh_max)] = 1
    
    return binary_img
```

![png](Out_images/Color_Binary.png)


#### Q 3. Rectify each image to a "birds-eye view". Perspective Transform


##### Detecting Source Perspective points

It is often hard to detect the right points and everytime there is a new image new perspective points are required to get lanes in paralle. Thus a function was implemented to automatically find perspective points given arbitrary points. 

the below code is used to detect source('src') points for the perspective transform. This code is partly obtained from project one. finding line points.

![png](Out_images/output_7_2.png)


```python
def detect_perspective_points(img, src, low_threshold = 40, high_threshold =150, color = [255, 0, 0],thickness = 10):

    gray = grayscale(img)
    blur_img = gaussian_blur(gray, 7)
    edge_img = cannyEdges(blur_img, low_threshold, high_threshold)
    
    lines = cv2.HoughLinesP(edge_img, rho, theta, threshold, min_line_length, max_line_gap)
    
    angle_min_mag = 25*np.pi/180 
    angle_max_mag = 70*np.pi/180 
    rho_min_diag = 0.1
    rho_max_diag = 0.6 
    
    lane_markers_x = [[],[]]
    lane_markers_y = [[],[]]
    #print(lines)
    diag_len = math.sqrt(img.shape[0]**2 + img.shape[1]**2)
    for line in lines:
        for x1,y1, x2,y2 in line:
            theta = math.atan2(y1-y2, x2-x1)
            rho = ((x1+x2)*math.cos(theta) + (y1+y2)*math.sin(theta))/2
            
            # cv2.line(img, (x1,y1), (x2,y2), color, thickness)
            
            if (abs(theta) >= angle_min_mag and abs(theta) <= angle_max_mag
                and rho >= rho_min_diag*diag_len and rho <= rho_max_diag*diag_len
               ):

                if theta > 0: # positive theta is downward in image space?
                    # Left lane marker
                    i = 0
                else:
                    # Right lane marker
                    i = 1
                lane_markers_x[i].append(x1)
                lane_markers_x[i].append(x2)
                lane_markers_y[i].append(y1)
                lane_markers_y[i].append(y2)
               
   
    p_left  = np.polyfit(lane_markers_x[0], lane_markers_y[0], 1)  
        # Gives out a polinomial function. Y = mx + c => P_left = (m, c) 
    p_right = np.polyfit(lane_markers_x[1], lane_markers_y[1], 1)
        
        
    left_bottom = math.ceil((src[3,1] - p_left[1])/p_left[0])   # x coordinate of Left bottom of the line. 
    right_bottom = math.ceil((src[2,1] - p_right[1])/p_right[0])  
        # x coordinate of the right lane bottom point. y coordinate img.shape[1] known
        
    left_top = math.ceil((src[0,1] - p_left[1])/p_left[0])   # x coordinate of the left lane top point
    right_top = math.ceil((src[1,1] - p_right[1])/p_right[0]) # x coordinate of the Right lane top point
    line_points = np.array([(0, left_bottom,left_top, 0),(right_bottom,img.shape[1], right_top, 0 )], dtype = np.uint8)
    src_points = np.float32([[left_top, src[0,1]], [right_top, src[1,1]], [right_bottom, src[2,1]], [left_bottom, src[3,1]]])

    return src_points
```

##### Perspective Transform

After Finding the source perspective points a perspective transform is applied to the image. This transformation of the image gives a 'birds eye' view of the lane lines.
This is in order to get the lanelines paralell to each other to make it easier for further processing. 

In order to get a perspective transform The source points ('src') detected from the previous section and the Destination points ('dst') into which the source has to be warped are passed to the openCV function 'cv2.getPerspectiveTransform()'. For inverse perspective or reverting the image back to its old position the destination points ('dst') are input first and then the source points ('src') in the same function. 

The function gives out a transform M matrix which is to be passed into another open cv function 'cv2.warpPerspective()'. 


The following image represent the Perspective transform or birds eye view of the source points detected. 

![png](Out_images/output_8_1.png)



The code for the function can be observed below.

```
def perspective_transform(undist_img, src, dst):
    img_size = (undist_img.shape[1], undist_img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped_img
```

#### Q 4. Detecting Left and right Lanes.

##### Histogram and Windowing detection.

###### Histogram

In order to first determine the base of the lane lines, Histogram of the bottom half of the image is taken. 

A function 'histogram_xy(edge_img)' was defined in the second code block of 'AdvancedLaneFindingPJT.ipynb' notebook file to take the histogram and obtain base points of the left and right lanelines.

The function is 
```
histogram_xy(edge_img)
```
The following image shows the histogram of the bottom half of the perspective transformed binary image. Two peak points of the histogram are taken as the base points of the left and right lanelines.

![png](Out_images/Histogram.png)

###### Windowing detection. 

When determining lanelines for the first time in the perspective transformed binary image, Windowing method is used. N windows exactly fitting the Y axis are determined with the y coordinates. The First windows for left and rightlanes are at the bottom of the image and the determined base points from the histogram are taken as the x axis centers for the first bottom windows. For both x and y lanes. 

The nonzero x and y points of the binary image are extracted for determining the next window position.
The following code is used to extract nonzero points in the image.
```
nonzero = edge_img.nonzero()
    
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])
```
The Next window above the first bottom window is determined by finding nonzero points in a positive and negative window margin in the x coordinates. If nonzero points are found in the windowing margin the mid base points of the next window are updated to the average mid point of the new found points. In this format all the windows for both left and right lanes are positioned on the image.

After making windows, all the nonzero points in the windows for each lane are taken separately and a 2d polinomial curve for each line is obtained with the nonzero points.

This polinomial is determined as the lanelines of the 

In the image below nonzero points of left line in red and the polinomial curve obtained in yellow,
 nonzero points of the right line in blue and the polinomial curve obtained in red, green boxes representing the windows for both lanes separately.

![png](Out_images/Boxes_Img.png)


The code for window detection function is defined in the second code block of 'AdvancedLaneFindingPJT.ipynb' notebook file.
The function is called as 
```
windowing(edge_img, leftx_base_pt, rightx_base_pt, nwindows = 9, window_x_margin = 70, min_pix = 30, out_margin = 35)
```

##### Detecting lanes around a predetermined points

The windowing method determines the base points. Since lanelines are contiuous lines, we can use the data from the previous detected lines to calculate the lane lines in the next frames. We assume that there is only a small difference in the position of the lanes from one frame to the next. So a positive and negative window margin is set for the obtained polinomial of each line from the previous frame. The nonzero x and y coordinates found in the window margin in the new frame are used to determine the new polynomial and thus left and right lanes for the image.


#### Q 5. Finding the radius of curvature of both the lanes and the vehicle position offset.


###### Radius of curvature measurement

The radius of curvature of each line is  

``` Python
def radiusOfCurvature(img_size, curve_poly):
    #curve_poly = np.polyfit(y_points*ym_per_pix, x_points*xm_per_pix, 2)
    ypoint =  img_size[1] #taking the max value of y
    rad_curve = (1 + (2*curve_poly[0]*ypoint + curve_poly[1])**2)**1.5/ np.absolute(2*curve_poly[0])
    return rad_curve
```

###### Vehicle position offset measurement

The camera is assumed to be in the center of the car. The 
``` Python
def vehiclePositionOffset(img_size, left_bottom_pt, right_bottom_pt):
    #Assuming camera is placed in the middle of the car
    midpoint = (left_bottom_pt + right_bottom_pt)//2
    car_pos = img_size[0]//2
    offset = (car_pos - midpoint)
    return offset
```


#### Q 6.  Warping the image back to its original transform.



#### Q 7. Video out file and code improvements

###### Sanity check

laneSeparationDist_sanity(x_dist, img_size, new_left_poly, new_right_poly) 
