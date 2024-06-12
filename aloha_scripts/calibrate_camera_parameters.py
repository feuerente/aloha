"""
Calibrate intrinsic camera parameters.

- https://docs.opencv.org/4.x/d6/d55/tutorial_table_of_content_calib3d.html
- [Calibration pattern generator](https://markhedleyjones.com/projects/calibration-checkerboard-collection)
"""

import glob
import os
import time

import cv2
import numpy as np

from robot_utils import ImageRecorder


def record_calibration_images(num_pictures, delay_sec, image_folder, cam_name):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # cap = cv2.VideoCapture(0)
    image_recorder = ImageRecorder(init_node=True)

    print("Press a key to start capturing images.")

    while True:
        # Read a frame from the camera
        frame = image_recorder.get_cam_low_image()

        # Display the frame
        cv2.imshow('Video Feed', frame)

        # Check for the key press
        cv2.waitKey(0)

        for i in range(num_pictures):
            start_time = time.time()
            while time.time() - start_time < delay_sec:
                # frame = image_recorder.get_images()[cam_name]
                frame = image_recorder.get_cam_low_image()


                # Calculate remaining time
                elapsed_time = time.time() - start_time
                remaining_time = delay_sec - elapsed_time

                # Display the countdown on the frame
                countdown_text = f"Capturing in {remaining_time:.1f} seconds"
                # cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Video Feed', frame)
                cv2.waitKey(10)  # Wait for 0.1 second

            # Capture and save the image
            frame = image_recorder.get_images()[cam_name]

            # Define the filename
            filename = f"{image_folder}/image_{i + 1}.jpg"

            # Save the captured frame
            cv2.imwrite(filename, frame)
            print(f"Captured {filename}")

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def calibrate_camera_parameters(image_folder):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 9)  # Inner corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []  # 3D points for each checkerboard image
    imgpoints = []  # 2D points for each checkerboard image

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob(f"{image_folder}/*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"fx = {mtx[0, 0]}")
    print(f"fy = {mtx[1, 1]}")
    print(f"cx = {mtx[0, 2]}")
    print(f"cy = {mtx[1, 2]}")
    print("Camera matrix : \n")
    print(mtx)
    print("Distortion coefficients : \n")
    print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)


if __name__ == '__main__':
    num_pictures = 11
    delay_sec = 2
    image_folder = "calibration_images"
    cam_name = "cam_low"
    # cam_name = "cam_high"

    # record_calibration_images(num_pictures, delay_sec, image_folder, cam_name)
    calibrate_camera_parameters(image_folder)
