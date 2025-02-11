import cv2
import numpy as np


def align_images(window_img, blind_img):
    # Convert to grayscale
    gray_window = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
    gray_blind = cv2.cvtColor(blind_img, cv2.COLOR_BGR2GRAY)

    # Detect features and match
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_window, None)
    kp2, des2 = orb.detectAndCompute(gray_blind, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_blind = cv2.warpPerspective(blind_img, M, (window_img.shape[1], window_img.shape[0]))

    return aligned_blind