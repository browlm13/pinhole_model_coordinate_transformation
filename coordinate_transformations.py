#!/usr/bin/env python

"""
        Pinhole Model Coordinate Transformations
"""

__author__ = "l.j. Brown"
__version__ = "1.0.1"

# imports

# internal
import os
import logging
import random
import math
import copy

# external
import glob
import pandas as pd
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy import stats

# my lib
import aruco_marker_detection as marker_detector
import frames

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Methods


def solve_pnp(model_points, image_points, camera_matrix, distortion_coefficients):
    """
    :param model_points:
    :param image_points:
    :param camera_matrix:
    :param distortion_coefficients:
    :return: rotation_vector,
    :return: translation_vector
    """
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                          distortion_coefficients)

    return rotation_vector, translation_vector


def model_camera_matrices_from_images_points(model_points, image_points, camera_matrix, distortion_coefficients):
    """
    Solve for the homogeneous model / camera transformation matrices using image points and pnp solve
    as this is the most common starting point.

    :param model_points: data type float
    :param image_points: data type float, matching dimensions to model_points
    :param camera_matrix: 3x3 intrinsic camera matrix, needed for cv2.solvePnP
    :param distortion_coefficients:
    :return: M2C: 4x4 homogeneous transformation matrix M2C, model to camera transformation matrix
    :return: C2M: 4x4 homogeneous transformation matrix C2M, camera to model transformation matrix
    """
    # find rotation and translation vectors
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                          distortion_coefficients)

    # obtain 3x3 rotation matrix, model to camera rotation, from Rodrigues rotation vector
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # obtain 3x3 rotation matrix, camera to model rotation,from the inverse rotation matrix which is equal
    # to its transpose
    inverse_rotation_matrix = rotation_matrix.T

    # obtain inverse translation vector, for camera to model translation, using -R.T*t.
    # Where t is the original translation vector
    inverse_translation_vector = np.negative(rotation_matrix).T.dot(translation_vector)

    # combine rotation matrix and translation vector into 4x4 homogeneous transformation matrix M2C
    M2C = np.concatenate((rotation_matrix, translation_vector), axis=1)   # 3x4
    # TODO: add row 0,0,0,1
    # M2C = np.concatenate((M2C, np.array([0,0,0,1])), axis=0)   # 4x4

    # combine inverse rotation matrix and inverse translation vector into 4x4 homogeneous transformation matrix C2M
    C2M = np.concatenate((inverse_translation_vector, inverse_translation_vector), axis=1)   # 3x4
    # TODO: add row 0,0,0,1
    # C2M = np.concatenate((C2M, np.array([0,0,0,1])), axis=0)   # 4x4

    # note: transformation matrices are homogeneous
    return M2C, C2M
