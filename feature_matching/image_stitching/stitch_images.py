import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

"""Image Stitching using SIFT and RANSAC with Affine and Projective Transformations.

Functions:
    - compute_affine_transform
    - compute_projective_transform
    - ransac
    - plot_best_matches
    - stitch_images
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import warp
from feature_matching.keypoint_detectors.feature_extraction import (
    custom_match_descriptors,
    detect_sift_keypoints_descriptors,
    plot_keypoint_matches,
)

__all__ = [
    "compute_affine_transform",
    "compute_projective_transform",
    "plot_best_matches",
    "ransac",
    "stitch_images",
]
__author__ = "Kshitij Patel"


def compute_affine_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the affine transformation matrix from src_points to dst_points.

    Args:
        src_points: Source points.
        dst_points: Destination points.

    Returns:
        ndarray: Affine transformation matrix.
    """
    A = np.zeros((2 * src_points.shape[0], 6))
    for i in range(src_points.shape[0]):
        A[2 * i] = [src_points[i][0], src_points[i][1], 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, src_points[i][0], src_points[i][1], 1]
    B = dst_points.flatten()
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [0, 0, 1]])


def compute_projective_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute the projective transformation matrix from src_points to dst_points.

    Args:
        src_points: Source points.
        dst_points: Destination points.

    Returns:
        ndarray: Projective transformation matrix.
    """
    A = np.zeros((2 * src_points.shape[0], 8))
    for i in range(src_points.shape[0]):
        x, y = src_points[i][0], src_points[i][1]
        xp, yp = dst_points[i][0], dst_points[i][1]
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
    B = dst_points.flatten()
    x = np.linalg.lstsq(A, B, rcond=None)[0]
    return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], 1]])


def ransac(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    model_func: callable,
    min_samples: int,
    threshold: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC (RANdom SAmple Consensus) algorithm to estimate a transformation matrix.

    Args:
        src_points: Source points.
        dst_points: Destination points.
        model_func: Function to estimate the transformation matrix.
        min_samples: Minimum number of samples to estimate the transformation matrix.
        threshold: Maximum distance to consider a point as an inlier.
        max_iterations: Maximum number of iterations.

    Returns:
        ndarray: Best transformation matrix.
        ndarray: Indices of the best inliers.
    """
    best_matrix = None
    best_inliers_indices = None
    best_inliers = 0

    for _ in range(max_iterations):
        # Randomly select min_samples points
        sample_indices = np.random.choice(src_points.shape[0], min_samples, replace=False)
        sample_src = src_points[sample_indices]
        sample_dst = dst_points[sample_indices]

        # Estimate transformation matrix using the selected points
        transformation_matrix = model_func(sample_src, sample_dst)

        # Identify inliers
        inliers = []
        for i in range(src_points.shape[0]):
            src_point = np.append(src_points[i], 1)  # Convert to homogeneous coordinates
            estimated_dst_point = transformation_matrix @ src_point
            estimated_dst_point /= estimated_dst_point[2]  # Perspective divide
            actual_dst_point = np.append(dst_points[i], 1)

            # Check if the distance is within the threshold
            if np.linalg.norm(estimated_dst_point - actual_dst_point) < threshold:
                inliers.append(i)

        # Update the best transformation matrix if this model has more inliers
        if len(inliers) > best_inliers:
            best_inliers = len(inliers)
            best_matrix = transformation_matrix
            best_inliers_indices = inliers

    return best_matrix, best_inliers_indices


def plot_best_matches(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_keypoints: np.ndarray,
    dst_keypoints: np.ndarray,
    best_matches: np.ndarray,
    matches: np.ndarray,
) -> None:
    """Plotting the best matches between the source and destination images.

    Args:
        src_img: Source image.
        dst_img: Destination image.
        src_keypoints: Source keypoints.
        dst_keypoints: Destination keypoints.
        best_matches: Indices of the best matches.
        matches: All matches.
    """
    # Allocating the best matches
    src_best = src_keypoints[matches[best_matches, 1]][:, ::-1]
    dst_best = dst_keypoints[matches[best_matches, 0]][:, ::-1]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img)
    ax2.imshow(src_img)

    for i in range(src_best.shape[0]):
        coordB = [dst_best[i, 0], dst_best[i, 1]]
        coordA = [src_best[i, 0], src_best[i, 1]]
        con = ConnectionPatch(
            xyA=coordA,
            xyB=coordB,
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            color="red",
        )
        ax2.add_artist(con)
        ax1.plot(dst_best[i, 0], dst_best[i, 1], "ro")
        ax2.plot(src_best[i, 0], src_best[i, 1], "ro")

    plt.show()


def stitch_images(src_img: np.ndarray, dst_img: np.ndarray, model_matrix: np.ndarray) -> None:
    """Stitching the source image to the destination image using the given transformation matrix.

    Args:
        src_img: Source image.
        dst_img: Destination image.
        model_matrix: Model matrix.
    """
    output_shape = dst_img.shape
    src_warped = warp(src_img, np.linalg.inv(model_matrix), output_shape=output_shape)

    # Create a mask to blend the images
    src_mask = src_warped != 0

    # Blend the images
    result = dst_img.copy()
    result[src_mask] = src_warped[src_mask]

    # Display the stitched image
    plt.imshow(result)
    plt.axis("off")
    plt.show()


def main() -> None:
    """Main function to stitch two images together using SIFT and RANSAC."""
    # Load the images
    dst_img_rgb = np.asarray(Image.open("./data/Rainier1.png"))
    src_img_rgb = np.asarray(Image.open("./data/Rainier2.png"))

    # Converting RGBA to RGB if necessary
    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    # Convert images to grayscale
    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)

    print("Detecting SIFT keypoints and descriptors...")
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_sift_keypoints_descriptors(
        dst_img, src_img
    )

    print("Matching descriptors between images...")
    matches = custom_match_descriptors(descriptors1, descriptors2, cross_check=True)

    print("Plotting the matched keypoints...")
    plot_keypoint_matches(dst_img, keypoints1, src_img, keypoints2, matches)

    # Extract the matched keypoints
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    print("\n\t----Affine Transformation----")
    print("Computing Affine Transformation without RANSAC...")

    compute_affine = compute_affine_transform(src[:, ::-1], dst[:, ::-1])

    print(f"Affine Matrix without RANSAC: \n{compute_affine}")
    print("\nComputing Affine Transformation with RANSAC...")

    affine_sk_M, affine_sk_best = ransac(
        src[:, ::-1],
        dst[:, ::-1],
        compute_affine_transform,
        4,
        1,
        300,
    )

    print(f"Affine Matrix with RANSAC: \n{affine_sk_M}")
    print("\nPlotting the best matches from the Affine Transformation...")

    plot_best_matches(src_img_rgb, dst_img_rgb, keypoints2, keypoints1, affine_sk_best, matches)

    print("Stitching the images...")
    stitch_images(src_img_rgb, dst_img_rgb, affine_sk_M)

    print("\n\t----Projective Transformation----")
    print("Computing Projective Transformation without RANSAC...")

    projective_transform = compute_projective_transform(src[:, ::-1], dst[:, ::-1])

    print(f"Projective Matrix without RANSAC: \n{projective_transform}")
    print("\nComputing Projective Transformation with RANSAC...")

    projective_sk_M, projective_sk_best = ransac(
        src[:, ::-1],
        dst[:, ::-1],
        compute_projective_transform,
        4,
        1,
        300,
    )

    print(f"Projective Matrix with RANSAC: \n{projective_sk_M}")
    print("\nPlotting the best matches from the Projective Transformation...")

    plot_best_matches(src_img_rgb, dst_img_rgb, keypoints2, keypoints1, projective_sk_best, matches)

    print("Stitching the images...")
    stitch_images(src_img_rgb, dst_img_rgb, projective_sk_M)


if __name__ == "__main__":
    main()
