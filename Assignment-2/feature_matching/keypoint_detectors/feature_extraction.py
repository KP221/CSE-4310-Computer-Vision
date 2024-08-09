"""Feature Extraction using SIFT, HOG, and Visual Vocabulary.

This module provides functions for extracting SIFT and HOG features,
building visual vocabularies, and creating histograms of visual words.

Functions:
    - extract_sift_keypoints_and_descriptors
    - match_descriptors
    - display_keypoint_matches
    - show_sample_images
    - sift_feature_extraction
    - hog_feature_extraction
    - create_visual_vocabulary
    - create_histograms
    - apply_tfidf
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from PIL import Image
from scipy.sparse import spmatrix
from scipy.spatial.distance import cdist
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import SIFT, hog
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm


__all__ = [
    "apply_tfidf",
    "create_histograms",
    "create_visual_vocabulary",
    "match_descriptors",
    "extract_sift_keypoints_and_descriptors",
    "sift_feature_extraction",
    "hog_feature_extraction",
    "display_keypoint_matches",
    "show_sample_images",
]


def detect_sift_keypoints_descriptors(
    target_img: np.ndarray, source_img: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract SIFT keypoints and descriptors from two images.

    Args:
        target_img: The target image.
        source_img: The source image.

    Returns:
        keypoints_target: Keypoints from the target image.
        descriptors_target: Descriptors from the target image.
        keypoints_source: Keypoints from the source image.
        descriptors_source: Descriptors from the source image.
    """
    sift_target = SIFT()
    sift_source = SIFT()
    sift_target.detect_and_extract(target_img)
    sift_source.detect_and_extract(source_img)
    keypoints_target = sift_target.keypoints
    descriptors_target = sift_target.descriptors
    keypoints_source = sift_source.keypoints
    descriptors_source = sift_source.descriptors

    return keypoints_target, descriptors_target, keypoints_source, descriptors_source


def custom_match_descriptors(
    descriptors_target: np.ndarray, descriptors_source: np.ndarray, cross_check: bool = True
) -> np.ndarray:
    """Match descriptors between two images.

    Args:
        descriptors_target: Descriptors from the target image.
        descriptors_source: Descriptors from the source image.
        cross_check: Whether to use cross-checking to find mutual matches. (default: True)

    Returns:
        matches: The indices of the matched descriptors.
    """
    distances = cdist(descriptors_target, descriptors_source, "euclidean")
    nearest_neighbors_target = np.argmin(distances, axis=1)

    if cross_check:
        nearest_neighbors_source = np.argmin(distances, axis=0)
        mutual_matches = [i for i, j in enumerate(nearest_neighbors_target) if nearest_neighbors_source[j] == i]
        return np.array([[i, j] for i, j in enumerate(nearest_neighbors_target) if i in mutual_matches])
    
    return np.column_stack([np.arange(descriptors_target.shape[0]), nearest_neighbors_target])


def plot_keypoint_matches(
    target_img: np.ndarray,
    keypoints_target: np.ndarray,
    source_img: np.ndarray,
    keypoints_source: np.ndarray,
    matches: np.ndarray,
) -> None:
    """Display the matched keypoints between two images.

    Args:
        target_img: The target image.
        keypoints_target: Keypoints from the target image.
        source_img: The source image.
        keypoints_source: Keypoints from the source image.
        matches: The indices of the matched keypoints.
    """
    target_points = keypoints_target[matches[:, 0]]
    source_points = keypoints_source[matches[:, 1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(target_img, cmap="gray")
    ax2.imshow(source_img, cmap="gray")

    for i in range(source_points.shape[0]):
        coord_target = (target_points[i, 1], target_points[i, 0])
        coord_source = (source_points[i, 1], source_points[i, 0])
        con = ConnectionPatch(
            xyA=coord_source,
            xyB=coord_target,
            coordsA="data",
            coordsB="data",
            axesA=ax2,
            axesB=ax1,
            color="red",
        )
        ax2.add_artist(con)
        ax1.plot(target_points[i, 1], target_points[i, 0], "ro")
        ax2.plot(source_points[i, 1], source_points[i, 0], "ro")

    plt.show()


def show_sample_images(data: np.ndarray) -> None:
    """Show the first 10 images in the dataset.

    Args:
        data: The input images.
    """
    _, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes):
        ax.imshow(data[i], cmap="gray")
        ax.axis("off")
    plt.show()


def sift_feature_extraction(X_data: np.ndarray, y_data: np.ndarray) -> tuple[list, list]:
    """Extract SIFT features from the images.

    Args:
        X_data: The input images.
        y_data: The image labels.

    Returns:
        descriptors_list: A list of SIFT descriptors for each image.
        y_features: The image labels.
    """
    sift = SIFT()
    descriptors_list = []
    y_features = []

    for img in tqdm(range(X_data.shape[0]), desc="Processing images"):
        try:
            sift.detect_and_extract(X_data[img])
            descriptors_list.append(sift.descriptors)
            y_features.append(y_data[img])
        except:
            pass

    return descriptors_list, y_features


def hog_feature_extraction(X_data: np.ndarray, y_data: np.ndarray) -> tuple[list, list]:
    """Extract HOG features from the images.

    Args:
        X_data: The input images.
        y_data: The image labels.

    Returns:
        descriptors_list: A list of HOG descriptors for each image.
        y_features: The image labels.
    """
    descriptors_list = []
    y_features = []

    for img in tqdm(range(X_data.shape[0]), desc="Processing images"):
        try:
            hog_descriptors = hog(X_data[img], pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            descriptors_list.append(hog_descriptors.reshape(1, -1))
            y_features.append(y_data[img])
        except:
            pass

    return descriptors_list, y_features


def create_visual_vocabulary(descriptor_list: list, vocab_size: int) -> KMeans:
    """Build a visual vocabulary using KMeans clustering.

    Args:
        descriptor_list: A list of SIFT descriptors for each image.
        vocab_size: The number of visual words to use.

    Returns:
        kmeans: The KMeans model trained on the descriptors.
    """
    descriptors = np.concatenate(descriptor_list)
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(descriptors)

    return kmeans


def create_histograms(descriptor_list: list, kmeans: KMeans, vocab_size: int) -> np.ndarray:
    """Build histograms of visual words for each image.

    Args:
        descriptor_list: A list of SIFT descriptors for each image.
        kmeans: The KMeans model trained on the descriptors.
        vocab_size: The number of visual words to use.

    Returns:
        histograms: An array of histograms of visual words for each image.
    """
    histograms = []

    for descriptors in tqdm(descriptor_list, desc="Building histograms"):
        clusters = kmeans.predict(descriptors)
        histogram, _ = np.histogram(clusters, bins=vocab_size, range=(0, vocab_size))
        histograms.append(histogram)

    return np.array(histograms)


def apply_tfidf(histograms: np.ndarray) -> spmatrix:
    """Adjust the frequency of visual words using TF-IDF.

    Args:
        histograms: An array of histograms of visual words for each image.

    Returns:
        The histogram data transformed using TF-IDF.
    """
    tfidf = TfidfTransformer()
    tfidf.fit(histograms)
    return tfidf.transform(histograms)


def main() -> None:
    """Main function to demonstrate the feature extraction process."""
    target_img_rgb = np.asarray(Image.open("./data/Rainier1.png"))
    source_img_rgb = np.asarray(Image.open("./data/Rainier2.png"))

    if target_img_rgb.shape[2] == 4:
        target_img_rgb = rgba2rgb(target_img_rgb)
    if source_img_rgb.shape[2] == 4:
        source_img_rgb = rgba2rgb(source_img_rgb)

    target_img = rgb2gray(target_img_rgb)
    source_img = rgb2gray(source_img_rgb)

    print("Detecting SIFT keypoints and descriptors...")
    keypoints_target, descriptors_target, keypoints_source, descriptors_source = detect_sift_keypoints_descriptors(target_img, source_img)

    print("Matching descriptors between images...")
    matches = custom_match_descriptors(descriptors_target, descriptors_source, cross_check=True)

    print(f"Number of matches: {len(matches)}")
    print("Plotting the matched keypoints...")
    plot_keypoint_matches(target_img, keypoints_target, source_img, keypoints_source, matches)

    print("Beginning feature extraction process...")

    data = np.load("./cifar10.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    X_train_rgb = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    X_test_rgb = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

    X_train_gray = rgb2gray(X_train_rgb)
    X_test_gray = rgb2gray(X_test_rgb)

    print("Visualizing the first 10 images of training set...")
    show_sample_images(X_train_gray)
    print("Visualizing the first 10 images of test set...")
    show_sample_images(X_test_gray)

    print("Extracting SIFT features from the training data...")
    X_train_sift, y_train_sift = sift_feature_extraction(X_train_gray, y_train)
    print(f"Number of training SIFT features: {len(X_train_sift)}")

    print("Extracting SIFT features from the testing data...")
    X_test_sift, y_test_sift = sift_feature_extraction(X_test_gray, y_test)
    print(f"Number of testing SIFT features: {len(X_test_sift)}")

    vocab_size = 50
    print(f"Building visual vocabulary with {vocab_size} words for training set...")
    sift_vocab_train = create_visual_vocabulary(X_train_sift, vocab_size)
    print(f"Building visual vocabulary with {vocab_size} words for testing set...")
    sift_vocab_test = create_visual_vocabulary(X_test_sift, vocab_size)

    print("Building histograms for training set...")
    X_train_sift_hist = create_histograms(X_train_sift, sift_vocab_train, vocab_size)
    print("Building histograms for testing set...")
    X_test_sift_hist = create_histograms(X_test_sift, sift_vocab_test, vocab_size)

    print("Adjusting frequency using TF-IDF for each set...")
    X_train_sift_tfidf = apply_tfidf(X_train_sift_hist)
    X_test_sift_tfidf = apply_tfidf(X_test_sift_hist)

    print("Saving processed data as 'processed_cifar10_sift.npz'...")
    sift_processed_data = {
        "X_train": X_train_sift_tfidf.toarray(),
        "X_test": X_test_sift_tfidf.toarray(),
        "y_train": y_train_sift,
        "y_test": y_test_sift,
    }
    np.savez("processed_cifar10_sift.npz", **sift_processed_data)

    print("Extracting HOG features from the training data...")
    X_train_hog, y_train_hog = hog_feature_extraction(X_train_gray, y_train)
    print(f"Number of training HOG features: {len(X_train_hog)}")

    print("Extracting HOG features from the testing data...")
    X_test_hog, y_test_hog = hog_feature_extraction(X_test_gray, y_test)
    print(f"Number of testing HOG features: {len(X_test_hog)}")

    print(f"Building visual vocabulary with {vocab_size} words for HOG training set...")
    hog_vocab_train = create_visual_vocabulary(X_train_hog, vocab_size)
    print(f"Building visual vocabulary with {vocab_size} words for HOG testing set...")
    hog_vocab_test = create_visual_vocabulary(X_test_hog, vocab_size)

    print("Building histograms for HOG training set...")
    X_train_hog_hist = create_histograms(X_train_hog, hog_vocab_train, vocab_size)
    print("Building histograms for HOG testing set...")
    X_test_hog_hist = create_histograms(X_test_hog, hog_vocab_test, vocab_size)

    print("Adjusting frequency using TF-IDF for HOG sets...")
    X_train_hog_tfidf = apply_tfidf(X_train_hog_hist)
    X_test_hog_tfidf = apply_tfidf(X_test_hog_hist)

    print("Saving processed HOG data as 'processed_cifar10_hog.npz'...")
    hog_processed_data = {
        "X_train": X_train_hog_tfidf.toarray(),
        "X_test": X_test_hog_tfidf.toarray(),
        "y_train": y_train_hog,
        "y_test": y_test_hog,
    }
    np.savez("processed_cifar10_hog.npz", **hog_processed_data)

    print("Feature extraction complete!")
    print("Evaluate the processed data using 'evaluate_sift.py' and 'evaluate_hog.py'.")


if __name__ == "__main__":
    main()
