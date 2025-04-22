from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d


def knee_point_detection(x, y) -> Tuple[int, float, float]:
    """
    Detect the 'knee' or 'elbow' point in a curve using the "maximum distance to chord" method.

    This method finds the point in a curve where it bends most — often used to identify
    a threshold, transition, or characteristic point in a signal or cumulative plot.

    Algorithm:
    1. Normalize x and y to [0, 1] range.
    2. Draw a straight line (chord) between the first and last point of the curve.
    3. Compute the perpendicular distance from each point on the curve to this line.
    4. Identify the index with the maximum distance — this is the 'knee' point.

    Parameters:
    ----------
    x : np.ndarray
        X-values of the curve.
    y : np.ndarray
        Y-values of the curve.

    Returns:
    -------
    knee_index : int
        Index of the detected knee point.
    x[knee_index] : float
        X-value of the knee point.
    y[knee_index] : float
        Y-value of the knee point.
    """
    # Normalize x and y to have values between 0 and 1
    x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Line connecting the first and last points
    line_vec = np.array(
        [x_normalized[-1] - x_normalized[0], y_normalized[-1] - y_normalized[0]]
    )
    line_vec_norm = np.sqrt(
        line_vec[0] ** 2 + line_vec[1] ** 2
    )  # Length of the line vector
    line_vec = line_vec / line_vec_norm  # Normalize the line vector

    # Calculate the distances from all points to the line
    vec_from_first = np.column_stack(
        [x_normalized - x_normalized[0], y_normalized - y_normalized[0]]
    )
    scalar_proj = np.dot(vec_from_first, line_vec)
    vec_along_line = np.outer(scalar_proj, line_vec)
    vec_to_line = vec_from_first - vec_along_line

    # Compute the distances
    distances = np.sqrt(np.sum(vec_to_line**2, axis=1))

    # Find the index of the maximum distance (knee point)
    knee_index = np.argmax(distances)

    return knee_index, x[knee_index], y[knee_index]


def outlier_detection_std(
    x: np.ndarray, y: np.ndarray, threshold: float = 3, max_iterations: int = 1
) -> np.ndarray:
    """
    Detect outliers in a signal based on the standard deviation of the signal.

    Parameters:
    x (numpy array): X-values of the signal.
    y (numpy array): Y-values of the signal.
    threshold (float): Threshold for the standard deviation to detect outliers.
    max_iterations (int): Maximum number of iterations to detect outliers.

    Returns:
    numpy array: A boolean array where True indicates an outlier in the signal.
    """

    is_outlier = np.zeros_like(y, dtype=bool)
    n_outliers = 0
    for _ in range(max_iterations):
        new_outliers = y[~is_outlier]
        if new_outliers.size == 0:  # Handle the case of no non-outliers left
            break

        std = np.nanstd(new_outliers)
        mean = np.nanmean(new_outliers)
        is_outlier[~is_outlier] = np.abs(y[~is_outlier] - mean) > threshold * std

        current_outliers = is_outlier.sum()
        if current_outliers == n_outliers:
            break
        n_outliers = current_outliers

    return is_outlier


def estimate_baseline_regions(
    x: np.ndarray,
    y: np.ndarray,
    pre_flatted_y: Optional[np.ndarray] = None,
    window: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the baseline region of a curve, useful for background correction or peak detection.

    This function uses a combination of smoothing, percentile thresholding, and outlier rejection
    to detect points in the signal that represent baseline regions (i.e., non-peak regions).

    Algorithm:
    ----------
    1. Optionally smooth the signal using a Gaussian filter.
    2. Normalize the smoothed signal and original signal.
    3. Create a cumulative histogram of low-intensity values.
    4. Use the "knee point" of this cumulative curve as the threshold for baseline.
    5. Mark points below this threshold as baseline candidates.
    6. Remove statistical outliers from these baseline candidates.
    7. Smooth the resulting mask to reduce fragmentation.
    8. Linearly interpolate the baseline using these identified regions.

    Parameters:
    ----------
    x : np.ndarray
        X-axis values (e.g., time or volume).
    y : np.ndarray
        Y-axis values (e.g., intensity or signal).
    pre_flatted_y : np.ndarray, optional
        If provided, use this pre-processed version of `y` for baseline detection.
    window : float, optional
        Smoothing window size:
            - If `None`: default to 1% of total data points.
            - If float: interpreted as a range in x-units and converted to number of points.

    Returns:
    -------
    y_bl : np.ndarray
        Interpolated baseline over the full range of `x`.
    is_baseline_region : np.ndarray
        Boolean mask indicating baseline regions (`True` = baseline).
    """

    if window is None:
        window = len(y) / 100  # default value as absolute number of points
    else:
        # if window is a float, the number of points is calculated as as if window is in x units
        window = int(window / (np.median(np.diff(x)) if x is not None else 1) + 0.5)

    window = max(1, window)

    if pre_flatted_y is None:
        pre_flatted_y = y

    smoothed_baseline_corrected_y = gaussian_filter1d(pre_flatted_y, window)

    percentage_cutoff = np.arange(0, 1, 0.0001)

    abs_smoothed_baseline_corrected_y = (
        smoothed_baseline_corrected_y + smoothed_baseline_corrected_y.min()
    )

    norm_smoothed_y = abs_smoothed_baseline_corrected_y / np.max(
        abs_smoothed_baseline_corrected_y
    )

    norm_y = np.abs(pre_flatted_y)
    norm_y = norm_y / np.max(norm_y)

    cumulative_n_points = np.sum(norm_smoothed_y[:, None] < percentage_cutoff, axis=0)
    cumulative_n_points = cumulative_n_points / np.max(cumulative_n_points)

    elbow_idx, elbow_x, elbow_y = knee_point_detection(
        percentage_cutoff, cumulative_n_points
    )

    is_baseline_region = norm_y <= elbow_x

    outliers = outlier_detection_std(
        x[is_baseline_region],
        norm_y[is_baseline_region],
        threshold=3,
        max_iterations=9,
    )

    is_baseline_region[is_baseline_region] &= ~outliers

    is_baseline_region = is_baseline_region.astype(float)

    smoothed_is_baseline_region = gaussian_filter1d(is_baseline_region, window)
    smoothed_is_baseline_region = (
        smoothed_is_baseline_region >= 0.99 * smoothed_is_baseline_region.max()
    )

    if np.any(smoothed_is_baseline_region):
        y_bl = np.interp(
            x, x[smoothed_is_baseline_region], y[smoothed_is_baseline_region]
        )
    else:
        y_bl = np.zeros_like(y)

    return y_bl, smoothed_is_baseline_region
