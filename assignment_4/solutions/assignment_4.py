import cv2
import numpy as np

def harris_corner_detection(reference_image):
    block_size = 2
    ksize = 3
    k = 0.04

    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    dst = cv2.dilate(dst, None)

    marked_img = reference_image.copy()
    marked_img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return marked_img

def align_sift(image_to_align, reference_image, max_features=10, good_match_precent=0.7):
    """
    Brukte SIFT + FLANN + RANSAC her
    """
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    MIN_MATCH_COUNT = 10

    sift = cv2.SIFT_create()
    keypoints_align, descriptors_align = sift.detectAndCompute(gray_align, None)
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(descriptors_align, descriptors_ref, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda m: m.distance)

    if max_features and len(good_matches) > max_features:
        selected_matches = good_matches[:max_features]
    else:
        selected_matches = good_matches

    points_align = np.float32([keypoints_align[m.queryIdx].pt for m in selected_matches]).reshape(-1, 1, 2)
    points_ref = np.float32([keypoints_ref[m.trainIdx].pt for m in selected_matches]).reshape(-1, 1, 2)

    print(f"Number of selected matches: {len(selected_matches)}")
    if len(selected_matches) < MIN_MATCH_COUNT:
        raise ValueError(f"Need at least {MIN_MATCH_COUNT} matches")
    homography, mask = cv2.findHomography(points_align, points_ref, cv2.RANSAC, 5.0)

    ref_h, ref_w = reference_image.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, homography, (ref_w, ref_h))

    matches_image = cv2.drawMatches(image_to_align, keypoints_align, reference_image, keypoints_ref, selected_matches, None, matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,)

    return aligned_image, matches_image

if __name__ == "__main__":
    reference_image = cv2.imread("reference_img.png")
    align_img = cv2.imread("align_this.jpg")

    if reference_image is None or align_img is None:
        raise FileNotFoundError("Reference image or align_img was not found.")

    harris_out = harris_corner_detection(reference_image)
    cv2.imwrite("harris.png", harris_out)

    aligned_out, matches_out = align_sift(
        image_to_align=align_img,
        reference_image=reference_image,
        max_features=10,
        good_match_precent=0.7
    )

    cv2.imwrite("aligned.png", aligned_out)
    cv2.imwrite("matches.png", matches_out)