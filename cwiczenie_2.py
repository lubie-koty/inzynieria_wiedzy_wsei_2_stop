import cv2
import numpy as np
import matplotlib.pyplot as plt

K = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
cluster_colors = {
    0: [255, 255, 255],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [0, 0, 0]
}

if __name__ == '__main__':
    image = cv2.imread('image.jpg')
    pixel_values = np.float32(image.reshape((-1, 3)))

    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()].reshape(image.shape)
    masked_image = np.copy(image).reshape((-1, 3))

    for cluster_key, colors in cluster_colors.items():
        masked_image[labels == cluster_key] = colors
    masked_image = masked_image.reshape(image.shape)

    plt.imshow(masked_image)
    plt.show()
