import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    image = cv2.cvtColor(
        cv2.imread('image.jpg'),
        cv2.COLOR_BGR2RGB
    )
    pixel_values = np.float32(image.reshape((-1, 3)))

    print(f'Ksztalt tablicy pikseli: {pixel_values.shape}')

    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()].reshape(image.shape)

    plt.imshow(segmented_image)
    plt.show()
