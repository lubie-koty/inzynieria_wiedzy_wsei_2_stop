import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    image = cv2.cvtColor(
        cv2.imread('image_2.jpg'),
        cv2.COLOR_BGR2RGB
    )
    data = image.reshape((-1, 3))
    labels = np.sum(data, axis=1) > 255 * 3 / 2

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(data, labels)

    segmented_image = knn.predict(data).reshape(image.shape[:2])

    plt.imshow(segmented_image, cmap='plasma')
    plt.show()
