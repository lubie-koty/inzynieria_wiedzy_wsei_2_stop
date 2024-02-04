import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from fer import FER


if __name__ == '__main__':
    test_image = plt.imread(sys.argv[1])
    emotion_detection = FER(mtcnn=True)
    emotions = emotion_detection.detect_emotions(test_image)

    print(f'{emotions=}')

    dominant_emotion, emotion_score = emotion_detection.top_emotion(test_image)
    print(f'{dominant_emotion=}: {emotion_score=}')

    plt.imshow(test_image)
    plt.title(dominant_emotion)
    plt.show()
