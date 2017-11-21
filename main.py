import os

import MachineLearning.TFApi as tfapi

if __name__ == "__main__":
    model = tfapi.Model()
    model.load_frozen_model()
    model.load_label_map()
    dir = './test_images'
    image_paths = [os.path.join(dir, 'image{}.jpg'.format(i)) for i in range(1, 3)]
    model.predict(image_path=image_paths[1])
