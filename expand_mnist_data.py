"""
This program takes the original 50,000 MNIST training images,
and create an expanded set of 250, 000 images, by translating each image
up, down, left, and right by NUM_PIXELS.
Save the data for the new images in ./data/mnist_expanded.pkl.gz
"""

## Libraries
import pickle
import gzip
import os.path
import random

# Third-party lib
import numpy as np

NUM_PIXELS = 1 # default displacement factor

def expand():
    print("Expanding MNIST dataset")

    if os.path.exists("./data/mnist_expanded.pkl.gz"):
        print("Expanded set already exists. Exiting..")
    else:
        f = gzip.open("./data/mnist.pkl.gz", 'rb')
        # Note from MNIST docs: The MNIST pickle returns data as a tuple containing the training data,
        #                       the validation data, and the test data.
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        f.close()
        expanded_training_input_output = []
        j = 0 # counter
        for x, y in zip(training_data[0], training_data[1]):
            expanded_training_input_output.append((x,y))
            image = np.reshape(x, (-1, 28))

            # tracking progress
            j += 1
            if j % 5000 == 0:
                print("Data augmentation completed: ", j)
                
            for disp, axis, index in [(1, 0, 0), (-1, 0, 27), (1, 1, 0), (-1, 1, 27)]:
                new_img = np.roll(image, disp, axis) # Note axis 0 represents the rows, 1 rep cols
                if axis == 0:
                    new_img[index, :] = np.zeros(28)
                else:
                    new_img[:, index] = np.zeros(28)
                expanded_training_input_output.append((np.reshape(new_img, 784), y))
        random.shuffle(expanded_training_input_output)
        expanded_training_data = [list(d) for d in zip(*expanded_training_input_output)]
        print("All 250,000 images processed!")
        print("Saving expanded data. May take several minutes...")
        f = gzip.open("./data/mnist_expanded.pkl.gz", "w")
        pickle.dump((expanded_training_data, validation_data, test_data), f)
        f.close()
        print("All done!")
            
