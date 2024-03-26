"""
    正负图像对的建立
"""
import numpy as np


def get_data_from_npy(size, total_sample_size, dataset_path='../data_Generated/datasetFacesORL.npy'):
    # Load the dataset
    data = np.load(dataset_path)

    # Reduce the size of the images
    data = data[:, :, ::size, ::size]

    # Get the new size
    dim1 = data.shape[2]
    dim2 = data.shape[3]

    count = 0

    # Initialize the numpy arrays for storing pairs
    x_genuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.ones([total_sample_size, 1])  # Genuine pairs have label 1

    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])  # Impostor pairs have label 0

    # Generate genuine pairs
    for i in range(40):
        for j in range(int(total_sample_size / 40)):
            while True:
                ind1, ind2 = np.random.randint(10, size=2)
                if ind1 != ind2:
                    break

            x_genuine_pair[count, 0, 0, :, :] = data[i, ind1]
            x_genuine_pair[count, 1, 0, :, :] = data[i, ind2]
            count += 1

    # Reset count for impostor pairs
    count = 0

    # Generate impostor pairs
    for i in range(int(total_sample_size / 10)):
        for j in range(10):
            while True:
                ind1, ind2 = np.random.randint(40, size=2)
                if ind1 != ind2:
                    break

            img1 = data[ind1, j]
            img2 = data[ind2, j]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            count += 1

    # Concatenate genuine and impostor pairs
    X = np.concatenate([x_genuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y
