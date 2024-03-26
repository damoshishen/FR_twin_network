"""
    AT&T Database数据集读取
                        --2023.3.24 物理系的计算机选手
"""
import matplotlib.pyplot as plt
import numpy as np

datasetFaces = []

for person in range(1, 41):
    temp = []

    for pose in range(1, 11):
        data = plt.imread('../data/s' + str(person) + '/' + str(pose) + '.pgm')
        temp.append(data)

    datasetFaces.append(np.array(temp))

datasetFaces = np.array(datasetFaces)
np.save('../data_Generated/datasetFacesORL.npy', datasetFaces)

print('Total number of datasets:', len(datasetFaces))
print('Dataset size:', datasetFaces.shape)