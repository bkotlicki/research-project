import cv2
import json
import numpy as np

path_one = '../data/annotations.json'

json_dict = {}

# images = np.genfromtxt('./data/data_preparation/images/images_of_characters.csv', delimiter=',')
# characters = np.genfromtxt('./data/data_preparation/annotations/annotations_of_characters.csv', delimiter=',')

images = np.array([])
characters = np.array([])

images = images.reshape(images.shape[0] // 7500, 50, 50, 3)

with open(path_one) as out_file:
    json_dict = json.load(out_file)

print(json_dict)

for k in json_dict:
    img = cv2.imread('./data/resized/' + k)
    cs = json_dict[k]['Characters']
    bbs = json_dict[k]['Bounding boxes']
    if len(cs) == len(bbs):
        for i in range(0, len(cs)):
            bb = bbs[i]
            face = img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
            face = cv2.resize(face, (50, 50))
            face = np.reshape(face, (50, -1))
            face = np.reshape(face, (-1))
            if cs[i] != "9":
                images = np.append(images, face)
                characters = np.append(characters, int(cs[i]))

print(images.shape)
print(characters.shape)

np.savetxt('../data/data_preparation/images/images_of_characters.csv', images, delimiter=",")
np.savetxt('../data/data_preparation/annotations/annotations_of_characters.csv', characters, delimiter=",")
