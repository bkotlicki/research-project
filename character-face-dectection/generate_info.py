import glob
import os

import cv2

f = open("../../opencv_workspace/dilbert/bg.txt", "w+")
# f = open("../../opencv_workspace/dilbert/pos/info.lst", "w+")

os.chdir('../../opencv_workspace/dilbert/negs')
for file in glob.glob("*.png"):
    path = '../../opencv_workspace/dilbert/negs/' + file
    img = cv2.imread(path)
    line = "negs/" + file + "\n"
    f.write(line)
    # line = file + " 1 0 0 50 50\n"
    # f.write(line)


# for i in range(0, 100):
#     path = "./faces/" + str(i) + ".png"
#     print(path)
#     img = cv2.imread(path)
#     #
#     nf = str(i) + ".jpg"
#     # counter = counter + 1
#     #
#     line = nf + " 1 0 0 50 50\n"
#     f.write(line)
#     #
#     cv2.imwrite(nf, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
