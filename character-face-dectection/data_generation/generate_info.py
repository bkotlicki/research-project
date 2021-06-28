import glob
import os

import cv2

f = open("../../../opencv_workspace/dilbert/bg.txt", "w+")
# f = open("../../opencv_workspace/dilbert/pos/info.lst", "w+")

os.chdir('../../../opencv_workspace/dilbert/negs')
for file in glob.glob("*.png"):
    path = '../../opencv_workspace/dilbert/negs/' + file
    img = cv2.imread(path)
    line = "negs/" + file + "\n"
    f.write(line)
