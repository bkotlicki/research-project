import glob, os
import cv2

f = open("./faces/info.lst", "w+")

for i in range(0, 100):
    path = "./faces/" + str(i) + ".png"
    print(path)
    img = cv2.imread(path)
    #
    nf = str(i) + ".jpg"
    # counter = counter + 1
    #
    line = nf + " 1 0 0 50 50\n"
    f.write(line)
    #
    cv2.imwrite(nf, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
