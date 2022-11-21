import cv2
import os
import sys

if len(sys.argv) != 3:
    sys.exit('please input source directory and image extension to convert')

imageDir = sys.argv[1]
if not os.path.exists(imageDir):
    sys.exit(f'directory {imageDir} does not exist')
print('image directory:', imageDir)

allowedExtension = ['jfif', 'png']
extensionBefore = sys.argv[2]
if extensionBefore not in allowedExtension:
    sys.exit(f'only jfif and png image extensions are supported')
print('image extension before converting to jpg:', extensionBefore)

for imagePath in os.listdir(imageDir):
    if imagePath.endswith(extensionBefore):
        image = cv2.imread(imagePath)
        fileName = imagePath.split('.')[0]
        cv2.imwrite(f'1-{fileName}.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        os.remove(f'{imageDir}/{imagePath}')