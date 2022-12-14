import cv2
import os
import sys

def resize(imageDir):
    processedCount = 0
    for imagePath in os.listdir(imageDir):
        if imagePath.endswith('jpg'):
            image = cv2.imread(f'{imageDir}/{imagePath}', cv2.IMREAD_UNCHANGED)
            height, width, _ = image.shape
            if height == 600 and width == 600:
                continue
            
            resizedImage = cv2.resize(image, (600,600), interpolation = cv2.INTER_AREA)
            cv2.imwrite(f'{imageDir}/{imagePath}', resizedImage)
            processedCount += 1
    print(f'successfully resized {processedCount} images')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('please input source directory')

    imageDir = sys.argv[1]
    if not os.path.exists(imageDir):
        sys.exit(f'directory {imageDir} does not exist')
    print('image directory:', imageDir)

    resize(imageDir)