import os
import pandas
import sys
import xml.etree.ElementTree as xet

def extractXML(imageDir):
    coordinates = {'id':[], 'xmin':[], 'xmax':[], 'ymin':[], 'ymax':[]}
    missingXMLs = []
    for imagePath in os.listdir(imageDir):
        xmlPath = f'{imageDir}/{imagePath.replace("jpg", "xml")}'
        if not os.path.exists(xmlPath):
            missingXMLs.append(xmlPath)
            continue
        parsedXML = xet.parse(xmlPath)
        root = parsedXML.getroot()
        coordinate = root.find('object').find('bndbox')
        xmin = int(coordinate.find('xmin').text)
        xmax = int(coordinate.find('xmax').text)
        ymin = int(coordinate.find('ymin').text)
        ymax = int(coordinate.find('ymax').text)
        
        # default image
        coordinates['id'].append(imagePath)
        coordinates['xmin'].append(xmin)
        coordinates['xmax'].append(xmax)
        coordinates['ymin'].append(ymin)
        coordinates['ymax'].append(ymax)
        
        # image rotated 90-deg clockwise
        coordinates['id'].append(imagePath.replace('1-', '2-'))
        coordinates['xmin'].append(600-ymax)
        coordinates['xmax'].append(600-ymin)
        coordinates['ymin'].append(xmin)
        coordinates['ymax'].append(xmax)
        
        # image rotated 180-deg clockwise
        coordinates['id'].append(imagePath.replace('1-', '3-'))
        coordinates['xmin'].append(600-xmax)
        coordinates['xmax'].append(600-xmin)
        coordinates['ymin'].append(600-ymax)
        coordinates['ymax'].append(600-ymin)
        
        # image rotated 270-deg clockwise
        coordinates['id'].append(imagePath.replace('1-', '4-'))
        coordinates['xmin'].append(ymin)
        coordinates['xmax'].append(ymax)
        coordinates['ymin'].append(600-xmax)
        coordinates['ymax'].append(600-xmin)
    
    return coordinates, missingXMLs

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('please input source directory')
    
    imageDir = sys.argv[1]
    if not os.path.exists(imageDir):
        sys.exit(f'directory {imageDir} does not exist')
    print('image directory:', imageDir)

    coordinates, missingXMLs = extractXML(imageDir)

    if len(missingXMLs) != 0:
        sys.exit(f'list of missing XMLs: {missingXMLs}')
    pandas.DataFrame(data=coordinates).set_index('id').to_csv('target.csv')