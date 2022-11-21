import os
import xml.etree.ElementTree as xet

coordinates = {}
brokenXMLs = []
for imagePath in path:
    xmlPath = imagePath.replace('jpg', 'xml')
    if not os.path.exists(xmlPath):
        coordinates[imagePath] = None
        brokenXMLs.append(xmlPath)
        continue
    parsedXML = xet.parse(xmlPath)
    root = parsedXML.getroot()
    coordinate = root.find('object').find('bndbox')
    xmin = int(coordinate.find('xmin').text)
    xmax = int(coordinate.find('xmax').text)
    ymin = int(coordinate.find('ymin').text)
    ymax = int(coordinate.find('ymax').text)
    coordinates[imagePath] = [xmin, xmax, ymin, ymax]
    
if len(brokenXMLs) == 0:
    coordinates2 = {}
    coordinates3 = {}
    coordinates4 = {}
    if len(brokenXMLs) == 0:
        for key, value in coordinates1.items():
            xmin1, xmax1, ymin1, ymax1 = value
            coordinates2[key] = 600-ymax1, 600-ymin1, xmin1, xmax1
            coordinates3[key] = 600-xmax1, 600-xmin1, 600-ymax1, 600-ymin1
            coordinates4[key] = ymin1, ymax1, 600-xmax1, 600-xmin1
else:
    print(f'List of broken XMLs: {brokenXMLs}')