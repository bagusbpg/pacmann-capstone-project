# How to train a model
## Training data preparation
- Collect images of cars with license plate for training purpose. It is better to have images of various brightness, orientation, perspective, scale, sharpness, etc.
- Create `PASCAL Visual Object Classes Challenge` for each image, identifying the location of license plate. In case `PASCAL VOC` is created in XML, you may run extractXML.py in data directory to extract `xmin`, `xmax`, `ymin`, and `ymax` coordinates and save them to csv file for later use in training model. Required argument to run this script is path to directory containing images and XML files.
```bash
python ./data/extractXML.py ./data/images
```
Beside XML, JSON is also valid format to store `PASCAL VOC`, you may create your own script to do the same thing as extractXML.py.
## Training model
- Run train.py in train directory.
```bash
python ./train/train.py
```
- In general, what it does are (1) splitting training, validation, and test for feature-dataset, (2) augment images, i.e. rotating each image 90-degrees, 180-degrees, and 270-degrees clockwise to artificially make dataset four times bigger, (3) preprocessing each image, i.e. channel-wise standardization to make zero mean and unit standard deviation of dataset, (4) save mean and standard deviation of training dataset to be used for inference, (5) splitting training, validation, and test for target-dataset, which comes from extracted Pascal VOC XMLs described earlier, (6) define model, and finallny (7) fit the model and save it. Model fitting is equipped with EarlyStopping to prevent overfitting.
