# About the project
This application provides REST API for car license number recognition. Sample implementations may include car parking billing system or likewise.
# Installation
First, clone this repository,
```bash
git clone https://github.com/bagusbpg/pacmann-capstone-project.git
```
Then, set up configuration file, place it in root directory
```json
{
    "DATABASE": {
        "DRIVER": ,
        "HOST": ,
        "PORT": ,
        "USERNAME": ,
        "PASSWORD": ,
        "NAME":
    },
    "APP": {
        "HOST": ,
        "PORT": ,
        "MAXIMUM_IMAGE_UPLOAD_SIZE": ,
        "THRESHOLD_OF_SIMILARITY":
    }
}
```
It is recommended to create new python environment first and then proceed with installing the requirement.
```bash
pip install -r requirements.txt
```
Depending on your machine, you may need to install additional libraries. In case for my Linux computer, I must install `libgl1-mesa-glx`, `ffmpeg`, `libsm6`, `libxext6` and some others for this going to work properly.

Trained model is not included in this repository. I encourage you to work on your own model as it provides more satisfying experience.


Next, initiate your local database. An example of DDL is given in repo directory. I use MySQL by the way. In case you have docker installed, you may run
```bash
docker run -d -p <host-port>:<container-port> --name app_db -e MYSQL_ROOT_PASSWORD=<yourRootPassword> mysql:<tag>
```

Finally, run the application and happy experimenting!
```bash
python ./app/api.py
```
Using docker, you may simply build and run docker image
```bash
docker build -t app .
docker run -d -p <host-port>:<container-port> --name <container-name> <image-name>
```
# How to train a model
![training diagram](asset/train-diagram.jpg)
## Training data preparation
- Collect images of cars with license plate for training purpose and place it under /data/images directory. It is better to have images of various brightness, orientation, perspective, scale, sharpness, etc. If you are interested, you may email me at bagusbpg[at]gmail[dot]com and I will share the dataset for training (uploading these huge image dataset to repository is consuming too much space)
- Create `PASCAL Visual Object Classes Challenge` for each image, identifying the location of license plate. In case `PASCAL VOC` is created in XML, you may run extractXML.py in data directory to extract `xmin`, `xmax`, `ymin`, and `ymax` coordinates and save them to csv file for later use in training model. Required argument to run this script is path to directory containing images and XML files. <p>NOTE: Beside XML, JSON is also valid format to store `PASCAL VOC`, you may create your own script to do the same thing as extractXML.py.
```bash
python ./data/extractXML.py ./data/images
```
## Training model
- Run train.py in train directory.
```bash
python ./train/train.py
```
- In general, what it does are (1) splitting training, validation, and test for feature-dataset, (2) augment images, i.e. rotating each image 90-degrees, 180-degrees, and 270-degrees clockwise to artificially make dataset four times bigger, (3) preprocessing each image, i.e. channel-wise standardization to make zero mean and unit standard deviation of dataset, (4) save mean and standard deviation of training dataset to be used for inference, (5) splitting training, validation, and test for target-dataset, which comes from extracted Pascal VOC XMLs described earlier, (6) define model, and finallny (7) fit the model and save it. Model fitting is equipped with EarlyStopping to prevent overfitting.
# How prediction works


# Documentation
Currently, two main endpoints are provided with one endpoint is for health-check.
## /checkin
This path only accepts `POST` requests. Requests should be `form-data` with a key `file` and value containing `image/jpeg` file. User may upload image of car to this endpoint. An example request using `curl` command is given below.
```bash
curl --location --request POST '<HOST>:<PORT>/checkin' 
--form 'file=@<PATH-TO-JPG-FILE>'
```
Response uses standard format as in the following example.
```json
{
    "code": 200,
    "message": "licence plate B-2467-UXY checked in with id 6454887f-1869-46e4-a92a-5c927f1938be"
}
```
In general, this endpoint attempts to recognize the available license plate on uploaded image, then stored it in database, effectively "checking in" a car.
## /checkout
This path expects request and gives response much like `/checkin`, but instead of "checking in" a car, it does the opposite. An example request using `curl` command is given below.
```bash
curl --location --request POST '<HOST>:<PORT>/checkout' 
--form 'file=@<PATH-TO-JPG-FILE>'
```
And corresponding response, if successfull, may looks like this.
```json
{
    "code": 200,
    "message": "licence plate ac-1616-fk-06126 checked out at Sat Dec  3 15:51:21 2022"
}
```
## /
This path is for health-check purpose only. Accepting `GET`, it requires no request. Executing this `curl` command
```bash
curl --location --request GET '<HOST>:<PORT>/'
```
it will throw following response
```json
{
    "code": 200,
    "message": "service works fine!"
}
```