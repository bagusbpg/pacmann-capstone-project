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
It is recommended to create new python environment first and then proceed with installing the requirement
```bash
pip install -r requirements.txt
```
Trained model is not included in this repository. I encourage you to work on your own model as it provides more satisfying experience. A simple training instruction is provided in train directory.


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
# Documentation
Currently, two main endpoints are provided
## /checkin
This path only accepts `POST` requests. Requests should be `form-data` with a key `file` and value containing `image/jpeg` file. User may upload image of car to this endpoint.

Response uses standard format as in the following example.
```json
{
    "code": 200,
    "message": "licence plate B-2467-UXY checked in with id 6454887f-1869-46e4-a92a-5c927f1938be"
}
```
In general, this endpoint attempts to recognize the available license plate on uploaded image, then stored it in database, effectively "checking in" a car.
## /checkout
This path expects request and gives response much like `/checkin`, but instead of "checking in" a car, it does the opposite.
## /
This path is for health-check purpose only. Accepting `GET`, it requires no request and throwing following response
```json
{
    "code": 200,
    "message": "service works fine!"
}
```