# About the project
This application provides REST API for car license number recognition. Sample implementations may include car parking billing system or likewise.
# Installation
First, clone this repository,
```bash
git clone https://github.com/bagusbpg/pacmann-capstone-project.git
```
Then, set up configuration file, plate it in root directory
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
Next, initiate your local database. An example of DDL is given in repo directory. (Sorry, we do not implement object-relational-mapping, here)
Finally, run the application and happy experimenting!
```
python ./app/api.py
```
# List of endpoint
TODO