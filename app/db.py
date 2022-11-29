from config import loadJSON
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

config = loadJSON()
userName = config["DATABASE"]["USERNAME"]
password = config["DATABASE"]["PASSWORD"]
host = config["DATABASE"]["HOST"]
dbName = config["DATABASE"]["NAME"]

DATABASE_URL = f'mysql+mysqldb://{userName}:{password}@{host}/{dbName}'

db_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()