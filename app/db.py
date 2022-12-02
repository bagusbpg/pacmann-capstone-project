from config import loadJSON
import datetime
from mysql.connector import connect
import time
from util import response
import uuid

def database_init(config):
    if not config:
        return None

    db = connect(
        host=config['DATABASE']['HOST'],
        user=config['DATABASE']['USERNAME'],
        password=config['DATABASE']['PASSWORD'],
        database=config['DATABASE']['NAME']
    )

    return db

def check_in(db, licenseNumber):
    query = 'INSERT INTO cars (id, license_number) VALUES (%s, %s)'
    id = str(uuid.uuid4())
    
    try:
        cursor = db.cursor()
        cursor.execute(query, (id, licenseNumber))
        db.commit()
    except:
        db.rollback()
        return None, response(500, 'failed to check in')

    return id, None
    
def fetch_all(db, checkedOut=False):
    if not checkedOut:
        query = 'SELECT id, license_number, check_in_at FROM cars WHERE check_out_at IS NULL'
    else:
        query = 'SELECT id, license_number, check_in_at FROM cars WHERE check_out_at NOT NULL'

    try:
        cursor = db.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
    except:
        return None, response(500, 'failed to fetch cars')

    return result, None

def check_out(db, id, text):
    now = time.time()
    currentTimeStamp = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
    query = 'UPDATE cars SET check_out_at = %f, prediction = %s WHERE id = %s'

    try:
        cursor = db.cursor()
        cursor.execute(query, (currentTimeStamp, text, id))
        db.commit()
    except:
        db.rollback()
        return None, response(500, 'failed to check out')

    return now, None