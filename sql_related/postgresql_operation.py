import os
import psycopg2
import configparser

config = configparser.ConfigParser()
config.read('config/db.ini', encoding='utf-8')  

db_host = config.get('PostgreSQL', 'host') 
db_port = config.getint('PostgreSQL', 'port') 
db_dbname = config.get('PostgreSQL', 'dbname') 
db_user= config.get('PostgreSQL', 'user') 
db_password= config.get('PostgreSQL', 'password') 

# conn = psycopg2.connect(
#     host= db_host,
#     port= db_port,
#     user= db_user,
#     password=db_password,
#     database=db_dbname
# )


# cur=conn.cursor() 
# cur.execute(f"SELECT * FROM users ")  

def get_default_postgresql_params():
    """
    Get default PostgreSQL connection parameters.
    
    Returns:
    - dict: Default connection parameters
    """
    default_params = {
        'host': db_host,
        'dbname': db_dbname,
        'user': db_user,
        'password': db_password,
        'port': db_port
    }
    return default_params

