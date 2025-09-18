import psycopg2
from psycopg2 import sql
from typing import Dict
from postgresql_operation import get_default_postgresql_params

def create_postgres_table(
    table_name: str,
    columns: Dict[str, str],
    db_params: Dict = None,
    if_not_exists: bool = True,
    schema: str = "public"
) -> None:
    """
    Create PostgreSQL table dynamically with specified columns and data types[9,12](@ref)
    
    Parameters:
    - table_name (str): Table name (supports schema, e.g., public.users)
    - columns (Dict[str, str]): Dictionary mapping column names to PostgreSQL data types
                Example: {'emp_id': 'SERIAL PRIMARY KEY', 'name': 'VARCHAR(100)'}
    - db_params (Dict): Database connection parameters
    - if_not_exists (bool): Add IF NOT EXISTS clause to prevent errors on existing tables[4](@ref)
    """
    # Validate input parameters
    if not table_name or not columns:
        raise ValueError("Table name and columns must be provided")
        
    # (optional) normalize to lowercase table name 
    table_name_lc = table_name.lower()

    # Build secure SQL statement using psycopg2.sql module[1,3](@ref)
    column_defs = [
        sql.SQL("{} {}").format(
            sql.Identifier(col_name),
            sql.SQL(data_type)
        ) for col_name, data_type in columns.items()
    ]

    qualified_table = sql.Identifier(schema, table_name_lc)
    
    # Construct CREATE TABLE query with parameterized components[1,6](@ref)
    query = sql.SQL("CREATE TABLE {}{} ({});").format(
        sql.SQL("IF NOT EXISTS ") if if_not_exists else sql.SQL(""),
        #sql.Identifier(table_name),
        qualified_table,
        sql.SQL(", ").join(column_defs)
    )

    # Set default connection parameters
    default_params = get_default_postgresql_params() 
    
    merged_params = {**default_params, **(db_params or {})}

    conn = None
    try:
        # Establish database connection[8,12](@ref)
        conn = psycopg2.connect(**merged_params)
        conn.autocommit = False  # Enable transaction control
        
        # Execute query using context manager for cursor[6,13](@ref)
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
            print(f"Table {table_name} created successfully!")
            
    except psycopg2.Error as e:
        print(f"Table creation failed: {str(e)}")
        if conn:
            conn.rollback()  # Transaction rollback on error[11](@ref)
    finally:
        # Ensure proper resource cleanup[6,9](@ref)
        if conn:
            conn.close()

def create_table_CM():
    table_name="CM"
    columns={
        "STUDYID":"VARCHAR(100)",
        "SUBJECTID":"VARCHAR(100)",
        "SUBJECT":"VARCHAR(100)",
        "SITEID":"VARCHAR(100)",
        "STUDYENVSITENUM":"VARCHAR(100)",
        "VISITID":"VARCHAR(100)",
        "INSTANCENAME":"VARCHAR(100)",
        "INSTREPNUM":"VARCHAR(100)",
        "DATAPAGENUM":"VARCHAR(100)",
        "DATAPAGEID":"VARCHAR(100)",
        "RECORDID":"VARCHAR(100)",
        "LOGID":"VARCHAR(100)",
        "PROJECTID":"VARCHAR(200)",
        "FOLDERNAME":"VARCHAR(100)",
        "RECORDDATE":"VARCHAR(100)",
        "RECORDPOSITION":"VARCHAR(100)",
        "TRANS_TYPE":"VARCHAR(100)",
        "CMDOSFRQ1_STD":"VARCHAR(400)",
        "CMDOSFRQ1":"VARCHAR(400)",
        "CMDSFRQO1":"VARCHAR(4000)",
        "CMROUTE1_STD":"VARCHAR(400)",
        "CMROUTEO1":"VARCHAR(4000)",
        "update_time":"TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
        "update_flag":"VARCHAR(10)"
    }
    create_postgres_table(table_name, columns)


def create_table_queryrepeatkey_hash():
    table_name="QUERY_REPEAT_KEY_test"
    columns={
        "check_id":"VARCHAR(100)",
        "STUDYID":"VARCHAR(100)",
        "SUBJECTID":"VARCHAR(100)",
        "SUBJECT":"VARCHAR(100)",
        "SITEID":"VARCHAR(100)",
        "STUDYENVSITENUM":"VARCHAR(100)",
        "VISITID":"VARCHAR(100)",
        "INSTANCENAME":"VARCHAR(100)",
        "INSTREPNUM":"VARCHAR(100)",
        "DATAPAGENUM":"VARCHAR(100)",
        "DATAPAGEID":"VARCHAR(100)",
        "RECORDID":"VARCHAR(100)",
        "LOGID":"VARCHAR(100)",
        "RECORDPOSITION":"VARCHAR(100)",
        "QUERYREPEATKEY":"VARCHAR(300)",
        "update_time":"TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP",
    }
    create_postgres_table(table_name, columns)

if __name__ == "__main__":
    # create_table_CM()
    create_table_queryrepeatkey_hash()