# data_loader.py
# -*- coding: utf-8 -*-
"""
Load business CSV data into existing PostgreSQL tables.
- Reuses DB config via postgresql_operation.get_default_postgresql_params()
- Validates columns against target table
- Supports chunked bulk insert with psycopg2.extras.execute_values
- CLI: paths (glob), --schema, --table, --truncate, --strict, --chunksize, --dry-run
Notes:
- CSV header must contain a subset (or equal set if --strict) of table columns.
- Extra table columns not present in CSV will be inserted as NULL.
"""

import os
import sys
import glob
import argparse
import pandas as pd
import psycopg2
from psycopg2 import extras, sql
from typing import List, Dict, Tuple
from postgresql_operation import get_default_postgresql_params

# ---------- DB Introspection ----------

def get_table_columns(conn, schema: str, table: str) -> List[Tuple[str, str]]:
    """
    Fetch target table columns and data types in ordinal order.
    Returns list of (column_name, data_type) in physical order.
    """
    q = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        rows = cur.fetchall()
    if not rows:
        raise ValueError(f"Target table not found: {schema}.{table}")
    # rows like [('Study_Name','text'), ...]
    return rows

# ---------- Import ----------

def build_insert_sql(schema: str, table: str, cols: List[str]) -> str:
    """
    Build INSERT ... VALUES %s template for execute_values.
    """
    fqtn = sql.Identifier(schema, table) if schema else sql.Identifier(table)
    col_idents = [sql.Identifier(c) for c in cols]
    stmt = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        fqtn,
        sql.SQL(", ").join(col_idents)
    )
    return stmt.as_string(conn=None)  # we'll pass to execute_values which doesn't need a cursor here

def coerce_df_for_pg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic normalization:
    - Keep as strings where unsure; Postgres will coerce common numerics/datetimes
    - Replace pandas NA with None in tuples creation step
    """
    # nothing heavy here; rely on server-side casts for numeric/timestamp strings
    return df

def dataframe_to_tuples(df: pd.DataFrame) -> List[tuple]:
    """
    Convert DataFrame to list of tuples and turn NaN into None (NULL).
    """
    return [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]

def load_one_file(conn, path: str, schema: str, table: str, truncate: bool,
                  strict: bool, chunksize: int, dry_run: bool) -> None:
    """
    Load a single CSV file into target table.
    """
    # Read CSV (Excel-friendly BOM)
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if df.empty:
        print(f"   [SKIP] Empty CSV: {path}")
        return

    # Trim header spaces and keep exact header names (case sensitive to match created columns)
    df.columns = [c.strip() for c in df.columns]

    # Fetch table columns (ordered)
    table_cols_types = get_table_columns(conn, schema, table)
    table_cols = [c for c, _ in table_cols_types]
    table_cols_set = set(table_cols)

    csv_cols = list(df.columns)
    csv_cols_set = set(csv_cols)

    # Validate columns
    missing_in_table = [c for c in csv_cols if c not in table_cols_set]
    if missing_in_table:
        raise ValueError(f"CSV has columns not in target table {schema}.{table}: {missing_in_table}")

    if strict:
        missing_in_csv = [c for c in table_cols if c not in csv_cols_set]
        if missing_in_csv:
            raise ValueError(f"--strict: CSV missing columns required by table {schema}.{table}: {missing_in_csv}")

    # Reorder to match table physical order; keep only columns present in CSV
    final_cols = [c for c in table_cols if c in csv_cols_set]
    df = df[final_cols].copy()

    # Optional normalization
    df = coerce_df_for_pg(df)

    # SQLs
    fqtn = f'{schema}.{table}' if schema else table
    drop_sql = f'TRUNCATE TABLE "{schema}"."{table}"' if truncate else None
    insert_sql = build_insert_sql(schema, table, final_cols)

    if dry_run:
        print(f"   -- DRY RUN: would load {len(df)} rows into {fqtn}")
        print(f"   Columns: {final_cols}")
        if truncate:
            print(f"   Would TRUNCATE TABLE {fqtn} first")
        return

    with conn.cursor() as cur:
        if truncate:
            cur.execute(drop_sql + " RESTART IDENTITY;")  # reset serials if any
        # chunked insert
        total = 0
        for i in range(0, len(df), chunksize):
            chunk = df.iloc[i:i+chunksize]
            data = dataframe_to_tuples(chunk)
            extras.execute_values(cur, insert_sql, data)
            total += len(chunk)
        conn.commit()
    print(f"   [OK] {path} -> {fqtn}, inserted rows: {len(df)}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Load CSV data into an existing PostgreSQL table.")
    ap.add_argument("paths", nargs="+", help="CSV file paths or glob patterns, e.g. ./data/*.csv")
    ap.add_argument("--schema", default="public", help="Target schema (default: public)")
    ap.add_argument("--table", required=True, help="Target table name (must already exist)")
    ap.add_argument("--truncate", action="store_true", help="TRUNCATE the table before loading")
    ap.add_argument("--strict", action="store_true", help="Require CSV columns == table columns (order ignored)")
    ap.add_argument("--chunksize", type=int, default=5000, help="Batch size for bulk insert (default: 5000)")
    ap.add_argument("--dry-run", action="store_true", help="Only validate and print actions, do not write")
    args = ap.parse_args()

    # Expand globs
    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        print("No CSV files matched. Please check your paths/globs.")
        sys.exit(1)

    # Connect
    params = get_default_postgresql_params()
    conn = psycopg2.connect(**params)
    conn.autocommit = False

    try:
        # sanity check: table exists
        get_table_columns(conn, args.schema, args.table)

        for path in files:
            print(f"\n==> Loading: {path}")
            load_one_file(conn, path, args.schema, args.table,
                          truncate=args.truncate, strict=args.strict,
                          chunksize=args.chunksize, dry_run=args.dry_run)
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()



#---------------------------------------------------------------------------------------
# run as : 
#
# python data_loader.py ./data/ec_study_data.csv --schema public --table ec_study --dry-run
# 覆盖性导入（先清空再写入）
#python data_loader.py ./data/ec_study_data.csv --schema public --table ec_study --truncate

# 追加导入（不清空）
#python data_loader.py ./data/ec_study_part*.csv --schema public --table ec_study --chunksize 10000

#-- 表结构
# SELECT column_name, data_type
# FROM information_schema.columns
# WHERE table_schema='public' AND table_name='ec_study'
# ORDER BY ordinal_position;

# -- 行数与示例
# SELECT COUNT(*) FROM public.ec_study;
# SELECT * FROM public.ec_study LIMIT 10;
