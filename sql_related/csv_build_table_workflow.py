# csv_workflow.py
# -*- coding: utf-8 -*-
"""
Create PostgreSQL tables from CSV blueprints (col_name, data_type, comment).
Also supports multiple files / glob patterns, schema selection, dry-run.
Comments are in English per user's preference.
"""

import os
import sys
import glob
import argparse
import pandas as pd
import psycopg2
from psycopg2 import sql
from typing import Optional
from postgresql_operation import get_default_postgresql_params

# ---------- Type mapping & helpers ----------

# canonical map for simple keywords -> PostgreSQL types
TYPE_MAP = {
    "string": "TEXT",
    "text": "TEXT",
    "int": "INTEGER",
    "integer": "INTEGER",
    "double": "DOUBLE PRECISION",
    "float": "DOUBLE PRECISION",
    "timestamp": "TIMESTAMP",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "bool": "BOOLEAN",
    "boolean": "BOOLEAN",
}

BLUEPRINT_COLS = {"col_name", "data_type", "comment"}  # minimal blueprint columns (case-insensitive)

def normalize_table_name(path: str) -> str:
    """Return lower-case table name from file path without extension."""
    return os.path.splitext(os.path.basename(path))[0].lower()

def is_blueprint_dataframe(df: pd.DataFrame) -> bool:
    """Detect if DataFrame looks like a blueprint (has col_name & data_type columns)."""
    cols = {c.strip().lower() for c in df.columns}
    return {"col_name", "data_type"}.issubset(cols)

def map_datatype(dt: str) -> str:
    """
    Map a blueprint data_type token to PostgreSQL type.
    If looks like a raw SQL type (contains '(' or ')' or space like 'varchar(100)'),
    return as-is.
    """
    if not isinstance(dt, str):
        return "TEXT"
    token = dt.strip()
    low = token.lower()
    # pass-through if it seems already a concrete SQL type (e.g., VARCHAR(100), DECIMAL(10,2))
    if "(" in token or ")" in token or " " in token:
        return token
    return TYPE_MAP.get(low, "TEXT")

def quote_ident(schema: Optional[str], table: str) -> sql.Composed:
    """Return a qualified identifier with schema.table quoting."""
    return sql.Identifier(schema, table) if schema else sql.Identifier(table)

def ensure_unique_columns(col_names: list[str]):
    """Raise error if duplicated column names exist (case-insensitive)."""
    seen = set()
    dups = set()
    for c in col_names:
        key = c.strip().lower()
        if key in seen:
            dups.add(c)
        seen.add(key)
    if dups:
        raise ValueError(f"Duplicated column names in blueprint: {sorted(dups)}")

# ---------- Build SQL ----------

def build_create_table_stmt(schema: str, table: str, rows: list[dict]) -> sql.SQL:
    """
    Build CREATE TABLE statement from blueprint rows.
    Each row needs: col_name, data_type, (optional) comment.
    """
    ensure_unique_columns([r["col_name"] for r in rows])

    col_defs = []
    for r in rows:
        col = r["col_name"]
        pg_type = map_datatype(r["data_type"])
        col_defs.append(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(pg_type))
        )

    qualified = quote_ident(schema, table)
    return sql.SQL("CREATE TABLE {} ({});").format(
        qualified,
        sql.SQL(", ").join(col_defs)
    )

def build_drop_table_stmt(schema: str, table: str) -> sql.SQL:
    qualified = quote_ident(schema, table)
    return sql.SQL("DROP TABLE IF EXISTS {};").format(qualified)

def build_comments_stmts(schema: str, table: str, rows: list[dict]) -> list[sql.SQL]:
    """
    Build COMMENT ON COLUMN statements for non-empty comments in blueprint.
    """
    stmts = []
    for r in rows:
        comment = (r.get("comment") or "").strip()
        if comment:
            stmts.append(
                sql.SQL("COMMENT ON COLUMN {}.{} IS %s;").format(
                    quote_ident(schema, table),
                    sql.Identifier(r["col_name"])
                )
            )
    return stmts

# ---------- I/O ----------

def read_blueprint_rows(path: str) -> list[dict]:
    """
    Read CSV blueprint and return a list of {col_name, data_type, comment}.
    Case-insensitive on column headers; empty comments allowed.
    """
    # utf-8-sig handles BOM from Excel-exported CSV
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Blueprint is empty: {path}")

    # normalize header names
    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)

    required = {"col_name", "data_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"Not a valid blueprint (need col_name, data_type): {path}")

    # ensure comment column exists and keep NaN as empty string (avoid 'nan')
    if "comment" not in df.columns:
        df["comment"] = ""
    else:
        # DO NOT cast to str before fillna; do fillna("") first to avoid 'nan'
        df["comment"] = df["comment"].fillna("")

    # drop fully empty col_name rows if any
    df = df[~df["col_name"].isna()]
    df["col_name"] = df["col_name"].astype(str).str.strip()
    df["data_type"] = df["data_type"].astype(str).str.strip()
    # keep comment as plain text (no 'nan')
    df["comment"] = df["comment"].astype(str)

    rows = df[["col_name", "data_type", "comment"]].to_dict(orient="records")
    if not rows:
        raise ValueError(f"No effective rows in blueprint: {path}")
    return rows


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Create PostgreSQL tables from CSV blueprints.")
    parser.add_argument("paths", nargs="+", help="CSV file paths or glob patterns, e.g. ./raw_csvs/*.csv")
    parser.add_argument("--schema", default="public", help="Target schema (default: public)")
    parser.add_argument("--table-name", default=None, help="Override table name for single-file input")
    parser.add_argument("--dry-run", action="store_true", help="Only print SQL, do not execute")
    parser.add_argument("--no-drop", action="store_true", help="Do not drop table before create")
    args = parser.parse_args()

    # Expand globs
    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        print("No CSV files matched. Please check your paths/globs.")
        sys.exit(1)

    # Connect using shared config
    params = get_default_postgresql_params()
    conn = psycopg2.connect(**params)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            for path in files:
                # resolve target table name
                table = (args.table_name or normalize_table_name(path)).lower()

                print(f"\n==> Blueprint: {path} -> {args.schema}.{table}")

                # read CSV and decide mode
                df_head = pd.read_csv(path, dtype=str, nrows=1, encoding="utf-8-sig")
                if not is_blueprint_dataframe(pd.read_csv(path, dtype=str, nrows=0, encoding="utf-8-sig").rename(columns=lambda c: c.strip().lower())):
                    raise ValueError(f"{path} is not a blueprint (needs col_name, data_type).")

                rows = read_blueprint_rows(path)

                # Build SQLs
                stmts = []
                if not args.no_drop:
                    stmts.append(build_drop_table_stmt(args.schema, table))
                stmts.append(build_create_table_stmt(args.schema, table, rows))
                comment_stmts = build_comments_stmts(args.schema, table, rows)

                if args.dry_run:
                    # Pretty print SQL with literal values
                    print("   -- SQL Preview --")
                    for s in stmts:
                        print(cur.mogrify(s.as_string(cur)).decode("utf-8"))
                    for s, r in zip(comment_stmts, rows):
                        if (r.get("comment") or "").strip():
                            print(cur.mogrify(s.as_string(cur), (r["comment"],)).decode("utf-8"))
                    continue

                # Execute DDL
                for s in stmts:
                    cur.execute(s)
                # Execute comments with parameters
                for s, r in zip(comment_stmts, rows):
                    cmt = (r.get("comment") or "").strip()
                    if cmt:
                        cur.execute(s, (cmt,))

                conn.commit()
                print(f"   Created table {args.schema}.{table} with {len(rows)} columns.")
    except Exception as e:
        conn.rollback()
        print(f"Error occurred: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()


# run as: 
# python csv_build_table_workflow.py ./raw_csvs/ec_fields.csv --schema public
# python csv_build_table_workflow.py ./raw_csvs/*.csv --schema public
# python csv_build_table_workflow.py ./raw_csvs/ec_fields.csv --schema public --dry-run

