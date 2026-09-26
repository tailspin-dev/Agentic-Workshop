"""Load seed/tickets.csv and seed/customers.csv into app.db, rebuilding both tables on every run."""

import csv
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TICKET_COLUMNS = ["ticket_id", "customer_id", "created_at", "text"]
CUSTOMER_COLUMNS = ["customer_id", "name", "plan", "open_tickets"]


class SeedError(Exception):
    """The seed data is malformed; nothing was written."""


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header != columns:
            raise SeedError(f"{path.name}: header {header} does not match expected columns {columns}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise SeedError(f"{path.name}: line {line} does not have exactly the columns {columns}")
            rows.append(row)
    return rows


def load_seed(db_path: Path, seed_dir: Path) -> tuple[int, int]:
    """Replace the tickets and customers tables in db_path with the CSVs in seed_dir.

    Returns (ticket_count, customer_count). Raises SeedError on bad seed data,
    leaving any existing tables unchanged.
    """
    tickets = _read_csv(seed_dir / "tickets.csv", TICKET_COLUMNS)
    customers = _read_csv(seed_dir / "customers.csv", CUSTOMER_COLUMNS)
    for line, row in enumerate(customers, start=2):
        try:
            row["open_tickets"] = int(row["open_tickets"])
        except ValueError:
            raise SeedError(
                f"customers.csv: column open_tickets on line {line} is not an integer: {row['open_tickets']!r}"
            ) from None

    # autocommit=False makes sqlite3 open a transaction before the DDL too,
    # so a failure anywhere below rolls back the DROP as well.
    conn = sqlite3.connect(db_path, autocommit=False)
    try:
        with conn:
            conn.execute("DROP TABLE IF EXISTS tickets")
            conn.execute("DROP TABLE IF EXISTS customers")
            conn.execute(
                "CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, customer_id TEXT, created_at TEXT, text TEXT)"
            )
            conn.execute(
                "CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT, plan TEXT, open_tickets INTEGER)"
            )
            conn.executemany(
                "INSERT INTO tickets (ticket_id, customer_id, created_at, text) VALUES (?, ?, ?, ?)",
                [tuple(row[c] for c in TICKET_COLUMNS) for row in tickets],
            )
            conn.executemany(
                "INSERT INTO customers (customer_id, name, plan, open_tickets) VALUES (?, ?, ?, ?)",
                [tuple(row[c] for c in CUSTOMER_COLUMNS) for row in customers],
            )
    except sqlite3.IntegrityError as exc:
        raise SeedError(f"seed data violates a table constraint (duplicate ID?): {exc}") from None
    finally:
        conn.close()
    return len(tickets), len(customers)


def main() -> int:
    try:
        ticket_count, customer_count = load_seed(ROOT / "app.db", ROOT / "seed")
    except SeedError as exc:
        print(f"Seed load failed, nothing written: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {ticket_count} tickets and {customer_count} customers into app.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
