import csv
import importlib.util
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import load_seed as load_seed_module
from load_seed import CUSTOMER_COLUMNS, TICKET_COLUMNS, SeedError, load_seed

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "seed"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table_rows(db: Path, table: str) -> list[tuple]:
    with closing(sqlite3.connect(db)) as conn, conn:
        return conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()


def column_names(db: Path, table: str) -> list[str]:
    with closing(sqlite3.connect(db)) as conn, conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def expected_tickets() -> list[tuple]:
    return sorted(tuple(row[c] for c in TICKET_COLUMNS) for row in read_csv(SEED / "tickets.csv"))


def expected_customers() -> list[tuple]:
    rows = read_csv(SEED / "customers.csv")
    return sorted((r["customer_id"], r["name"], r["plan"], int(r["open_tickets"])) for r in rows)


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "app.db"


@pytest.fixture
def seed_copy(tmp_path) -> Path:
    dest = tmp_path / "seed"
    shutil.copytree(SEED, dest)
    return dest


def test_fresh_load(db):
    assert not db.exists()
    assert load_seed(db, SEED) == (24, 20)
    assert column_names(db, "tickets") == TICKET_COLUMNS
    assert column_names(db, "customers") == CUSTOMER_COLUMNS
    assert table_rows(db, "tickets") == expected_tickets()
    assert table_rows(db, "customers") == expected_customers()
    with closing(sqlite3.connect(db)) as conn, conn:
        types = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(customers)")}
        assert types["open_tickets"] == "INTEGER"
        assert {t for (t,) in conn.execute("SELECT DISTINCT typeof(open_tickets) FROM customers")} == {"integer"}


def test_main_prints_counts(tmp_path, monkeypatch, capsys):
    shutil.copytree(SEED, tmp_path / "seed")
    monkeypatch.setattr(load_seed_module, "ROOT", tmp_path)
    assert load_seed_module.main() == 0
    out = capsys.readouterr().out
    assert "24 tickets" in out and "20 customers" in out
    assert (tmp_path / "app.db").exists()


def test_rerun_is_identical(db):
    load_seed(db, SEED)
    first = (table_rows(db, "tickets"), table_rows(db, "customers"))
    assert load_seed(db, SEED) == (24, 20)
    assert (table_rows(db, "tickets"), table_rows(db, "customers")) == first


def test_stale_data_is_replaced(db):
    load_seed(db, SEED)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("INSERT INTO tickets VALUES ('T-9999', 'C-77', '2026-01-01T00:00:00', 'stale')")
        conn.execute("UPDATE customers SET plan = 'Edited', open_tickets = 99 WHERE customer_id = 'C-77'")
        conn.execute("DELETE FROM tickets WHERE ticket_id = 'T-1042'")
    load_seed(db, SEED)
    assert table_rows(db, "tickets") == expected_tickets()
    assert table_rows(db, "customers") == expected_customers()


def test_other_tables_untouched(db):
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE notes (id INTEGER, body TEXT)")
        conn.execute("INSERT INTO notes VALUES (1, 'keep me')")
    load_seed(db, SEED)
    assert table_rows(db, "notes") == [(1, "keep me")]


def test_text_with_commas_and_quotes_is_verbatim(db, seed_copy):
    tricky = 'She said "it\'s broken", then left,  with  spaces '
    with (seed_copy / "tickets.csv").open("a", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerow(["T-9000", "C-77", "2026-09-02T00:00:00", tricky])
    load_seed(db, seed_copy)
    with closing(sqlite3.connect(db)) as conn, conn:
        (text,) = conn.execute("SELECT text FROM tickets WHERE ticket_id = 'T-9000'").fetchone()
    assert text == tricky
    # Every real seed value is stored exactly as the CSV has it.
    load_seed(db, SEED)
    assert table_rows(db, "tickets") == expected_tickets()


def _rewrite(path: Path, old: str, new: str, count: int = 1) -> None:
    content = path.read_text(encoding="utf-8")
    assert old in content
    path.write_text(content.replace(old, new, count), encoding="utf-8")


@pytest.mark.parametrize(
    ("filename", "old", "new", "named"),
    [
        ("tickets.csv", "ticket_id,customer_id,created_at,text", "ticket_id,customer_id,text,created_at", "created_at"),
        ("customers.csv", "customer_id,name,plan,open_tickets", "customer_id,name,plan", "open_tickets"),
        ("customers.csv", "Hooli,Enterprise,4", "Hooli,Enterprise,four", "open_tickets"),
        ("customers.csv", "Hooli,Enterprise,4", "Hooli,Enterprise,", "open_tickets"),
        (
            "tickets.csv",
            "T-1042,C-77,2026-09-01T09:14:00,I was charged twice this month and nobody answers.",
            "T-1042,C-77,2026-09-01T09:14:00",
            "tickets.csv",
        ),
        ("customers.csv", "Hooli,Enterprise,4", "Hooli,Enterprise,4,extra", "customers.csv"),
    ],
    ids=[
        "tickets-header",
        "customers-header",
        "open-tickets-word",
        "open-tickets-empty",
        "tickets-missing-field",
        "customers-extra-field",
    ],
)
def test_bad_seed_writes_nothing(db, seed_copy, filename, old, new, named):
    load_seed(db, SEED)
    before = (table_rows(db, "tickets"), table_rows(db, "customers"))
    _rewrite(seed_copy / filename, old, new)
    with pytest.raises(SeedError) as exc:
        load_seed(db, seed_copy)
    assert filename in str(exc.value)
    assert named in str(exc.value)
    assert (table_rows(db, "tickets"), table_rows(db, "customers")) == before


def test_bad_seed_on_fresh_db_creates_no_tables(db, seed_copy):
    _rewrite(seed_copy / "customers.csv", "Hooli,Enterprise,4", "Hooli,Enterprise,x")
    with pytest.raises(SeedError):
        load_seed(db, seed_copy)
    assert not db.exists()


def test_duplicate_id_rolls_back(db, seed_copy):
    load_seed(db, SEED)
    before = (table_rows(db, "tickets"), table_rows(db, "customers"))
    with (seed_copy / "tickets.csv").open("a", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerow(["T-1042", "C-77", "2026-09-02T00:00:00", "dup"])
    with pytest.raises(SeedError, match="UNIQUE constraint failed"):
        load_seed(db, seed_copy)
    assert (table_rows(db, "tickets"), table_rows(db, "customers")) == before


def test_root_is_repo_root():
    assert load_seed_module.ROOT == Path(__file__).resolve().parent.parent


def test_seed_with_utf8_bom_loads(db, seed_copy):
    path = seed_copy / "tickets.csv"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    assert load_seed(db, seed_copy) == (24, 20)
    assert table_rows(db, "tickets") == expected_tickets()


def test_main_exits_nonzero_on_bad_seed(tmp_path, monkeypatch, capsys):
    shutil.copytree(SEED, tmp_path / "seed")
    _rewrite(tmp_path / "seed" / "customers.csv", "Hooli,Enterprise,4", "Hooli,Enterprise,x")
    monkeypatch.setattr(load_seed_module, "ROOT", tmp_path)
    assert load_seed_module.main() != 0
    err = capsys.readouterr().err
    assert "customers.csv" in err and "open_tickets" in err


def test_mcp_server_reads_loaded_db(db, monkeypatch):
    load_seed(db, SEED)
    spec = importlib.util.spec_from_file_location("triage_server", ROOT / "mcp" / "triage_server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    monkeypatch.setattr(server, "DB_PATH", db)

    ticket = server.get_ticket("T-1042")
    assert ticket["ticket_id"] == "T-1042"
    assert ticket["customer_id"] == "C-77"

    customer = server.get_customer_history("C-77")
    assert customer["customer_id"] == "C-77"
    assert "T-1042" in customer["ticket_ids"]
    assert type(customer["open_tickets"]) is int
