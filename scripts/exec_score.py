"""Execution-based scoring of saved predictions (no GPU needed).

Re-scores a predictions JSONL from eval.py by *executing* gold and predicted
SQL against synthetic in-memory SQLite databases and comparing result sets,
instead of string-matching the SQL text.

The dataset ships schemas with no rows, and on an empty database every query
returns the same nothing — so we synthesize rows. To make wrong queries
actually produce different results, each table is populated with a mix of:
  - rows built from the literals appearing in the gold/predicted queries
    (so WHERE clauses select non-empty sets),
  - partial-match rows (so a missing/extra AND condition changes the result),
  - random filler rows (so aggregates and bare-column selects differ).
Queries are compared on several independently-randomized databases; they
must produce identical results on all of them to count as a match. This is
an approximation of semantic equivalence — more databases, fewer false
equivalences.

Usage: python scripts/exec_score.py outputs/predictions-sql-lora.jsonl
"""

import argparse
import json
import random
import re
import sqlite3
from collections import Counter

N_DATABASES = 5
N_FILLER_ROWS = 6


def extract_schema(question_text: str) -> str:
    m = re.search(
        r"Given this database schema:\n\n(.*?)\n\nWrite a SQL query",
        question_text,
        re.DOTALL,
    )
    return m.group(1) if m else ""


def parse_tables(schema_sql: str) -> dict:
    """Return {table_name: [(column, type), ...]}."""
    tables = {}
    for name, body in re.findall(
        r"CREATE TABLE\s+(\S+)\s*\((.*?)\)", schema_sql, re.DOTALL | re.IGNORECASE
    ):
        columns = []
        for coldef in body.split(","):
            parts = coldef.strip().split()
            if len(parts) >= 2:
                columns.append((parts[0], parts[1].upper()))
        tables[name] = columns
    return tables


def column_literals(sql: str) -> dict:
    """Map column -> set of literal values compared against it in the query.

    Every numeric literal K also contributes K-1, K, K+1 (boundary values),
    so generated rows straddle each threshold: some rows satisfy the
    condition and some falsify it. For inequalities this is required for
    correctness — without it a condition like `extra_points < 0` is never
    satisfied, every query returns empty, and empty == empty makes wrong
    queries look equivalent. For equalities it sharpens discrimination
    (col = 5 vs col >= 5 differ only if a row with 6 exists).
    """
    lits = {}
    for col, _op, val in re.findall(
        r"(\w+)\s*(>=|<=|!=|=|>|<|LIKE)\s*(\"[^\"]*\"|'[^']*'|-?[\d.]+)",
        sql,
        re.IGNORECASE,
    ):
        val = val.strip("\"'")
        pool = lits.setdefault(col.lower(), set())
        pool.add(val)
        try:
            n = float(val)
            n = int(n) if n == int(n) else n
            pool.update({str(n - 1), str(n), str(n + 1)})
        except ValueError:
            pass
    return lits


def is_numeric(sql_type: str) -> bool:
    return any(t in sql_type for t in ("INT", "REAL", "NUMERIC", "DECIMAL", "FLOAT"))


def make_value(col: str, sql_type: str, rng: random.Random, pool: set):
    # Small ranges so join keys and repeated categories collide across rows.
    if pool and rng.random() < 0.4:
        return rng.choice(sorted(pool))
    if is_numeric(sql_type) or col.endswith("_id") or col.endswith("_number"):
        return rng.randint(1, 6)
    return rng.choice(["alpha", "bravo", "carol", "delta", "echo"])


def build_database(tables: dict, lits: dict, rng: random.Random) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for name, columns in tables.items():
        quoted = ", ".join(f'"{c}" {t}' for c, t in columns)
        conn.execute(f'CREATE TABLE "{name}" ({quoted})')

        rows = []
        # One row satisfying every gold/pred equality literal at once.
        rows.append([
            next(iter(lits.get(c.lower(), set())), None) or make_value(c, t, rng, set())
            for c, t in columns
        ])
        # Partial-match rows: one literal kept, everything else random.
        for c_target, _ in columns:
            if c_target.lower() in lits:
                rows.append([
                    (next(iter(lits[c.lower()])) if c == c_target else make_value(c, t, rng, lits.get(c.lower(), set())))
                    for c, t in columns
                ])
        # Random filler.
        for _ in range(N_FILLER_ROWS):
            rows.append([make_value(c, t, rng, lits.get(c.lower(), set())) for c, t in columns])

        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', rows)
    return conn


def run_query(conn: sqlite3.Connection, sql: str):
    """Execute and return results as an order-insensitive multiset, or None on error."""
    try:
        cursor = conn.execute(sql)
        return Counter(
            tuple(round(v, 6) if isinstance(v, float) else str(v) for v in row)
            for row in cursor.fetchall()
        )
    except sqlite3.Error:
        return None


def is_empty(result: Counter) -> bool:
    """True for zero rows, or rows that are entirely NULL (e.g. SUM of nothing)."""
    return not result or all(all(v == "None" for v in row) for row in result)


def score_example(ex: dict, seed: int) -> str:
    """Return 'match', 'mismatch', 'gold_error', or 'non_discriminative'.

    'non_discriminative': gold returned an empty/all-NULL result on every
    database, so agreement proves nothing — any query that also returns
    nothing would "match". A mismatch on such an example is still valid
    evidence (the prediction produced rows where gold produced none).
    """
    schema = extract_schema(ex["question"])
    tables = parse_tables(schema)
    if not tables:
        return "gold_error"

    lits = column_literals(ex["gold"])
    for col, vals in column_literals(ex["prediction"]).items():
        lits.setdefault(col, set()).update(vals)

    gold_ever_nonempty = False
    for i in range(N_DATABASES):
        try:
            conn = build_database(tables, lits, random.Random(seed * 1000 + i))
        except sqlite3.Error:
            # Malformed schema (e.g. duplicate column names in the dataset).
            # If we can't build the database, we can't execute gold either.
            return "gold_error"
        gold_result = run_query(conn, ex["gold"])
        pred_result = run_query(conn, ex["prediction"])
        conn.close()
        if gold_result is None:
            return "gold_error"
        if pred_result is None or pred_result != gold_result:
            return "mismatch"
        if not is_empty(gold_result):
            gold_ever_nonempty = True
    return "match" if gold_ever_nonempty else "non_discriminative"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions", help="predictions JSONL written by eval.py")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.predictions)]
    verdicts = [score_example(ex, seed=i) for i, ex in enumerate(rows)]
    outcomes = Counter(verdicts)

    scored = outcomes["match"] + outcomes["mismatch"]
    print(f"=== {args.predictions} ===")
    print(f"gold query unexecutable (excluded): {outcomes['gold_error']}")
    print(f"non-discriminative, gold empty on all DBs (excluded): {outcomes['non_discriminative']}")
    if scored:
        print(f"execution match: {outcomes['match']}/{scored} = {outcomes['match'] / scored:.1%}")
    disagreements = sum(
        1 for ex, v in zip(rows, verdicts) if ex["match"] != (v == "match")
    )
    print(f"(string metric said {sum(ex['match'] for ex in rows)}/{len(rows)}; "
          f"{disagreements} examples judged differently)")


if __name__ == "__main__":
    main()
