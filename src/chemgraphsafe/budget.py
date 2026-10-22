"""Cross-command, cross-run reservations; ambiguous charges retain their reservation."""

import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class BudgetExceeded(ValueError):
    pass


class Ledger:
    """SQLite-backed spend ledger shared across all live commands for a given project.

    Every inference attempt is atomically reserved before the HTTP call and then
    settled with the actual charge. If the actual charge exceeds the reservation,
    the run stops immediately rather than silently over-spending.

    ``budget_mode="attempts_only"`` rows count toward the attempt ceiling but not
    the dollar ceiling, supporting subscription accounts with no per-token pricing.
    """

    def __init__(self, path: Path, max_usd: float, max_attempts: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.resolve()
        self.max_usd = max_usd
        self.max_attempts = max_attempts
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts "
                "(id INTEGER PRIMARY KEY, run TEXT, reserved REAL, charged REAL, status TEXT)"
            )
            db.execute("CREATE TABLE IF NOT EXISTS limits (usd REAL, attempts INTEGER)")
            columns = {r[1] for r in db.execute("PRAGMA table_info(attempts)")}
            if "budget_mode" not in columns:
                db.execute(
                    "ALTER TABLE attempts ADD COLUMN budget_mode TEXT NOT NULL DEFAULT 'usd'"
                )
            old = db.execute("SELECT usd, attempts FROM limits").fetchone()
            if old is None:
                db.execute("INSERT INTO limits VALUES (?,?)", (max_usd, max_attempts))
            elif tuple(old) != (max_usd, max_attempts):
                raise ValueError(
                    "Budget limits differ from the established ledger; retain its limits"
                )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, run: str, amount: float, *, budget_mode: str = "usd") -> int:
        """Atomically reserve ``amount`` USD (or 0 for attempts-only) before an inference call.

        Returns the row ID used to settle the reservation. Raises ``BudgetExceeded``
        if either the attempt count or the cumulative dollar total would be exceeded.
        """
        if budget_mode not in {"usd", "attempts_only"}:
            raise ValueError("Unknown budget mode")
        if not math.isfinite(amount) or (
            (budget_mode == "usd" and not 0 < amount <= self.max_usd)
            or (budget_mode == "attempts_only" and amount != 0)
        ):
            raise BudgetExceeded("Per-attempt reservation exceeds budget or is invalid")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            n, spent = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN budget_mode='usd' "
                "THEN COALESCE(charged,reserved) ELSE 0 END),0) FROM attempts"
            ).fetchone()
            if n >= self.max_attempts or (
                budget_mode == "usd" and spent + amount > self.max_usd + 1e-12
            ):
                raise BudgetExceeded("Cumulative budget/attempt limit reached")
            cursor = db.execute(
                "INSERT INTO attempts(run,reserved,status,budget_mode) VALUES (?,?,?,?)",
                (run, amount, "reserved", budget_mode),
            )
            return cursor.lastrowid

    def settle(self, id_: int, amount: float | None, status: str):
        """Record the actual charge for a reserved attempt.

        Pass ``amount=None`` when the provider did not report a cost; the reservation
        stays in the ledger as an unreconciled charge. Raises ``BudgetExceeded`` if
        the actual charge exceeds the reservation — this surfaces mispriced configs
        immediately rather than letting the overrun accumulate silently.
        """
        if amount is not None and (not math.isfinite(amount) or amount < 0):
            raise ValueError("Invalid settlement amount")
        with self.connect() as db:
            reserved, mode = db.execute(
                "SELECT reserved,budget_mode FROM attempts WHERE id=?", (id_,)
            ).fetchone()
            if mode == "attempts_only":
                amount = None
            db.execute(
                "UPDATE attempts SET charged=?,status=? WHERE id=?",
                (amount, status, id_),
            )
        if amount is not None and amount > reserved + 1e-12:
            # The bill cannot be undone. Surface incorrect price/ceiling configuration immediately.
            raise BudgetExceeded("Provider cost exceeded its reservation; stop and correct pricing")

    def summary(self) -> dict:
        """Return a snapshot of total attempts, accounted spend, and unreconciled charges."""
        with self.connect() as db:
            n, spent, unknown, included = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN budget_mode='usd' "
                "THEN COALESCE(charged,reserved) ELSE 0 END),0), "
                "COALESCE(SUM(budget_mode='usd' AND charged IS NULL),0), "
                "COALESCE(SUM(budget_mode='attempts_only'),0) FROM attempts"
            ).fetchone()
        return {
            "path": str(self.path),
            "attempts": n,
            "attempts_only": included,
            "metered_attempts": n - included,
            "accounted_usd": spent,
            "unreconciled_attempts": unknown,
            "max_usd": self.max_usd,
            "max_attempts": self.max_attempts,
        }
