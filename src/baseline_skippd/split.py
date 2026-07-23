"""
Day-level train/validation split for SKIPP'D.

The split is defined **purely on whole local calendar days**, never on windows
or minutes: a day is the natural frame boundary (nights split the record into
one contiguous block per day). This guarantees no window can straddle the split
and the same day never lands in two sets.

- The official ``test`` group is preserved as-is (its days are the test set).
- The ``trainval`` group's days are split into train / validation.
- Default strategy ``chronological_day_holdout``: the latest ``val_frac`` of
  days become validation (a forward-in-time holdout inside trainval).
- Optional ``repo_day_10fold``: a seeded day-level k-fold (validation = one
  fold). NOTE: this is a generic day-level k-fold, not the exact Stanford fold
  membership (their fold file is not available here).

Every model — CNN, MLP, and the vanilla Transformer — reads the SAME manifest,
so the split is shared and reproducible (verifiable via ``config_hash``).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STRATEGIES = ("chronological_day_holdout", "repo_day_10fold")


def unique_dates(store, group):
    """Sorted unique local calendar days (``YYYY-MM-DD`` strings) of a group."""
    idx = store.timestamps_index(group)          # tz-aware local (US/Pacific)
    return sorted(set(idx.strftime("%Y-%m-%d")))


@dataclass
class SplitManifest:
    """A reproducible day-level split shared by every model."""

    train_dates: list
    validation_dates: list
    test_dates: list
    strategy: str
    seed: int
    params: dict = field(default_factory=dict)
    trainval_test_overlap: list = field(default_factory=list)
    created_at: str = ""
    config_hash: str = ""

    # ---- integrity ----------------------------------------------------------
    def _hash(self):
        payload = {
            "strategy": self.strategy, "seed": self.seed, "params": self.params,
            "train_dates": self.train_dates, "validation_dates": self.validation_dates,
            "test_dates": self.test_dates,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def validate(self):
        """Fail loudly if any day appears in more than one set, or a set repeats a day."""
        for name, dates in [("train", self.train_dates), ("validation", self.validation_dates),
                            ("test", self.test_dates)]:
            if len(dates) != len(set(dates)):
                raise ValueError(f"{name}_dates contains duplicate days")
        tr, va, te = map(set, (self.train_dates, self.validation_dates, self.test_dates))
        for a, b, an, bn in [(tr, va, "train", "validation"), (tr, te, "train", "test"),
                             (va, te, "validation", "test")]:
            if a & b:
                raise ValueError(f"{an} and {bn} share {len(a & b)} day(s): {sorted(a & b)[:3]}")
        return self

    # ---- helpers ------------------------------------------------------------
    def as_sets(self):
        return {"train": set(self.train_dates), "validation": set(self.validation_dates),
                "test": set(self.test_dates)}

    def split_of_date(self, date_str):
        """Return 'train'|'validation'|'test'|None for a ``YYYY-MM-DD`` day."""
        for name, dates in self.as_sets().items():
            if date_str in dates:
                return name
        return None

    # ---- persistence --------------------------------------------------------
    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))
        return Path(path)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def make_split(store, strategy="chronological_day_holdout", seed=42,
               val_frac=0.15, n_folds=10, fold=0):
    """
    Build a day-level SplitManifest from a store.
    Input : store; strategy; seed; val_frac (chronological) or n_folds/fold (k-fold).
    Output: a validated SplitManifest (test = the store's test-group days).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}', choose from {STRATEGIES}")

    tv_dates = unique_dates(store, "trainval")
    test_dates = unique_dates(store, "test")
    if len(tv_dates) < 2:
        raise ValueError(f"need >=2 trainval days to split, got {len(tv_dates)}")

    if strategy == "chronological_day_holdout":
        n_val = min(max(1, round(len(tv_dates) * val_frac)), len(tv_dates) - 1)
        validation = tv_dates[-n_val:]                 # latest days (ISO sorts chronologically)
        train = tv_dates[:-n_val]
        params = {"val_frac": val_frac, "n_val_days": n_val}
    else:  # repo_day_10fold
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold {fold} out of range [0, {n_folds})")
        perm = np.random.default_rng(seed).permutation(len(tv_dates))
        val_pos = set(np.array_split(perm, n_folds)[fold].tolist())
        validation = sorted(tv_dates[i] for i in val_pos)
        train = sorted(tv_dates[i] for i in range(len(tv_dates)) if i not in val_pos)
        params = {"n_folds": n_folds, "fold": fold}

    manifest = SplitManifest(
        train_dates=train, validation_dates=validation, test_dates=test_dates,
        strategy=strategy, seed=seed, params=params,
        trainval_test_overlap=sorted(set(tv_dates) & set(test_dates)),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest.config_hash = manifest._hash()
    return manifest.validate()
