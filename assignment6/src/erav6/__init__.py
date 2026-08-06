"""
ERA V5 — Session 6: Training Data Execution System.

Turns the Session-5 mixture plan into an executed, ledgered, replayable data
stream:

    documents -> tokenized shards -> manifests -> mixture schedule -> packing
    -> batches -> training -> consumption ledger -> learning ledger
    -> checkpoint -> crash -> resume -> replay -> audit

Every module here is imported by `run_demo.py`, which produces
`submission_artifacts/` in one command.
"""

__version__ = "1.0.0"

# Bumped whenever anything that changes the *bytes of a batch* changes.
# Recorded in every consumption-ledger event so a replay can refuse to compare
# batches that were built by a different loader.
DATALOADER_VERSION = "erav6-loader-1.0.0"
