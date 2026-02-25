"""
core/loggingSetup.py
--------------------
Centralized logging configuration for the WealthAI backend server.

Sets up:
  - Console handler (INFO level, timestamped format)
  - Rotating file handler → logs/server.log (10 MB, 5 backups)
  - Silences noisy third-party libraries at ERROR level
"""

import logging
import logging.handlers
import os


def setupLogging(
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file: str = "server.log",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure the root logger with a console + rotating-file handler.
    Safe to call multiple times — re-configuring is a no-op if handlers
    are already attached.
    """
    root = logging.getLogger()

    # Don't add handlers twice (e.g. if called by reloader subprocess)
    if root.handlers:
        return

    root.setLevel(log_level)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ───────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except Exception as exc:
        # If the log directory can't be created, just log to console
        root.warning(f"Could not create log file handler: {exc}")

    # ── Suppress noisy third-party loggers ────────────────────────
    for noisy_lib in (
        "uvicorn", "uvicorn.access",
        "sqlalchemy", "sqlalchemy.engine", "sqlalchemy.pool", "sqlalchemy.dialects",
        "fastapi", "httpx",
    ):
        logging.getLogger(noisy_lib).setLevel(logging.ERROR)
