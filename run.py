"""MLOps-style batch job: rolling mean signal pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS = ("seed", "window", "version")
REQUIRED_COLUMNS = ("close",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute rolling-mean binary signals from OHLCV data."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", required=True, help="Path to metrics JSON output")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def write_metrics(output_path: str, payload: dict[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_config(config_path: str, logger: logging.Logger) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping with seed, window, and version")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {', '.join(missing)}")

    seed = config["seed"]
    window = config["window"]
    version = config["version"]

    if not isinstance(seed, int):
        raise ValueError("Config field 'seed' must be an integer")
    if not isinstance(window, int) or window < 1:
        raise ValueError("Config field 'window' must be a positive integer")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Config field 'version' must be a non-empty string")

    logger.info(
        "Config loaded and validated (seed=%s, window=%s, version=%s)",
        seed,
        window,
        version,
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError("Input file is empty")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Input CSV is empty or unreadable") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV format: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV contains no rows")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input CSV missing required column(s): {', '.join(missing_columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise ValueError("Column 'close' must be numeric")

    logger.info("Rows loaded: %s", len(df))
    return df


def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> float:
    logger.info("Computing rolling mean on close (window=%s)", window)
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()

    # First window-1 rows have NaN rolling means; exclude them from signal computation.
    valid_mask = rolling_mean.notna()
    signal = pd.Series(0, index=df.index, dtype=int)
    signal.loc[valid_mask] = (df.loc[valid_mask, "close"] > rolling_mean.loc[valid_mask]).astype(
        int
    )

    valid_signals = signal.loc[valid_mask]
    signal_rate = float(valid_signals.mean()) if len(valid_signals) else 0.0

    logger.info(
        "Signal generation complete (%s rows with valid rolling mean)",
        int(valid_mask.sum()),
    )
    return signal_rate


def run_pipeline(
    input_path: str,
    config_path: str,
    output_path: str,
    log_file: str,
) -> dict[str, Any]:
    logger = setup_logging(log_file)
    logger.info("Job started")

    config = load_config(config_path, logger)
    np.random.seed(config["seed"])

    df = load_dataset(input_path, logger)
    signal_rate = compute_signals(df, config["window"], logger)

    metrics = {
        "version": config["version"],
        "rows_processed": int(len(df)),
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "seed": config["seed"],
        "status": "success",
    }

    logger.info(
        "Metrics summary: rows_processed=%s, signal_rate=%.4f",
        metrics["rows_processed"],
        metrics["value"],
    )
    logger.info("Job finished with status=success")
    return metrics


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    version_fallback = "unknown"

    try:
        metrics = run_pipeline(
            input_path=args.input,
            config_path=args.config,
            output_path=args.output,
            log_file=args.log_file,
        )
        version_fallback = metrics["version"]
        metrics["latency_ms"] = int((time.perf_counter() - start) * 1000)
        write_metrics(args.output, metrics)
        print(json.dumps(metrics, indent=2))
        return 0
    except Exception as exc:
        error_metrics: dict[str, Any] = {
            "version": version_fallback,
            "status": "error",
            "error_message": str(exc),
        }

        try:
            with open(args.config, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if isinstance(config, dict) and isinstance(config.get("version"), str):
                error_metrics["version"] = config["version"]
        except Exception:
            pass

        write_metrics(args.output, error_metrics)

        logger = logging.getLogger("mlops_task")
        if not logger.handlers:
            setup_logging(args.log_file)
            logger = logging.getLogger("mlops_task")
        logger.exception("Job failed: %s", exc)
        logger.info("Job finished with status=error")

        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
