# MLOps Task 0 — Rolling Mean Signal Pipeline

A minimal MLOps-style batch job that loads OHLCV data, computes a rolling mean on `close`, generates a binary trading signal, and emits structured metrics and logs.

## Features

- **Reproducibility**: YAML config with fixed `seed` and `window`
- **Observability**: Python `logging` to file + stderr, machine-readable `metrics.json`
- **Deployment readiness**: Dockerized one-command run

## Project layout

```
.
├── run.py              # CLI entrypoint
├── config.yaml         # Job configuration
├── data.csv            # 10,000-row OHLCV dataset
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── metrics.json        # Sample successful metrics output
└── run.log             # Sample successful log output
```

## Local run

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the job

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

On success, the job:

1. Writes `metrics.json`
2. Writes `run.log`
3. Prints the final metrics JSON to stdout
4. Exits with code `0`

On failure, it still writes an error payload to `metrics.json`, logs the exception, prints the error JSON to stdout, and exits with a non-zero code.

## Docker build and run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container bundles `data.csv` and `config.yaml`, runs the pipeline with default paths, writes `metrics.json` and `run.log` inside the container, prints metrics to stdout, and exits `0` on success.

To copy outputs from a container run:

```bash
docker run --rm -v "%cd%:/out" mlops-task sh -c "python run.py --input data.csv --config config.yaml --output /out/metrics.json --log-file /out/run.log"
```

## Configuration

`config.yaml`:

```yaml
seed: 42
window: 5
version: "v1"
```

| Field     | Description                                      |
|-----------|--------------------------------------------------|
| `seed`    | NumPy random seed for reproducibility            |
| `window`  | Rolling mean window size (positive integer)      |
| `version` | Pipeline version echoed in metrics               |

## Processing logic

1. Load and validate config and input CSV (`close` column required).
2. Set `numpy.random.seed(seed)`.
3. Compute rolling mean on `close` with `min_periods=window`.
4. For rows with a valid rolling mean: `signal = 1` if `close > rolling_mean`, else `0`.
5. The first `window - 1` rows have no rolling mean and are excluded from `signal_rate`.
6. Emit metrics including `rows_processed`, `signal_rate`, and `latency_ms`.

## Example `metrics.json`

Successful run:

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 65,
  "seed": 42,
  "status": "success"
}
```

Error run:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Input file not found: missing.csv"
}
```

## Validation and error handling

The job validates:

- Missing or invalid config file / YAML structure
- Missing required config fields (`seed`, `window`, `version`)
- Missing, empty, or malformed input CSV
- Missing `close` column or non-numeric `close` values

All failures produce an error `metrics.json` and detailed log output.
