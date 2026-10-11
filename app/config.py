"""Runtime constants for parsing, batching, and provider fallback."""

MODEL_NAME = "claude-haiku-5-5"
BATCH_SIZE = 25
MAX_CONCURRENCY = 5
# SPEC 7: a 429 or 5xx must switch providers immediately. SDK retries would
# sit on the same provider, so the client does not retry before fallback.
MAX_RETRIES = 0
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_ACTIVE_JOBS = 4
JOB_TTL_SECONDS = 6 * 60 * 60
