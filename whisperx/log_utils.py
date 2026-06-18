import json
import logging
import sys
from typing import Optional, Union

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Stage identifiers, used to keep event/log output consistent across steps.
STAGE_TRANSCRIPTION = "transcription"
STAGE_ALIGNMENT = "alignment"
STAGE_DIARIZATION = "diarization"

# When enabled, machine-readable JSON events are emitted on stdout instead of
# human-readable log messages. Toggled via ``setup_logging(structured_output=...)``.
_structured_output = False


def setup_logging(
    level: str = "info",
    log_file: Optional[str] = None,
    structured_output: bool = False,
) -> None:
    """
    Configure logging for WhisperX.

    Args:
        level: Logging level (debug, info, warning, error, critical). Default: info
        log_file: Optional path to log file. If None, logs only to console.
        structured_output: If True, emit machine-readable JSON events on stdout and
            route human-readable log messages to stderr, so stdout stays parseable.
    """
    global _structured_output
    _structured_output = structured_output

    logger = logging.getLogger("whisperx")

    logger.handlers.clear()

    try:
        log_level = getattr(logging, level.upper())
    except AttributeError:
        log_level = logging.WARNING
    logger.setLevel(log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # In structured mode keep stdout reserved for JSON events only.
    console_stream = sys.stderr if structured_output else sys.stdout
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (OSError) as e:
            logger.warning(f"Failed to create log file '{log_file}': {e}")
            logger.warning("Continuing with console logging only")

    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance configured with WhisperX settings
    """
    whisperx_logger = logging.getLogger("whisperx")
    if not whisperx_logger.handlers:
        setup_logging()

    logger_name = "whisperx" if name == "__main__" else name
    return logging.getLogger(logger_name)


def is_structured_output() -> bool:
    """Return True if structured (JSON) output is enabled."""
    return _structured_output


def _clean_percent(percent: float) -> Union[int, float]:
    """Round a progress percentage, returning an int when it is a whole number."""
    value = round(float(percent), 2)
    return int(value) if value == int(value) else value


def _emit(
    event: str,
    stage: str,
    human_message: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **extra,
) -> None:
    """
    Emit a single event.

    In structured mode a JSON object is printed to stdout. Otherwise a
    human-readable message is logged using the given (or default) logger.
    """
    if _structured_output:
        payload = {"event": event, "stage": stage}
        payload.update(extra)
        print(json.dumps(payload), flush=True)
    else:
        (logger or get_logger("whisperx")).log(level, human_message)


def log_stage_started(stage: str, logger: Optional[logging.Logger] = None) -> None:
    """Signal that a processing stage has started."""
    _emit("stage_started", stage, f"Starting {stage}...", logger)


def log_model_loading(stage: str, model: Optional[str] = None, logger: Optional[logging.Logger] = None, ) -> None:
    """Signal that a stage's model is being loaded."""
    extra = {}
    human_message = f"Loading {stage} model..."
    if model:
        extra["model"] = model
        human_message = f"Loading {stage} model: {model}"
    _emit("model_loading", stage, human_message, logger, **extra)


def log_model_loaded(stage: str, logger: Optional[logging.Logger] = None) -> None:
    """Signal that a stage's model has finished loading."""
    _emit("model_loaded", stage, f"Loaded {stage} model", logger)


def log_progress(stage: str, percent: float, logger: Optional[logging.Logger] = None, ) -> None:
    """Report progress (0-100) for a stage."""
    pct = _clean_percent(percent)
    _emit("progress", stage, f"{stage.capitalize()} progress: {pct}%", logger, percent=pct)


def log_stage_completed(stage: str, logger: Optional[logging.Logger] = None) -> None:
    """Signal that a processing stage has completed."""
    _emit("stage_completed", stage, f"Completed {stage}", logger)


def log_error(stage: str, message: str, logger: Optional[logging.Logger] = None, ) -> None:
    """Report an error that occurred during a stage."""
    _emit(
        "error",
        stage,
        f"Error during {stage}: {message}",
        logger,
        level=logging.ERROR,
        message=str(message),
    )
