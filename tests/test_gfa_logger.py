# tests/test_gfa_logger.py
import logging
import sys
from datetime import datetime

import pytest
import kspec_gfa_controller.gfa_logger as mod


GFALogger = mod.GFALogger


def _close_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _reset_gfa_logger_state():
    names = set(GFALogger._initialized_loggers)
    names.update({"dummy.py", "default.py", "methods.py"})
    for name in names:
        _close_logger(name)
    GFALogger._initialized_loggers.clear()

    yield

    names = set(GFALogger._initialized_loggers)
    names.update({"dummy.py", "default.py", "methods.py"})
    for name in names:
        _close_logger(name)
    GFALogger._initialized_loggers.clear()


def _console_handlers(logger: logging.Logger):
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]


def _file_handlers(logger: logging.Logger):
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]


def test_creates_stream_and_file_handlers(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path))

    assert g.file_name == "dummy.py"
    assert g.logger.level == logging.DEBUG
    assert len(_console_handlers(g.logger)) == 1
    assert len(_file_handlers(g.logger)) == 1
    assert "dummy.py" in GFALogger._initialized_loggers


def test_default_log_directory_is_created_beside_module(tmp_path, monkeypatch):
    fake_module_file = tmp_path / "gfa_logger.py"
    monkeypatch.setattr(mod, "__file__", str(fake_module_file))

    g = GFALogger(file="default.py", log_dir=None)

    expected_log = tmp_path / "log" / f"gfa_{datetime.now():%Y-%m-%d}.log"
    for handler in g.logger.handlers:
        handler.flush()
    assert expected_log.exists()


def test_prevents_duplicate_handlers_on_same_logger(tmp_path):
    first = GFALogger(file="dummy.py", log_dir=str(tmp_path))
    original_handlers = list(first.logger.handlers)

    second = GFALogger(file="dummy.py", log_dir=str(tmp_path))

    assert second.logger is first.logger
    assert second.logger.handlers == original_handlers


def test_stream_level_and_stdout_are_applied(tmp_path):
    g = GFALogger(
        file="dummy.py",
        log_dir=str(tmp_path),
        stream_level=logging.WARNING,
    )

    handlers = _console_handlers(g.logger)
    assert len(handlers) == 1
    assert handlers[0].level == logging.WARNING
    assert handlers[0].stream is sys.stdout


def test_writes_messages_to_dated_log_file(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path))
    g.info("hello-gfa-logger")

    for handler in g.logger.handlers:
        handler.flush()

    log_path = tmp_path / f"gfa_{datetime.now():%Y-%m-%d}.log"
    assert log_path.exists()
    assert "hello-gfa-logger" in log_path.read_text(encoding="utf-8")


def test_all_convenience_methods_forward_to_standard_logger(tmp_path, monkeypatch):
    g = GFALogger(file="methods.py", log_dir=str(tmp_path))
    calls = []

    for method_name in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "critical",
    ):
        monkeypatch.setattr(
            g.logger,
            method_name,
            lambda message, method_name=method_name: calls.append(
                (method_name, message)
            ),
        )

    g.debug("debug-message")
    g.info("info-message")
    g.warning("warning-message")
    g.error("error-message")
    g.exception("exception-message")
    g.critical("critical-message")

    assert calls == [
        ("debug", "debug-message"),
        ("info", "info-message"),
        ("warning", "warning-message"),
        ("error", "error-message"),
        ("exception", "exception-message"),
        ("critical", "critical-message"),
    ]


def test_unknown_attributes_are_forwarded_to_underlying_logger(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path))

    assert g.name == g.logger.name
    assert g.getEffectiveLevel() == g.logger.getEffectiveLevel()
