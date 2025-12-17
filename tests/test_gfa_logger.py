# tests/test_gfa_logger.py
import logging
import os
from datetime import datetime

import pytest

from gfa_logger import GFALogger


@pytest.fixture(autouse=True)
def _reset_gfa_logger_state():
    """
    GFALogger는 전역 상태(_initialized_loggers)와 logging.getLogger(...) 전역 로거를 쓰므로
    테스트 간 간섭을 막기 위해 매 테스트마다 초기화한다.
    """
    # 테스트에서 사용할 "가짜 파일명" 로거를 미리 정리
    # (각 테스트에서 file 인자를 고정값으로 넣을 것이므로)
    test_logger_name = "dummy.py"
    logger = logging.getLogger(test_logger_name)

    # 핸들러 제거
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # 전역 세트 초기화
    GFALogger._initialized_loggers.discard(test_logger_name)

    yield

    # 종료 시에도 한번 더 정리(혹시 테스트 중 추가된 핸들러 남아있으면)
    logger = logging.getLogger(test_logger_name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    GFALogger._initialized_loggers.discard(test_logger_name)


def test_creates_stream_and_file_handlers(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path))

    # StreamHandler 1개, FileHandler 1개가 붙었는지 확인
    stream_handlers = [h for h in g.logger.handlers if isinstance(h, logging.StreamHandler)]
    file_handlers = [h for h in g.logger.handlers if isinstance(h, logging.FileHandler)]

    # FileHandler는 StreamHandler의 subclass가 아니므로 안전하게 따로 체크 가능
    assert len(file_handlers) == 1
    assert len(stream_handlers) >= 1  # FileHandler는 StreamHandler가 아니라서 보통 1개가 됨


def test_prevents_duplicate_handlers_on_same_logger(tmp_path):
    g1 = GFALogger(file="dummy.py", log_dir=str(tmp_path))
    handler_count_1 = len(g1.logger.handlers)

    # 같은 file_name으로 다시 생성하면 early return → 핸들러가 늘면 안됨
    g2 = GFALogger(file="dummy.py", log_dir=str(tmp_path))
    handler_count_2 = len(g2.logger.handlers)

    assert handler_count_2 == handler_count_1


def test_stream_level_is_applied(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path), stream_level=logging.WARNING)

    stream_handlers = [h for h in g.logger.handlers if isinstance(h, logging.StreamHandler)]
    # 보통 stream handler 1개
    assert len(stream_handlers) >= 1
    assert stream_handlers[0].level == logging.WARNING


def test_writes_log_to_file(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path))
    msg = "hello-gfa-logger"

    g.info(msg)

    # 생성되는 파일명 규칙: gfa_YYYY-MM-DD.log
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = tmp_path / f"gfa_{today}.log"

    assert log_path.exists()

    content = log_path.read_text(encoding="utf-8")
    assert msg in content


def test_log_methods_do_not_raise(tmp_path):
    g = GFALogger(file="dummy.py", log_dir=str(tmp_path))

    # 단순 호출 시 예외 없어야 함
    g.debug("d")
    g.info("i")
    g.warning("w")
    g.error("e")
