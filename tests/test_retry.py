import random
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from api_client.errors import AuthenticationError, ServerError
from api_client.retry import RetryPolicy, compute_delay, execute_with_retry, parse_retry_after

NO_JITTER = RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=8.0, jitter=False)


def test_backoff_doubles_then_caps():
    delays = [compute_delay(n, NO_JITTER) for n in range(1, 7)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]


def test_jitter_stays_within_half_to_full_of_ceiling():
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=True)
    rng = random.Random(7)
    for attempt in range(1, 6):
        ceiling = min(30.0, 2 ** (attempt - 1))
        for _ in range(200):
            d = compute_delay(attempt, policy, rng=rng)
            assert ceiling / 2 <= d <= ceiling


def test_retry_after_overrides_shorter_backoff():
    assert compute_delay(1, NO_JITTER, retry_after=5) == 5


def test_retry_after_is_capped():
    policy = RetryPolicy(base_delay=0.5, max_retry_after=10, jitter=False)
    assert compute_delay(1, policy, retry_after=9999) == 10


def test_retry_after_never_shortens_backoff():
    assert compute_delay(4, NO_JITTER, retry_after=1) == 4.0


def test_parse_retry_after_seconds_and_garbage():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("-3") is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None


def test_parse_retry_after_http_date():
    future = datetime.now(timezone.utc) + timedelta(seconds=30)
    value = parse_retry_after(format_datetime(future, usegmt=True))
    assert 25 <= value <= 31


def test_policy_validation():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_succeeds_after_transient_failures():
    calls, sleeps = [], []

    def op(attempt):
        calls.append(attempt)
        if attempt < 3:
            raise ServerError("boom")
        return "ok"

    assert execute_with_retry(op, NO_JITTER, sleep=sleeps.append) == "ok"
    assert calls == [1, 2, 3]
    assert sleeps == [0.5, 1.0]


def test_gives_up_after_max_attempts_and_reports_attempts():
    sleeps = []

    def op(attempt):
        raise ServerError("down")

    with pytest.raises(ServerError) as info:
        execute_with_retry(op, NO_JITTER, sleep=sleeps.append)
    assert info.value.attempts == 5
    assert len(sleeps) == 4  # no sleep after the final failure


def test_non_retryable_error_raises_immediately():
    calls = []

    def op(attempt):
        calls.append(attempt)
        raise AuthenticationError("nope")

    with pytest.raises(AuthenticationError):
        execute_with_retry(op, NO_JITTER, sleep=lambda s: None)
    assert calls == [1]


def test_can_retry_false_disables_retries():
    calls = []

    def op(attempt):
        calls.append(attempt)
        raise ServerError("down")

    with pytest.raises(ServerError):
        execute_with_retry(op, NO_JITTER, can_retry=False, sleep=lambda s: None)
    assert calls == [1]


def test_on_retry_callback_receives_delay():
    seen = []

    def op(attempt):
        if attempt == 1:
            raise ServerError("x")
        return 1

    execute_with_retry(op, NO_JITTER, sleep=lambda s: None,
                       on_retry=lambda a, e, d: seen.append((a, type(e).__name__, d)))
    assert seen == [(1, "ServerError", 0.5)]