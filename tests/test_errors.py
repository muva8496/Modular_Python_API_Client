import httpx
import pytest

from api_client import errors as E


@pytest.mark.parametrize("status,cls,retryable", [
    (400, E.BadRequestError, False),
    (422, E.BadRequestError, False),
    (401, E.AuthenticationError, False),
    (403, E.AuthenticationError, False),
    (404, E.NotFoundError, False),
    (418, E.ClientError, False),
    (408, E.NetworkError, True),
    (429, E.RateLimitError, True),
    (500, E.ServerError, True),
    (502, E.ServerError, True),
    (503, E.ServerError, True),
    (504, E.ServerError, True),
])
def test_status_mapping(status, cls, retryable):
    assert E.error_class_for_status(status) is cls
    assert cls.retryable is retryable


def test_everything_is_an_apierror():
    for cls in (E.ClientError, E.RateLimitError, E.ServerError, E.NetworkError,
                E.ResponseValidationError, E.NotFoundError):
        assert issubclass(cls, E.APIError)


def test_error_from_response_carries_context_and_truncates_body():
    request = httpx.Request("GET", "https://example.test/x")
    response = httpx.Response(503, text="x" * 5000, request=request)
    err = E.error_from_response(response, retry_after=4.0, attempts=2)
    assert isinstance(err, E.ServerError)
    assert err.status_code == 503 and err.retry_after == 4.0 and err.attempts == 2
    assert len(err.body) == 300
    assert "status=503" in str(err) and "attempts=2" in str(err)