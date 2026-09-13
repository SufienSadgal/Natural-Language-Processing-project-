from typing import Optional


class MillionaireError(Exception):
    """Base exception for all Millionaire client errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}

    def __str__(self):
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(MillionaireError):
    """Raised when authentication fails or user is not authenticated."""
    pass


class GameError(MillionaireError):
    """Raised when a game operation fails."""
    pass


class TimeoutError(MillionaireError):
    """Raised when a question times out."""
    pass


class ValidationError(MillionaireError):
    """Raised when request validation fails."""
    pass


class NotFoundError(MillionaireError):
    """Raised when a requested resource is not found."""
    pass


class ServerError(MillionaireError):
    """Raised when the server returns a 5xx error."""
    pass


class RateLimitError(MillionaireError):
    """Raised when rate limit is exceeded (HTTP 429)."""
    pass