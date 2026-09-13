"""Base HTTP client for making requests to the Millionaire API."""

import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from .exceptions import (
    MillionaireError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    ServerError,
    RateLimitError,
)


class BaseClient:
    """Base HTTP client handling authentication and request/response logic."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the base client.
        
        Args:
            base_url: The base URL of the API (e.g., "http://localhost:4000")
            timeout: Default request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._auth_cookie: Optional[str] = None
    
    def _get_full_url(self, endpoint: str) -> str:
        """Construct full URL from endpoint."""
        return urljoin(f"{self.base_url}/", endpoint.lstrip("/"))
    
    def set_auth_cookie(self, cookie_value: str):
        """Set the authentication cookie manually."""
        self._auth_cookie = cookie_value
        self._session.cookies.set("polimillionaire_auth", cookie_value)
    
    def clear_auth(self):
        """Clear authentication."""
        self._auth_cookie = None
        self._session.cookies.clear()
    
    @property
    def is_authenticated(self) -> bool:
        """Check if client has authentication."""
        return "polimillionaire_auth" in self._session.cookies
    
    def _handle_response(self, response: requests.Response, raw: bool = False) -> Any:
        """Handle API response and raise appropriate exceptions."""
        if raw and response.status_code in (200, 201):
            return response.content  # return bytes for audio endpoints

        try:
            data = response.json() if response.text else {}
        except ValueError:
            data = {}

        if response.status_code in (200, 201, 204):
            return data
        
        message = data.get("message", data.get("error", f"HTTP {response.status_code}"))

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code, data)
        elif response.status_code == 404:
            raise NotFoundError(message, response.status_code, data)
        elif response.status_code == 400:
            raise ValidationError(message, response.status_code, data)
        elif response.status_code == 429:
            raise RateLimitError(message, response.status_code, data)
        elif response.status_code >= 500:
            raise ServerError(message, response.status_code, data)
        else:
            raise MillionaireError(message, response.status_code, data)
    
    def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        auth_required: bool = True,
        raw: bool = False,
    ) -> Any:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data (for POST/PUT/PATCH)
            params: Query parameters
            headers: Additional headers
            auth_required: Whether authentication is required
            
        Returns:
            Parsed JSON response
            
        Raises:
            AuthenticationError: If auth_required and not authenticated
            Various MillionaireError subclasses based on response status
        """
        if auth_required and not self.is_authenticated:
            raise AuthenticationError("Authentication required. Call login() first.")
        
        url = self._get_full_url(endpoint)
        request_headers = headers or {}
        
        try:
            if method.upper() == "GET":
                response = self._session.get(
                    url, params=params, headers=request_headers, timeout=self.timeout
                )
            elif method.upper() == "POST":
                response = self._session.post(
                    url, json=data, params=params, headers=request_headers, timeout=self.timeout
                )
            elif method.upper() == "PUT":
                response = self._session.put(
                    url, json=data, headers=request_headers, timeout=self.timeout
                )
            elif method.upper() == "PATCH":
                response = self._session.patch(
                    url, json=data, headers=request_headers, timeout=self.timeout
                )
            elif method.upper() == "DELETE":
                response = self._session.delete(
                    url, headers=request_headers, timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            return self._handle_response(response, raw=raw)

        except requests.Timeout:
            raise MillionaireError(f"Request to {endpoint} timed out after {self.timeout}s")
        except requests.ConnectionError as e:
            raise MillionaireError(f"Could not connect to server at {self.base_url}: {e}")
    
    def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a GET request."""
        return self.request("GET", endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a POST request."""
        return self.request("POST", endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a PUT request."""
        return self.request("PUT", endpoint, **kwargs)
    
    def patch(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a PATCH request."""
        return self.request("PATCH", endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make a DELETE request."""
        return self.request("DELETE", endpoint, **kwargs)