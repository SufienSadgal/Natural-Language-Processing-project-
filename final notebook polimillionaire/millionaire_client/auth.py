"""Authentication module for the Millionaire client."""

from .base import BaseClient
from .models import User
from .exceptions import AuthenticationError


class AuthModule:
    """Handles authentication operations."""
    
    def __init__(self, client: BaseClient):
        self._client = client
    
    def login(self, username: str, password: str) -> User:
        """
        Log in with username and password.
        
        Args:
            username: The username
            password: The password
            
        Returns:
            User object with user details
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        response = self._client.post(
            "/api/auth/login",
            data={"username": username, "password": password},
            auth_required=False
        )
        user = User.from_dict(response["user"])
        return user
    
    def logout(self):
        """Log out and clear authentication."""
        try:
            self._client.post("/api/auth/logout", auth_required=False)
        finally:
            self._client.clear_auth()
    
    def get_current_user(self) -> User:
        """
        Get information about the currently logged-in user.
        
        Returns:
            User object with user details
        """
        response = self._client.get("/api/auth/me")
        return User.from_dict(response["user"])
    
    def is_logged_in(self) -> bool:
        """Check if currently authenticated."""
        return self._client.is_authenticated