"""Leaderboard module for viewing rankings."""

from typing import List, Optional
from .base import BaseClient
from .models import Leaderboard, LeaderboardEntry


class LeaderboardModule:
    """Handles leaderboard operations."""
    
    def __init__(self, client: BaseClient):
        self._client = client
    
    def get(self, competition_id: int, limit: int = 10) -> Leaderboard:
        """
        Get the leaderboard for a competition.
        
        Args:
            competition_id: The competition ID
            limit: Maximum number of entries (1-100, default 10)
            
        Returns:
            Leaderboard object with entries
        """
        response = self._client.get(
            f"/api/leaderboard/{competition_id}",
            params={"limit": min(max(limit, 1), 100)}
        )
        return Leaderboard.from_dict(response)
    
    def get_top(self, competition_id: int, n: int = 10) -> List[LeaderboardEntry]:
        """
        Get the top N entries from a leaderboard.
        
        Args:
            competition_id: The competition ID
            n: Number of top entries to get
            
        Returns:
            List of LeaderboardEntry objects
        """
        leaderboard = self.get(competition_id, limit=n)
        return leaderboard.entries[:n]
    
    def find_player(self, competition_id: int, username: str) -> Optional[LeaderboardEntry]:
        """
        Find a specific player's entry on the leaderboard.
        
        Args:
            competition_id: The competition ID
            username: The username to search for
            
        Returns:
            LeaderboardEntry if found, None otherwise
        """
        # Get a larger leaderboard to search through
        leaderboard = self.get(competition_id, limit=100)
        
        for entry in leaderboard.entries:
            if entry.username.lower() == username.lower():
                return entry
        
        return None