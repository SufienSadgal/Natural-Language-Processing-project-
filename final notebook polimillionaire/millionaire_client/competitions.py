"""Competitions module for managing competitions."""

from typing import List
from .base import BaseClient
from .models import Competition, CompetitionConfig


class CompetitionsModule:
    """Handles competition-related operations."""

    def __init__(self, client: BaseClient):
        self._client = client

    def list_all(self) -> List[Competition]:
        """
        Get a list of all available competitions.
        Competition IDs are sequential public IDs (0, 1, 2, ...) assigned by the server.

        Returns:
            List of Competition objects
        """
        response = self._client.get("/api/competitions")
        return [Competition.from_dict(c) for c in response.get("competitions", [])]

    def get_config(self, competition_id: int) -> CompetitionConfig:
        """
        Get detailed configuration for a competition.

        Args:
            competition_id: The public competition ID (0, 1, 2, ...)

        Returns:
            CompetitionConfig object with prize details
        """
        response = self._client.get(f"/api/competitions/{competition_id}/config")
        return CompetitionConfig.from_dict(response)

    def find_by_name(self, name: str, case_sensitive: bool = False) -> Competition:
        """
        Find a competition by name.

        Args:
            name: The competition name to search for
            case_sensitive: Whether matching is case-sensitive

        Returns:
            Competition object

        Raises:
            ValueError: If no competition with that name is found
        """
        competitions = self.list_all()
        search_name = name if case_sensitive else name.lower()

        for comp in competitions:
            comp_name = comp.name if case_sensitive else comp.name.lower()
            if comp_name == search_name:
                return comp

        available = [c.name for c in competitions]
        raise ValueError(f"Competition '{name}' not found. Available: {available}")