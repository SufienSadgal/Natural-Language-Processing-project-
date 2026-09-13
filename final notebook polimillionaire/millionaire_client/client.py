"""Main client class combining all modules."""

from .base import BaseClient
from .auth import AuthModule
from .game import GameModule, GameSession
from .competitions import CompetitionsModule
from .leaderboard import LeaderboardModule
from .models import User


class MillionaireClient:
    """
    Main client for interacting with the Poli-Millionaire API.
    
    This client provides a unified interface for all API operations,
    organized into logical modules (auth, game, competitions, leaderboard).
    
    Example:
        client = MillionaireClient("http://localhost:4000")
        client.login("myusername", "mypassword")
        
        # List competitions
        competitions = client.competitions.list_all()
        
        # Start a game
        game = client.game.start(competition_id=1)
        
        # Answer questions
        while game.in_progress:
            question = game.current_question
            # Your answering logic here
            result = game.answer(option_id=1)
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the Millionaire client.
        
        Args:
            base_url: The base URL of the API server (e.g., "http://localhost:4000")
            timeout: Default request timeout in seconds
        """
        self._base = BaseClient(base_url, timeout)
        
        # Initialize modules
        self._auth = AuthModule(self._base)
        self._game = GameModule(self._base)
        self._competitions = CompetitionsModule(self._base)
        self._leaderboard = LeaderboardModule(self._base)
    
    # Authentication delegation
    def login(self, username: str, password: str) -> User:
        """Log in with username and password."""
        return self._auth.login(username, password)
    
    def logout(self):
        """Log out and clear authentication."""
        self._auth.logout()
    
    @property
    def user(self) -> User:
        """Get the currently logged-in user."""
        return self._auth.get_current_user()
    
    @property
    def is_authenticated(self) -> bool:
        """Check if the client is authenticated."""
        return self._auth.is_logged_in()
    
    # Module accessors
    @property
    def auth(self) -> AuthModule:
        """Access authentication operations."""
        return self._auth
    
    @property
    def game(self) -> GameModule:
        """Access game operations."""
        return self._game
    
    @property
    def competitions(self) -> CompetitionsModule:
        """Access competition operations."""
        return self._competitions
    
    @property
    def leaderboard(self) -> LeaderboardModule:
        """Access leaderboard operations."""
        return self._leaderboard
    
    # Convenience methods for common workflows
    def play_game(self, competition_id: int, answer_strategy):
        """
        Play a complete game using a provided answer strategy function.
        
        This is a high-level convenience method that handles the game loop.
        
        Args:
            competition_id: The ID of the competition to play
            answer_strategy: A callable that receives a Question and returns
                            either an option_id (int) or option_text (str)
            
        Returns:
            The final GameSession state
            
        Example:
            def my_strategy(question):
                # Simple strategy: always pick the first option
                return question.options[0].id
            
            final_state = client.play_game(1, my_strategy)
            print(f"Earned: {final_state.earned_amount}")
        """
        game = self._game.start(competition_id)
        
        while game.in_progress:
            question = game.current_question
            if not question:
                break
            
            answer = answer_strategy(question)
            
            if isinstance(answer, int):
                result = game.answer(answer)
            else:
                result = game.answer_by_text(str(answer))
            
            if result.game_over:
                break
        
        return game