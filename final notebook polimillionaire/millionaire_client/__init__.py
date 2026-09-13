"""
Poli-Millionaire Python Client

A modular Python client for interacting with the Poli-Millionaire quiz game API.
Students can use this to programmatically play the game and build automated answering systems.

Basic Usage:
    from millionaire_client import MillionaireClient
    
    client = MillionaireClient("http://localhost:4000")
    client.login("username", "password")
    
    # Start a game
    game = client.game.start(competition_id=1)
    
    # Answer questions
    while game.in_progress:
        question = game.current_question
        print(f"Q: {question.text}")
        for opt in question.options:
            print(f"  {opt.id}: {opt.text}")
        
        # Your answering logic here
        result = game.answer(option_id=1)
        print(f"Correct: {result.correct}")
"""

from .client import MillionaireClient
from .models import (
    User,
    Competition,
    Question,
    Option,
    GameState,
    AnswerResult,
    MoneyLevel,
    LeaderboardEntry,
    CompetitionConfig,
)
from .game import GameSession
from .exceptions import (
    MillionaireError,
    AuthenticationError,
    GameError,
    TimeoutError,
    ValidationError,
    NotFoundError,
    ServerError,
    RateLimitError,
)

__version__ = "1.0.0"
__all__ = [
    "MillionaireClient",
    "User",
    "Competition",
    "Question",
    "Option",
    "GameSession",
    "GameState",
    "AnswerResult",
    "MoneyLevel",
    "LeaderboardEntry",
    "CompetitionConfig",
    "MillionaireError",
    "AuthenticationError",
    "GameError",
    "TimeoutError",
    "ValidationError",
    "NotFoundError",
    "ServerError",
    "RateLimitError",
]