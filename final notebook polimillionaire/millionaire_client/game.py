"""Game module for managing game sessions."""

from typing import Optional, List
from .base import BaseClient
from .models import GameState, AnswerResult, Question, Option
from .exceptions import GameError, TimeoutError


__all__ = ["GameSession", "GameModule"]


class GameSession:
    """
    Represents an active game session.
    
    This class provides a convenient interface for interacting with a specific
    game session, handling the session ID automatically.
    """
    
    def __init__(self, client: BaseClient, state: GameState):
        self._client = client
        self._state = state
    
    @property
    def session_id(self) -> int:
        """Get the session ID."""
        return self._state.session_id
    
    @property
    def state(self) -> GameState:
        """Get the current game state."""
        return self._state
    
    @property
    def current_question(self) -> Optional[Question]:
        """Get the current question, if any."""
        return self._state.question
    
    @property
    def current_level(self) -> int:
        """Get the current level."""
        return self._state.current_level
    
    @property
    def earned_amount(self) -> float:
        """Get the amount earned so far."""
        return self._state.earned_amount
    
    @property
    def in_progress(self) -> bool:
        """Check if the game is still in progress."""
        return self._state.in_progress
    
    @property
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self._state.is_game_over
    
    @property
    def time_remaining(self) -> Optional[float]:
        """Get seconds remaining to answer the current question."""
        return self._state.time_remaining
    
    @property
    def money_pyramid(self) -> List:
        """Get the money pyramid."""
        return self._state.money_pyramid
    
    @property
    def mode(self) -> str:
        """Get the game mode (text or speech)."""
        return self._state.mode

    def fetch_audio_question(self) -> bytes:
        """Fetch the current question audio (speech mode only). Returns WAV bytes."""
        if self.mode != "speech":
            raise GameError("Audio is only available in speech mode")
        return self._client.get(
            f"/api/game/{self.session_id}/audio/question",
            raw=True,
        )

    def fetch_audio_option_next(self) -> bytes:
        """Fetch the next option audio in sequence A→B→C→D (speech mode only)."""
        if self.mode != "speech":
            raise GameError("Audio is only available in speech mode")
        return self._client.get(
            f"/api/game/{self.session_id}/audio/option/next",
            raw=True,
        )

    def fetch_audio_option(self, index: int) -> bytes:
        """Fetch an already-delivered option audio by index 0-3 (speech mode only)."""
        if self.mode != "speech":
            raise GameError("Audio is only available in speech mode")
        if not 0 <= index <= 3:
            raise ValueError("Option index must be 0-3")
        return self._client.get(
            f"/api/game/{self.session_id}/audio/option/{index}",
            raw=True,
        )

    def refresh_state(self) -> GameState:
        """
        Refresh the game state from the server.
        
        Returns:
            Updated GameState
        """
        response = self._client.get(f"/api/game/{self.session_id}/state")
        self._state = GameState.from_dict(response)
        return self._state
    
    def answer(self, option_id: int) -> AnswerResult:
        """
        Submit an answer to the current question.
        
        Args:
            option_id: The ID of the selected option
            
        Returns:
            AnswerResult with the outcome
            
        Raises:
            GameError: If no question is available or game is over
            TimeoutError: If the question has timed out
        """
        if not self._state.question:
            raise GameError("No active question to answer")
        
        try:
            response = self._client.post(
                f"/api/game/{self.session_id}/answer",
                data={"optionId": option_id}
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                raise TimeoutError("Question timed out")
            raise
        
        result = AnswerResult.from_dict(response)

        # Handle timeout
        if result.timed_out or result.status == "timeout":
            from .models import GameStatus
            self._state.status = GameStatus.TIMEOUT
            self._state.earned_amount = result.earned_amount
            self._state.question = None
            return result

        # Update internal state if game continues
        if result.question:
            self._state = GameState.from_dict({
                "sessionId": self.session_id,
                "competition": self._state.competition.__dict__,
                "status": "in_progress" if not result.game_over else "completed",
                "earnedAmount": result.earned_amount,
                "currentLevel": result.current_level or self._state.current_level,
                "moneyPyramid": [ml.__dict__ for ml in result.money_pyramid] if result.money_pyramid else self._state.money_pyramid,
                "questionDeadline": result.question_deadline.isoformat() if result.question_deadline else None,
                "mode": self._state.mode,
                "question": {
                    "id": result.question.id,
                    "level": result.question.level,
                    "text": result.question.text,
                    "options": [{"id": opt.id, "text": opt.text} for opt in result.question.options]
                } if result.question else None
            })
        elif result.game_over:
            from .models import GameStatus
            if result.correct:
                self._state.status = GameStatus.COMPLETED
            elif result.correct is False:
                self._state.status = GameStatus.FAILED
            self._state.earned_amount = result.earned_amount
            self._state.question = None

        return result
    
    def answer_by_text(self, answer_text: str, case_sensitive: bool = False) -> AnswerResult:
        """
        Submit an answer by matching option text.
        
        Args:
            answer_text: The text of the answer to select
            case_sensitive: Whether text matching is case-sensitive
            
        Returns:
            AnswerResult with the outcome
            
        Raises:
            GameError: If no matching option is found
        """
        if not self._state.question:
            raise GameError("No active question to answer")
        
        option = self._state.question.get_option_by_text(answer_text, case_sensitive)
        if not option:
            available = [opt.text for opt in self._state.question.options]
            raise GameError(f"Answer '{answer_text}' not found. Available: {available}")
        
        return self.answer(option.id)
    
    def timeout(self):
        """
        Signal that the current question has timed out.
        This is typically called automatically by the server.
        """
        return self._client.post(f"/api/game/{self.session_id}/timeout")


class GameModule:
    """Handles game-related operations."""
    
    def __init__(self, client: BaseClient):
        self._client = client
    
    def start(self, competition_id: int, mode: str = "text") -> GameSession:
        """
        Start a new game session.

        Args:
            competition_id: The ID of the competition to play
            mode: "text" (default) or "speech"

        Returns:
            GameSession object for the new game
        """
        response = self._client.post(
            "/api/game/start",
            data={"competitionId": competition_id, "mode": mode}
        )
        state = GameState.from_dict(response)
        return GameSession(self._client, state)
    
    def get_state(self, session_id: int) -> GameState:
        """
        Get the state of an existing game session.
        
        Args:
            session_id: The game session ID
            
        Returns:
            GameState object
        """
        response = self._client.get(f"/api/game/{session_id}/state")
        return GameState.from_dict(response)