"""Data models for the Millionaire API responses."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class GameStatus(Enum):
    """Possible game session statuses."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class User:
    """Represents a user account."""
    id: int
    username: str
    role: str

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            id=data["id"],
            username=data["username"],
            role=data["role"]
        )


@dataclass
class Option:
    """Represents an answer option for a question."""
    id: int
    text: Optional[str] = None  # None in speech mode (audio only)

    @classmethod
    def from_dict(cls, data: dict) -> "Option":
        return cls(
            id=data["id"],
            text=data.get("text"),
        )


@dataclass
class Question:
    """Represents a quiz question."""
    id: int
    text: Optional[str] = None  # None in speech mode (audio only)
    options: List[Option] = field(default_factory=list)
    level: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(
            id=data["id"],
            text=data.get("text"),
            options=[Option.from_dict(opt) for opt in data.get("options", [])],
            level=data.get("level", 0),
        )

    def get_option_by_id(self, option_id: int) -> Optional[Option]:
        """Get an option by its ID."""
        for opt in self.options:
            if opt.id == option_id:
                return opt
        return None

    def get_option_by_text(self, text: str, case_sensitive: bool = False) -> Optional[Option]:
        """Get an option by its text content."""
        for opt in self.options:
            opt_text = opt.text if case_sensitive else opt.text.lower()
            search_text = text if case_sensitive else text.lower()
            if opt_text == search_text:
                return opt
        return None


@dataclass
class MoneyLevel:
    """Represents a prize level in the money pyramid."""
    level: int
    amount: float

    @classmethod
    def from_dict(cls, data: dict) -> "MoneyLevel":
        return cls(
            level=data["level"],
            amount=data["amount"]
        )


@dataclass
class Competition:
    """Represents a game competition.

    Note: The id field is a public sequential ID (0, 1, 2, ...) assigned by the server,
    not the internal database ID. This ensures consistent, user-friendly identifiers
    even when competitions are deleted.
    """
    id: int
    name: str
    description: Optional[str] = None
    max_levels: int = 15
    is_infinite: bool = False
    created_at: Optional[str] = None
    question_count: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Competition":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            max_levels=data.get("maxLevels", 15),
            is_infinite=data.get("isInfinite", False),
            created_at=data.get("createdAt"),
            question_count=data.get("questionCount")
        )


@dataclass
class PrizeConfig:
    """Represents prize configuration for a competition."""
    type: str
    base_amount: float
    growth_rate: float
    milestone_levels: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PrizeConfig":
        return cls(
            type=data["type"],
            base_amount=data["baseAmount"],
            growth_rate=data["growthRate"],
            milestone_levels=data.get("milestoneLevels", [])
        )


@dataclass
class CompetitionConfig:
    """Represents full competition configuration including prize pyramid."""
    id: int
    name: str
    max_levels: int
    is_infinite: bool
    prize_config: Optional[PrizeConfig] = None
    money_pyramid: List[MoneyLevel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "CompetitionConfig":
        return cls(
            id=data["id"],
            name=data["name"],
            max_levels=data["maxLevels"],
            is_infinite=data["isInfinite"],
            prize_config=PrizeConfig.from_dict(data["prizeConfig"]) if data.get("prizeConfig") else None,
            money_pyramid=[MoneyLevel.from_dict(ml) for ml in data.get("moneyPyramid", [])]
        )


@dataclass
class GameState:
    """Represents the current state of a game session."""
    session_id: int
    competition: Competition
    status: GameStatus
    earned_amount: float
    current_level: int
    money_pyramid: List[MoneyLevel]
    question_deadline: Optional[datetime] = None
    question: Optional[Question] = None
    max_level: Optional[int] = None
    mode: str = "text"

    @classmethod
    def from_dict(cls, data: dict) -> "GameState":
        deadline = data.get("questionDeadline")
        if deadline:
            try:
                deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                deadline = None

        return cls(
            session_id=data.get("sessionId", data.get("id", 0)),
            competition=Competition.from_dict(data["competition"]),
            status=GameStatus(data.get("status", "in_progress")),
            earned_amount=data.get("earnedAmount", 0),
            current_level=data.get("currentLevel", 1),
            money_pyramid=[MoneyLevel.from_dict(ml) for ml in data.get("moneyPyramid", [])],
            question_deadline=deadline,
            question=Question.from_dict(data["question"]) if data.get("question") else None,
            max_level=data.get("maxLevel"),
            mode=data.get("mode", "text"),
        )

    @property
    def in_progress(self) -> bool:
        """Check if the game is still in progress."""
        return self.status == GameStatus.IN_PROGRESS

    @property
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.status in (GameStatus.COMPLETED, GameStatus.FAILED, GameStatus.TIMEOUT)

    @property
    def time_remaining(self) -> Optional[float]:
        """Get seconds remaining to answer the current question."""
        if not self.question_deadline:
            return None
        remaining = (self.question_deadline - datetime.now(self.question_deadline.tzinfo)).total_seconds()
        return max(0, remaining)

    def get_safe_amount(self) -> float:
        """Get the safe amount (amount that would be kept on wrong answer)."""
        for level in reversed(self.money_pyramid):
            if level.level < self.current_level:
                return level.amount
        return 0


@dataclass
class AnswerResult:
    """Represents the result of submitting an answer."""
    correct: Optional[bool] = None
    game_over: bool = False
    earned_amount: float = 0
    timed_out: bool = False
    status: Optional[str] = None
    current_level: Optional[int] = None
    reached_level: Optional[int] = None
    question_deadline: Optional[datetime] = None
    question: Optional[Question] = None
    money_pyramid: List[MoneyLevel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "AnswerResult":
        deadline = data.get("questionDeadline")
        if deadline:
            try:
                deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                deadline = None

        return cls(
            correct=data.get("correct"),
            game_over=data.get("gameOver", False),
            earned_amount=data.get("earnedAmount", 0),
            timed_out=data.get("timedOut", False),
            status=data.get("status"),
            current_level=data.get("currentLevel"),
            reached_level=data.get("reachedLevel"),
            question_deadline=deadline,
            question=Question.from_dict(data["question"]) if data.get("question") else None,
            money_pyramid=[MoneyLevel.from_dict(ml) for ml in data.get("moneyPyramid", [])]
        )


@dataclass
class LeaderboardEntry:
    """Represents a leaderboard entry."""
    id: int
    username: str
    score: float
    reached_level: int
    finished_at: Optional[str] = None
    total_trials: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "LeaderboardEntry":
        return cls(
            id=data["id"],
            username=data["username"],
            score=data["score"],
            reached_level=data["reachedLevel"],
            finished_at=data.get("finishedAt"),
            total_trials=data.get("totalTrials", 1)
        )


@dataclass
class Leaderboard:
    """Represents a competition leaderboard."""
    competition: Competition
    entries: List[LeaderboardEntry]

    @classmethod
    def from_dict(cls, data: dict) -> "Leaderboard":
        return cls(
            competition=Competition.from_dict(data["competition"]),
            entries=[LeaderboardEntry.from_dict(e) for e in data.get("entries", [])]
        )