from datetime import datetime, timedelta, timezone

import jwt

from app.config import settings


class AuthService:
    """Issues and verifies opaque participant session tokens (JWT with short TTL)."""

    ALGORITHM = "HS256"

    @classmethod
    def create_token(cls, participant_id: int) -> str:
        """Create a session token for the participant."""
        payload = {
            "sub": str(participant_id),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_ttl_minutes),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, settings.jwt_secret, algorithm=cls.ALGORITHM)

    @classmethod
    def create_demo_token(cls) -> str:
        """Create an anonymous demo token: not bound to any participant row in the database."""
        payload = {
            "sub": "0",
            "demo": True,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.demo_ttl_minutes),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, settings.jwt_secret, algorithm=cls.ALGORITHM)

    @classmethod
    def is_demo_token(cls, payload: dict) -> bool:
        """Whether the decoded payload belongs to an anonymous demo session."""
        return bool(payload.get("demo", False))

    @classmethod
    def decode_payload(cls, token: str) -> dict:
        """Verify the token and return its payload.

        Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError on invalid tokens.
        """
        return jwt.decode(token, settings.jwt_secret, algorithms=[cls.ALGORITHM])

    @classmethod
    def decode_participant_id(cls, token: str) -> int:
        """Verify the token and return the participant id."""
        payload = cls.decode_payload(token)
        participant_id = int(payload["sub"])
        if participant_id < 0:
            raise jwt.InvalidTokenError("Invalid participant id in token")
        return participant_id
