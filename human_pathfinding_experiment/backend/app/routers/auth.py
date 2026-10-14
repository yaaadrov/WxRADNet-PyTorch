import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

from app.config import settings
from app.db.engine import get_session
from app.db.models import Participant
from app.services.auth_service import AuthService


bearer_scheme = HTTPBearer(auto_error=False)


def require_admin(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> None:
    """Guard admin endpoints (results download, summary/export) with a shared secret.

    When ADMIN_TOKEN is not configured the endpoints are disabled entirely (404),
    which keeps local docker-compose development usable without extra secrets.
    """
    if not settings.admin_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Недостаточно прав")


def get_current_participant(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_session),
) -> Participant:
    """Resolve the authenticated participant for the current request.

    Anonymous demo tokens resolve to a transient participant (id=0) that is not stored in the database.
    """
    invalid = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Требуется авторизация",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise invalid
    try:
        payload = AuthService.decode_payload(credentials.credentials)
    except jwt.InvalidTokenError:
        raise invalid from None

    if AuthService.is_demo_token(payload):
        # Demo session: transient participant, nothing is persisted
        return Participant(id=0, surname="demo", name="demo", group_name="demo")

    participant = session.get(Participant, int(payload["sub"]))
    if participant is None:
        raise invalid
    return participant
