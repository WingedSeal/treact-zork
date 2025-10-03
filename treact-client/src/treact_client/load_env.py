import os

from pydantic import BaseModel, Field

from .log import get_logger


class Environment(BaseModel):
    SERVER_IP: str = Field(
        ..., min_length=1, description="IP address the server binds to"
    )
    SERVER_PORT: int = Field(..., ge=1, le=65535, description="Server port")
    CLIENT_PORT: int = Field(..., ge=1, le=65535, description="Client port")
    API_KEY: str = Field(..., min_length=1, description="API key must not be empty")


env = Environment(
    SERVER_IP=os.getenv("SERVER_IP") or "",
    SERVER_PORT=int(os.getenv("SERVER_PORT") or 0),
    CLIENT_PORT=int(os.getenv("CLIENT_PORT") or 0),
    API_KEY=os.getenv("API_KEY") or "",
)

logger = get_logger(__name__)
logger.debug(f"Environment loaded: {env}")
