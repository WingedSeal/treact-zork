from .log import get_logger


logger = get_logger(__name__)


class NoSessionException(Exception):
    def __init__(self) -> None:
        msg = "Session not found. Run MCPClient.connect_to_server to start new session.\n(Either self.session or self.agent is still None.)"
        logger.critical(msg)
        super().__init__(msg)


class AINotInvokedException(Exception):
    def __init__(self) -> None:
        msg = "Agent was never invoked. Run MCPClient.invoke_agent to invoke it.\n(ai_model_response is None.)"
        logger.critical(msg)
        super().__init__(msg)


class UnreachableServerException(Exception):
    def __init__(self, url: str) -> None:
        msg = f"MCP-Server at {url} is unreachable as it appears to be down."
        logger.critical(msg)
        super().__init__(msg)


class InvalidToolCallResultException(Exception):
    def __init__(self, content_type: type) -> None:
        msg = f"Expected TextContent from tool (got {content_type})."
        logger.critical(msg)
        super().__init__(msg)
