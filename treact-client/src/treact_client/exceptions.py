from .log import get_logger

logger = get_logger(__name__)


class MCPClientException(Exception):
    def __init__(self, msg: str) -> None:
        logger.critical(msg)
        super().__init__(msg)


class NoSessionException(MCPClientException):
    def __init__(self) -> None:
        msg = "Session not found. Run MCPClient.connect_to_server to start new session.\n(Either self.session or self.agent is still None.)"
        super().__init__(msg)


class AINotInvokedException(MCPClientException):
    def __init__(self) -> None:
        msg = "Agent was never invoked. Run MCPClient.invoke_agent to invoke it.\n(ai_model_response is None.)"
        super().__init__(msg)


class UnreachableServerException(MCPClientException):
    def __init__(self, url: str) -> None:
        msg = f"MCP-Server at {url} is unreachable as it appears to be down."
        super().__init__(msg)


class InvalidToolCallResultException(MCPClientException):
    def __init__(self, content_type: type) -> None:
        msg = f"Expected TextContent from tool (got {content_type})."
        super().__init__(msg)


class MaxBranchPerNodeExceededException(MCPClientException):
    def __init__(self, branch_per_node: int, max_branch_per_node: int) -> None:
        msg = f"Max branch per node exceeded the limit of {max_branch_per_node}. (got {branch_per_node})"
        super().__init__(msg)
