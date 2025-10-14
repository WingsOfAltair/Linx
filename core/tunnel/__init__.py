"""
Tunnel provider interface and factory for Linx.
"""
import abc
import logging
from asyncio.subprocess import Process
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class TunnelProvider(abc.ABC):
    """Abstract base class for tunnel providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the tunnel provider with configuration."""
        self.config = config
        self._process: Optional[Process] = None
        self._url: Optional[str] = None

    @property
    def url(self) -> Optional[str]:
        """Get the current tunnel URL."""
        return self._url

    @property
    def process(self) -> Optional[Process]:
        """Get the tunnel process handle."""
        return self._process

    @abc.abstractmethod
    async def start(self, port: int, host: str) -> Optional[Tuple[str, Process]]:
        """Start the tunnel.

        Args:
            port: The local port to tunnel to
            host: The local host to tunnel from

        Returns:
            Optional[Tuple[str, Process]]: A tuple of (tunnel_url, process)
                if successful, or None if the tunnel couldn't be started
        """
        raise NotImplementedError

    @classmethod
    def create(cls, tunnel_type: str, config: Dict[str, Any]) -> 'TunnelProvider':
        """Factory method to create a tunnel provider instance.

        Args:
            tunnel_type: Type of tunnel to create ('localhost_run' or 'ngrok')
            config: The tunnel configuration dictionary

        Returns:
            TunnelProvider: An instance of the appropriate tunnel provider

        Raises:
            ValueError: If the tunnel type is not supported
        """
        if tunnel_type == "localhost_run":
            from .localhost import LocalhostRunProvider
            return LocalhostRunProvider(config)
        elif tunnel_type == "ngrok":
            from .ngrok import NgrokProvider
            return NgrokProvider(config)
        else:
            raise ValueError(f"Unknown tunnel type: {tunnel_type}")