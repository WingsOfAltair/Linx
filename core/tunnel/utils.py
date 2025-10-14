"""
Utility functions for tunnel handling.
"""
import asyncio
import logging
from asyncio.subprocess import Process
from typing import Any, Dict, Optional, Tuple

from . import TunnelProvider

logger = logging.getLogger(__name__)

async def auto_start_tunnel(
    port: int,
    host: str,
    config: Dict[str, Any],
) -> Optional[Tuple[str, Process]]:
    """
    Start a tunnel using the configured provider.

    Args:
        port: The local port to tunnel to
        host: The local host to tunnel from
        config: The tunnel configuration dictionary

    Returns:
        Optional[Tuple[str, Process]]: A tuple of (tunnel_url, process)
            if successful, or None if the tunnel couldn't be started
    """
    tunnel_type = config.get("type", "localhost_run")
    provider = TunnelProvider.create(tunnel_type, config)

    result = None
    tries = 0
    max_tries = 3

    while tries < max_tries and not result:
        result = await provider.start(port, host)
        if not result:
            tries += 1
            if tries < max_tries:
                logger.info(f"Retrying tunnel start ({tries}/{max_tries})...")
                await asyncio.sleep(1)

    if not result and tunnel_type != "localhost_run":
        logger.info(
            f"Failed to start {tunnel_type} tunnel, falling back to localhost.run..."
        )
        provider = TunnelProvider.create("localhost_run", config)
        result = await provider.start(port, host)

    return result


def start_tunnel_in_thread(port: int, host: str, config: Dict[str, Any]) -> None:
    """Synchronous wrapper to start the async tunnel function in a thread.

    Args:
        port: The local port to tunnel to
        host: The local host to tunnel from
        config: The tunnel configuration dictionary
    """
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(auto_start_tunnel(port, host, config))

        if result:
            tunnel_url, _ = result
            # Display instructions
            print(f"\n{'-' * 40}")
            print("Tunnel started successfully!")
            api_url = f"{tunnel_url}/v1"
            print("\nUse this URL in Cursor AI:")
            print(f"\033[1m{api_url}\033[0m")
            print("\nInstructions:")
            print("1. In Cursor, go to settings > AI > Configure AI Provider")
            print("2. Choose OpenAI Compatible and paste the URL above")
            print("3. Chat with local Ollama models in Cursor!")
            print(f"\n{'-' * 40}\n")

    except Exception as e:
        logger.error(f"Tunnel thread failed: {e}")
        raise
    finally:
        if loop:
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()
            except Exception as e:
                logger.debug(f"Error during loop cleanup: {e}")
