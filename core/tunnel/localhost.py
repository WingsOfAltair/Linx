"""
localhost.run tunnel provider for Linx.
"""
import asyncio
import logging
import platform
import re
import subprocess
from asyncio.subprocess import Process
from typing import Optional, Tuple

from . import TunnelProvider

logger = logging.getLogger(__name__)

class LocalhostRunProvider(TunnelProvider):
    """Tunnel provider implementation for localhost.run."""

    async def start(self, port: int, host: str) -> Optional[Tuple[str, Process]]:
        """Start a localhost.run tunnel using SSH and return the URL and process.

        Args:
            port: The local port to tunnel to
            host: The local host to tunnel from

        Returns:
            Optional[Tuple[str, Process]]: A tuple of (tunnel_url, process)
                if successful, or None if the tunnel couldn't be started
        """
        logger.info("Starting localhost.run tunnel...")

        # Check if another instance of the tunnel might already be running
        try:
            if platform.system() == "Windows":
                check_process = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq ssh.exe"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if "localhost.run" in check_process.stdout:
                    cmd = [
                        "taskkill", "/F", "/FI",
                        "WINDOWTITLE eq localhost.run", "/T"
                    ]
                    try:
                        subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    except Exception as e:
                        logger.debug(f"Error terminating process: {str(e)}")
            else:
                check_process = subprocess.run(
                    ["ps", "aux"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if "localhost.run" in check_process.stdout:
                    try:
                        subprocess.run(
                            ["pkill", "-f", "ssh.*localhost.run"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    except Exception as e:
                        logger.debug(f"Error killing SSH process: {str(e)}")
        except Exception as e:
            logger.debug(f"Could not check for existing tunnel processes: {str(e)}")

        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["where", "ssh"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                result = subprocess.run(
                    ["which", "ssh"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            if result.returncode != 0:
                logger.error("SSH is not installed or not found in PATH")
                return None

            # Start SSH tunnel to localhost.run
            # Create tunnel to local server on specified port
            process = await asyncio.create_subprocess_exec(
                "ssh",
                "-o", "ServerAliveInterval=60",
                "-o", "ServerAliveCountMax=60",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ExitOnForwardFailure=yes",
                "-R", f"80:{host}:{port}",
                "nokey@localhost.run",
                "--",
                "--inject-http-proxy-headers",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            self._process = process
            tunnel_url = None
            start_time = asyncio.get_event_loop().time()
            timeout = 120

            logger.info("Waiting for localhost.run tunnel to start...")

            while asyncio.get_event_loop().time() - start_time < timeout:
                if process.stdout:
                    line = await process.stdout.readline()
                    if not line:
                        if process.returncode is not None:
                            logger.error(
                                f"Tunnel exited with code: {process.returncode}"
                            )
                            return None
                        continue

                    line_str = line.decode('utf-8', errors='ignore')
                    logger.debug(f"localhost.run: {line_str.strip()}")

                    if "admin.localhost.run" in line_str:
                        logger.debug(
                            "Ignoring admin URL - this is not a valid tunnel URL"
                        )
                        continue

                    # Several detection patterns, in order of reliability:
                    # Pattern 1: Line contains "tunneled with tls termination"
                    if "tunneled with tls termination" in line_str:
                        match = re.search(
                            r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)',
                            line_str
                        )
                        if match:
                            found_url = match.group(0)
                            if "admin.localhost.run" not in found_url:
                                tunnel_url = found_url
                                logger.info(f"Found tunnel URL (TLS): {tunnel_url}")
                                self._url = tunnel_url
                                return tunnel_url, process

                    # Pattern 2: Line containing "forwarding" info
                    elif f"forwarding to localhost:{port}" in line_str:
                        match = re.search(
                            r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)',
                            line_str
                        )
                        if match:
                            found_url = match.group(0)
                            if "admin.localhost.run" not in found_url:
                                tunnel_url = found_url
                                logger.info(f"Found tunnel URL: {tunnel_url}")
                                self._url = tunnel_url
                                return tunnel_url, process

                    # Pattern 3: Line contains "Follow" link
                    elif "Follow" in line_str:
                        match = re.search(
                            r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)',
                            line_str
                        )
                        if match:
                            found_url = match.group(0)
                            if "admin.localhost.run" not in found_url:
                                tunnel_url = found_url
                                logger.info(f"Found tunnel URL: {tunnel_url}")
                                self._url = tunnel_url
                                return tunnel_url, process

                    # Pattern 4: General URL detection - fallback
                    elif re.search(
                        r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)',
                        line_str
                    ):
                        match = re.search(
                            r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)',
                            line_str
                        )
                        if match:
                            found_url = match.group(0)
                            if "admin.localhost.run" not in found_url:
                                tunnel_url = found_url
                                logger.info(f"Found tunnel URL: {tunnel_url}")
                                self._url = tunnel_url
                                return tunnel_url, process

                    # Error detection
                    if "permission denied" in line_str.lower():
                        logger.debug("SSH permission denied (using nokey)")
                        if "publickey" in line_str.lower():
                            logger.debug("SSH keys are being ignored")

                    if "connection refused" in line_str.lower():
                        logger.error("Connection refused to localhost.run")

                    if "no route to host" in line_str.lower():
                        logger.error(
                            "No route to localhost.run - check internet connection"
                        )

                await asyncio.sleep(0.1)

            logger.error("Timed out waiting for localhost.run tunnel URL")
            return None

        except Exception as e:
            logger.error(f"Error starting localhost.run tunnel: {str(e)}")
            return None
