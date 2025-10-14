"""
ngrok tunnel provider for OllamaLink.
"""
import asyncio
import logging
import platform
import shutil
import subprocess
import urllib.request
from asyncio.subprocess import Process
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import TunnelProvider

logger = logging.getLogger(__name__)

NGROK_DOWNLOAD_URLS = {
    "Windows": {
        "amd64": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip",
        "386": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-386.zip"
    },
    "Linux": {
        "amd64": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip",
        "386": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-386.zip",
        "arm": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-arm.zip",
        "arm64": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-arm64.zip"
    },
    "Darwin": {
        "amd64": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip",
        "arm64": "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-arm64.zip"
    }
}

class NgrokProvider(TunnelProvider):
    """Tunnel provider implementation for ngrok."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the ngrok tunnel provider."""
        super().__init__(config)
        self.ngrok_path = self._get_ngrok_path()

    def _get_ngrok_path(self) -> Optional[Path]:
        """Get path to ngrok binary, downloading if necessary."""
        # Check if ngrok is in PATH
        ngrok_cmd = "ngrok.exe" if platform.system() == "Windows" else "ngrok"
        which_path = shutil.which(ngrok_cmd)
        if which_path is not None:
            return Path(which_path)

        # Check for bundled ngrok
        app_dir = Path.home() / ".ollamalink"
        app_dir.mkdir(exist_ok=True)
        ngrok_dir = app_dir / "ngrok"
        ngrok_dir.mkdir(exist_ok=True)

        if platform.system() == "Windows":
            ngrok_path = ngrok_dir / "ngrok.exe"
        else:
            ngrok_path = ngrok_dir / "ngrok"

        if ngrok_path.exists():
            return ngrok_path

        # Download ngrok
        try:
            system = platform.system()
            machine = platform.machine().lower()

            # Map machine architecture to ngrok download key
            arch_map = {
                "amd64": "amd64",
                "x86_64": "amd64",
                "x64": "amd64",
                "i386": "386",
                "i686": "386",
                "x86": "386",
                "armv7l": "arm",
                "armv6l": "arm",
                "aarch64": "arm64",
                "arm64": "arm64"
            }

            arch = arch_map.get(machine)
            if not arch:
                logger.error(f"Unsupported architecture: {machine}")
                return None

            if system not in NGROK_DOWNLOAD_URLS:
                logger.error(f"Unsupported operating system: {system}")
                return None

            if arch not in NGROK_DOWNLOAD_URLS[system]:
                logger.error(f"Unsupported architecture for {system}: {arch}")
                return None

            download_url = NGROK_DOWNLOAD_URLS[system][arch]
            zip_path = ngrok_dir / "ngrok.zip"

            logger.info(f"Downloading ngrok from {download_url}")
            urllib.request.urlretrieve(download_url, zip_path)

            import zipfile
            with zipfile.ZipFile(zip_path) as zip_ref:
                zip_ref.extractall(ngrok_dir)

            zip_path.unlink()

            # Make binary executable on Unix
            if platform.system() != "Windows":
                ngrok_path.chmod(0o755)

            return ngrok_path

        except Exception as e:
            logger.error(f"Error downloading ngrok: {str(e)}")
            return None

    async def start(self, port: int, host: str) -> Optional[Tuple[str, Process]]:
        """Start an ngrok tunnel and return the URL and process.

        Args:
            port: The local port to tunnel to
            host: The local host to tunnel from

        Returns:
            Optional[Tuple[str, Process]]: A tuple of (tunnel_url, process)
                if successful, or None if the tunnel couldn't be started
        """
        if not self.ngrok_path:
            logger.error("ngrok binary not found and could not be downloaded")
            return None

        ngrok_config = self.config["providers"]["ngrok"]

        try:
            # Kill any existing ngrok processes
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ngrok.exe"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            else:
                subprocess.run(
                    ["pkill", "ngrok"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
        except Exception as e:
            logger.debug(f"Error killing existing ngrok processes: {str(e)}")

        try:
            cmd = [
                str(self.ngrok_path),
                "http",
                "--log", "stdout",
                f"--region={ngrok_config.get('region', 'us')}"
            ]

            if ngrok_config.get("authtoken"):
                cmd.extend(["--authtoken", ngrok_config["authtoken"]])

            cmd.append(f"{host}:{port}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            self._process = process
            tunnel_url = None
            start_time = asyncio.get_event_loop().time()
            timeout = 30

            logger.info("Starting ngrok tunnel...")

            while asyncio.get_event_loop().time() - start_time < timeout:
                if process.stdout:
                    line = await process.stdout.readline()
                    if not line:
                        if process.returncode is not None:
                            logger.error(f"ngrok exited with code {process.returncode}")
                            return None
                        continue

                    line_str = line.decode('utf-8', errors='ignore')
                    logger.debug(f"ngrok: {line_str.strip()}")

                    # Look for URL in ngrok output
                    if "url=" in line_str:
                        parts = line_str.split("url=")
                        if len(parts) > 1:
                            tunnel_url = parts[1].strip()
                            logger.info(f"Tunnel URL found: {tunnel_url}")
                            self._url = tunnel_url
                            return tunnel_url, process

                    # Check for common errors
                    if "error" in line_str.lower():
                        if "invalid authtoken" in line_str.lower():
                            logger.error("Invalid ngrok authtoken")
                        elif "expired" in line_str.lower():
                            logger.error("Ngrok tunnel quota expired")
                        else:
                            logger.error(f"Ngrok error: {line_str.strip()}")
                        return None

                await asyncio.sleep(0.1)

            logger.error("Timed out waiting for ngrok tunnel URL")
            return None

        except Exception as e:
            logger.error(f"Error starting ngrok tunnel: {str(e)}")
            return None
