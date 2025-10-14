"""
Tests for tunnel providers.
"""
import asyncio
import os
import json
import platform
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from core.tunnel import TunnelProvider
from core.tunnel.utils import auto_start_tunnel
from core.tunnel.localhost import LocalhostRunProvider
from core.tunnel.ngrok import NgrokProvider

@pytest.fixture
def base_config():
    return {
        "use_tunnel": True,
        "type": "localhost_run",
        "providers": {
            "localhost_run": {},
            "ngrok": {
                "authtoken": "test-token",
                "region": "us"
            }
        }
    }

@pytest.fixture
def mock_process():
    process = AsyncMock()
    process.stdout = AsyncMock()
    process.returncode = None
    return process

# Test TunnelProvider Factory
def test_tunnel_provider_factory(base_config):
    # Test localhost.run provider
    provider = TunnelProvider.create("localhost_run", base_config)
    assert isinstance(provider, LocalhostRunProvider)

    # Test ngrok provider
    provider = TunnelProvider.create("ngrok", base_config)
    assert isinstance(provider, NgrokProvider)

    # Test unknown provider
    with pytest.raises(ValueError):
        TunnelProvider.create("invalid", base_config)

# Test LocalhostRunProvider
@pytest.mark.asyncio
async def test_localhost_run_provider(base_config, mock_process):
    provider = LocalhostRunProvider(base_config)

    # Test successful tunnel start
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        mock_process.stdout.readline = AsyncMock(side_effect=[
            b"some output\n",
            b"Forwarding traffic to localhost:8080\n",
            b"https://test.localhost.run tunneled with tls termination\n",
            b""
        ])
        result = await provider.start(8080, "127.0.0.1")
        
        assert result is not None
        assert result[0] == "https://test.localhost.run"
        assert result[1] == mock_process
        assert provider.url == "https://test.localhost.run"

    # Test tunnel start failure
    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        mock_process.stdout.readline = AsyncMock(side_effect=[
            b"some output\n",
            b"error: connection refused\n",
            b""
        ])
        mock_process.returncode = 1
        result = await provider.start(8080, "127.0.0.1")
        assert result is None

# Test NgrokProvider
@pytest.mark.asyncio
async def test_ngrok_provider(base_config, mock_process, tmp_path):
    provider = NgrokProvider(base_config)

    # Mock ngrok binary path
    ngrok_path = tmp_path / ("ngrok.exe" if platform.system() == "Windows" else "ngrok")
    ngrok_path.touch()
    if platform.system() != "Windows":
        ngrok_path.chmod(0o755)
    
    with patch.object(provider, "_get_ngrok_path", return_value=ngrok_path):
        # Test successful tunnel start
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            mock_process.stdout.readline = AsyncMock(side_effect=[
                b"some output\n",
                b"url=https://test.ngrok.io\n",
                b""
            ])
            result = await provider.start(8080, "127.0.0.1")
            
            assert result is not None
            assert result[0] == "https://test.ngrok.io"
            assert result[1] == mock_process
            assert provider.url == "https://test.ngrok.io"

        # Test tunnel start failure
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            mock_process.stdout.readline = AsyncMock(side_effect=[
                b"some output\n",
                b"error: invalid authtoken\n",
                b""
            ])
            mock_process.returncode = 1
            result = await provider.start(8080, "127.0.0.1")
            assert result is None

# Test auto_start_tunnel utility
@pytest.mark.asyncio
async def test_auto_start_tunnel(base_config, mock_process):
    # Test successful start with primary provider
    with patch("core.tunnel.localhost.LocalhostRunProvider.start", 
               new_callable=AsyncMock) as mock_start:
        mock_start.return_value = ("https://test.localhost.run", mock_process)
        result = await auto_start_tunnel(8080, "127.0.0.1", base_config)
        assert result[0] == "https://test.localhost.run"
        assert mock_start.call_count == 1

    # Test fallback to localhost.run
    base_config["type"] = "ngrok"
    with patch("core.tunnel.ngrok.NgrokProvider.start", 
               new_callable=AsyncMock) as mock_ngrok_start:
        with patch("core.tunnel.localhost.LocalhostRunProvider.start",
                  new_callable=AsyncMock) as mock_lhr_start:
            mock_ngrok_start.return_value = None
            mock_lhr_start.return_value = ("https://test.localhost.run", mock_process)
            result = await auto_start_tunnel(8080, "127.0.0.1", base_config)
            assert result[0] == "https://test.localhost.run"
            assert mock_ngrok_start.call_count == 3  # Should try 3 times before fallback
            assert mock_lhr_start.call_count == 1  # Should try localhost.run once

# Test NgrokProvider Binary Management
def test_ngrok_provider_binary_management(base_config, tmp_path):
    provider = NgrokProvider(base_config)

    # Test binary detection in PATH
    with patch("shutil.which", return_value="/usr/local/bin/ngrok"):
        path = provider._get_ngrok_path()
        assert path == Path("/usr/local/bin/ngrok")

    # Test binary download
    with patch("shutil.which", return_value=None), \
         patch("urllib.request.urlretrieve") as mock_download, \
         patch("zipfile.ZipFile"):
        path = provider._get_ngrok_path()
        assert path is not None
        assert mock_download.called

    # Test download error handling
    with patch("shutil.which", return_value=None), \
         patch("urllib.request.urlretrieve", side_effect=Exception("Download failed")):
        path = provider._get_ngrok_path()
        assert path is None