# OllamaLink Tunnel Support

OllamaLink supports multiple tunnel providers to make your local Ollama models accessible from other devices or locations. This document covers the available tunnel options and how to configure them.

## Available Tunnel Providers

### localhost.run (Default)
[localhost.run](https://localhost.run) provides free tunnels using SSH. This is the default tunnel provider and requires no additional setup.

Benefits:
- No registration required
- Free to use
- Automatic HTTPS
- Uses standard SSH client

### ngrok
[ngrok](https://ngrok.com) is a popular tunneling service that offers both free and paid plans with additional features.

Benefits:
- More stable connections
- Regional endpoint selection
- Better performance
- Advanced features with paid plans

## Configuration

### Command Line Options

The tunnel behavior can be controlled using command-line arguments:

```bash
# Enable/disable tunneling
ollamalink-cli --tunnel       # Enable tunneling (default)
ollamalink-cli --no-tunnel    # Disable tunneling

# Select tunnel provider
ollamalink-cli --tunnel-type localhost_run    # Use localhost.run (default)
ollamalink-cli --tunnel-type ngrok            # Use ngrok
```

### Configuration File

Tunnel settings can be configured in `config.json`:

```json
{
    "tunnel": {
        "use_tunnel": true,
        "type": "localhost_run",
        "providers": {
            "localhost_run": {},
            "ngrok": {
                "authtoken": "your-ngrok-auth-token",
                "region": "us"
            }
        }
    }
}
```

#### localhost.run Options
localhost.run requires no additional configuration.

#### ngrok Options
- `authtoken`: Your ngrok authentication token
- `region`: Region for tunnel endpoint (us, eu, ap, au, sa, jp, in)
- `edge`: Edge configuration (optional)

## Installation

### localhost.run
The localhost.run tunnel provider only requires SSH to be installed and available in your PATH.

### ngrok
1. Install the ngrok dependency:
   ```bash
   pip install ollamalink[ngrok]
   ```

2. Get your ngrok authentication token:
   - Sign up at [ngrok.com](https://ngrok.com)
   - Copy your authtoken from your ngrok dashboard
   - Add it to your config.json

## Troubleshooting

### localhost.run Issues
- **SSH not found**: Make sure SSH is installed and available in your PATH
- **Connection refused**: Check your internet connection
- **Permission denied**: This is normal, the tunnel uses 'nokey@localhost.run'

### ngrok Issues
- **Invalid authtoken**: Verify your ngrok authtoken in config.json
- **Tunnel quota exceeded**: Free ngrok accounts have usage limits
- **Region not available**: Try a different region or upgrade your ngrok plan

## Notes

- If a tunnel provider fails to connect, OllamaLink will automatically fall back to localhost.run
- The tunnel URL is displayed in the console when the server starts
- For security, always use HTTPS URLs provided by the tunnel services
- Tunnel connections are terminated when OllamaLink is stopped