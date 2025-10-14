# Linx

**Unify local and remote models into one OpenAI-compatible endpoint**

Linx is a bridge application that connects **local models** (via Ollama or Llama.cpp) and **remote models** (via OpenRouter.ai or other OpenAI-compatible providers) under a single unified API. It exposes all connected models through an **OpenAI-compatible interface**, allowing seamless use in applications like **Cursor AI**, **VSCode extensions**, or any client supporting the OpenAI API format. Both **CLI** and **GUI** versions exist, with the CLI being fully functional and the GUI in active development.

## Features

* **Unified Endpoint** — Merge local and remote models into one `/v1` API
* **Multi-Provider Support** — Ollama, Llama.cpp, OpenRouter, and OpenAI-compatible APIs
* **OpenAI-Compatible** — Works with any OpenAI-style client (Cursor, Continue, etc.)
* **Privacy First** — Keep your data local with Ollama or Llama.cpp
* **Smart Routing** — Automatic provider selection with intelligent fallback
* **Tunneling** — Public access via `localhost.run` or ngrok
* **CLI & GUI** — Command-line interface ready, GUI in development
* **Model Mapping** — Custom model name aliases across providers
* **Secure** — Optional API key authentication
* **Stream Support** — Full streaming for real-time responses
* **No Timeout Limits** — Long-running tasks supported

## Quick Start

### 1. Install a Local Provider (Optional)

**Option A: Ollama**
```sh
ollama serve
```

**Option B: Llama.cpp Server**
```sh
./llama-server -m model.gguf --port 8080
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure Linx

Edit `config.json` to configure your providers (see Configuration section below).

### 4. Run Linx

**CLI Mode:**
```sh
python run_cli.py
```

**With Options:**
```sh
python run_cli.py --port 8080 --tunnel
```

*Note: Electron GUI is in active development.*

## Integration

Linx works with any OpenAI-compatible tool:

* **Cursor AI** — Set API URL to `http://localhost:8080/v1`
* **Continue.dev** — Configure as OpenAI-compatible provider
* **VSCode Extensions** — Use Linx endpoint for AI features
* **Custom Applications** — Query via standard OpenAI API format

## API Endpoints

**Base URL:** `http://localhost:8080`

### OpenAI-Compatible Endpoints
* `GET /v1/models` — List available models
* `POST /v1/chat/completions` — Chat completions (streaming & non-streaming)

### Ollama Proxy Endpoints
* `GET /api/tags` — List Ollama models
* `POST /api/chat` — Ollama native chat (NDJSON)
* `POST /api/generate` — Ollama generate endpoint
* `POST /api/show` — Model information

### Management Endpoints
* `GET /v1/providers/status` — Provider health status
* `POST /api/tunnel/start` — Start localhost.run tunnel
* `POST /api/tunnel/stop` — Stop tunnel
* `GET /api/tunnel/status` — Tunnel status

## Configuration

Example `config.json`:

```json
{
  "ollama": {
    "enabled": true,
    "endpoint": "http://localhost:11434",
    "thinking_mode": true,
    "model_mappings": {
      "gpt-4o": "qwen2.5-coder:32b",
      "gpt-4": "llama3.1:70b",
      "gpt-3.5-turbo": "llama3.2:3b",
      "default": "qwen2.5-coder:7b"
    }
  },
  "llamacpp": {
    "enabled": false,
    "endpoint": "http://localhost:8080",
    "model_mappings": {
      "gpt-4": "local-model"
    }
  },
  "openrouter": {
    "enabled": false,
    "api_key": "sk-or-v1-your-api-key-here",
    "endpoint": "https://openrouter.ai/api/v1",
    "model_mappings": {
      "gpt-4o": "openai/gpt-4o",
      "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
      "deepseek-chat": "deepseek/deepseek-chat"
    }
  },
  "routing": {
    "provider_priority": ["ollama", "llamacpp", "openrouter"],
    "fallback_enabled": true,
    "cost_optimization": true
  },
  "server": {
    "port": 8080,
    "hostname": "127.0.0.1"
  },
  "tunnel": {
    "use_tunnel": true,
    "type": "localhost_run"
  }
}
```

**Configuration Options:**

* **enabled** — Enable/disable provider
* **endpoint** — Provider API URL
* **thinking_mode** — Enable extended reasoning (Ollama/Llama.cpp)
* **model_mappings** — Map requested model names to provider-specific models
* **provider_priority** — Order of provider selection
* **fallback_enabled** — Auto-fallback to next provider on failure
* **cost_optimization** — Prefer cheaper providers when possible

## CLI Usage

```sh
python run_cli.py [options]
```

**Options:**

* `--port PORT` — Server port (default: 8080)
* `--host HOST` — Bind address (default: 127.0.0.1)
* `--tunnel` — Enable localhost.run tunnel
* `--no-tunnel` — Disable tunnel
* `--ollama URL` — Override Ollama endpoint URL
* `--api-key KEY` — Require API key authentication

**Examples:**

```sh
# Basic usage
python run_cli.py

# Custom port with tunnel
python run_cli.py --port 9000 --tunnel

# With API key protection
python run_cli.py --api-key sk-your-secret-key

# Custom Ollama endpoint
python run_cli.py --ollama http://192.168.1.100:11434
```

## Model Mapping

Linx allows you to map common model names (like `gpt-4o`) to your preferred local or remote models:

```json
"model_mappings": {
  "gpt-4o": "qwen2.5-coder:32b",
  "gpt-4": "llama3.1:70b",
  "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet"
}
```

**How it works:**
1. Client requests `gpt-4o`
2. Linx checks mappings for each provider
3. Routes to first available provider with that mapping
4. Falls back to next provider if primary fails

**Benefits:**
* Use familiar model names across providers
* Seamless switching between local and remote models
* Easy A/B testing of different models

## Build Executables

### Windows

```sh
pyinstaller --name Linx-CLI --onefile --console --icon=icon.ico --add-data "config.json;." run_cli.py
```

### macOS

```sh
python setup.py py2app --cli
```

## Architecture

Linx acts as an intelligent proxy between AI clients and model providers:

```
┌─────────────┐
│ AI Client  │ (Cursor, Continue, Kilocode, Custom App)
│ (OpenAI    │
│  API)      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         Linx Router                 │
│  ┌──────────────────────────────┐   │
│  │  Smart Routing & Fallback    │   │
│  │  Model Mapping & Translation │   │
│  │  Health Checks & Monitoring  │   │
│  └──────────────────────────────┘   │
└───┬─────────┬─────────┬─────────────┘
    │         │         │
    ▼         ▼         ▼
┌────────┐ ┌──────┐ ┌──────────┐
│ Ollama │ │Llama │ │OpenRouter│
│ Local  │ │.cpp  │ │ Remote   │
└────────┘ └──────┘ └──────────┘
```

## Recent Updates

### v0.1.0 - Complete Rebrand

* **Renamed:** OllamaLink → **Linx**
* **Multi-Provider:** Added Llama.cpp support alongside Ollama
* **Enhanced Routing:** Smart provider selection with health monitoring
* **OpenAI Compatible:** Full `/v1` API compliance
* **Streaming:** Proper SSE streaming for all providers
* **Tunnel Support:** localhost.run integration for remote access
* **Code Optimization:** Cleaner architecture, removed global variables
* **GUI Development:** Electron-based interface in progress

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see [license.md](license.md) for details
