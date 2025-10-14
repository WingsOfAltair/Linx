# Linx

**Unify local and remote models into one OpenAI-compatible endpoint**

Linx is a bridge application that connects **local models** (via Ollama) and **remote models** (via OpenRouter.ai or other OpenAI-compatible providers) under a single unified API. It exposes all connected models through an **OpenAI-compatible interface**, allowing seamless use in applications like **Cursor AI**, **VSCode extensions**, or any client supporting the OpenAI API format. Both **CLI** and **GUI** versions exist, with the CLI being fully functional and the GUI in active development.

## Features

* **Unified Endpoint** — Merge local and remote models into one `/v1` API
* **Hybrid Access** — Combine Ollama, OpenRouter, or other APIs
* **OpenAI-Compatible** — Works with any OpenAI-style client
* **Private** — Keep your data local when using Ollama
* **Flexible** — Compatible with multiple AI model providers
* **Routing Logic** — Automatic provider selection and fallback
* **Tunneling** — Public access via `localhost.run`
* **CLI Ready** — Fully functional command-line version
* **GUI (coming soon)** — Visual management dashboard
* **Model Mapping** — Custom model name mapping
* **Secure** — Optional API key protection
* **No Timeout Limits** — Long-running tasks supported

## Quick Start

### 1. Install Ollama

Ensure Ollama is installed and running:

```sh
ollama serve
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run Linx (CLI)

```sh
python run_cli.py
```

*Note: GUI version is in development.*

## Integration

Linx can be integrated with any OpenAI-compatible tool or application. Example use cases:

* **Cursor AI** — override the API URL with Linx’s endpoint
* **VSCode or JetBrains plugins** — connect as OpenAI-compatible API
* **Custom scripts or dashboards** — query Linx directly via REST

## Endpoints

When started, Linx provides:

* **Local URL:** `http://localhost:8080/v1`
* **Tunnel URL:** `https://randomsubdomain.localhost.run/v1`

Use the local URL for same-machine access or the tunnel URL for remote clients.

## Configuration

Example `config.json`:

```json
{
  "ollama": {
    "endpoint": "http://localhost:11434",
    "model_mappings": {
      "gpt-4o": "qwen2.5-coder",
      "gpt-3.5-turbo": "llama3",
      "default": "qwen2.5"
    }
  },
  "openrouter": {
    "enabled": true,
    "api_key": "sk-or-your-api-key",
    "endpoint": "https://openrouter.ai/api/v1",
    "model_mappings": {
      "gpt-4o": "openai/gpt-4o",
      "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet"
    }
  },
  "routing": {
    "provider_priority": ["ollama", "openrouter"],
    "fallback_enabled": true
  },
  "server": {
    "port": 8080,
    "hostname": "127.0.0.1"
  },
  "tunnels": {
    "use_tunnel": true,
    "preferred": "localhost.run"
  }
}
```

## CLI Usage

```sh
python run_cli.py [options]
```

**Options:**

* `--port PORT` — Custom port
* `--host HOST` — Bind host
* `--tunnel` — Enable localhost.run tunnel
* `--no-tunnel` — Disable tunnel
* `--ollama URL` — Override Ollama endpoint

## Model Mapping

Map model names between providers:

```json
"model_mappings": {
  "gpt-4o": "qwen2.5-coder",
  "claude-3.5": "llama3"
}
```

Supports fuzzy matching for variant names.

## Build Executables

### Windows

```sh
pyinstaller --name Linx-CLI --onefile --console --icon=icon.ico --add-data "config.json;." run_cli.py
```

### macOS

```sh
python setup.py py2app --cli
```

## Recent Changes

### Rebrand

* **Renamed:** OllamaLink → **Linx**
* Unified **Ollama**, **OpenRouter**, and **other APIs** into one endpoint
* Fully **OpenAI-compatible** `/v1` API
* Added provider routing, fallback, and health checks
* GUI in development

## License

MIT
