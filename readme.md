# Linx

**Connect Cursor AI to your local Ollama or OpenRouter.ai models**

Linx is a bridge application that connects **Cursor AI** to your **local Ollama models** or **cloud-based OpenRouter.ai models**. It acts as an **OpenAI-compatible endpoint** and supports **hybrid model routing** between local and remote providers. Currently, only the **CLI version** is fully functional — the **GUI** is in active development.

## Features

* **Hybrid Access** — Connect to both local Ollama and OpenRouter models
* **OpenAI-Compatible** — Standard `/v1` API endpoint
* **Private** — Keep your code and data local
* **Flexible** — Works with any Ollama or OpenRouter model
* **Routing Logic** — Automatic provider priority and fallback
* **Tunneling** — Public access via `localhost.run`
* **CLI Ready** — Full feature support
* **GUI (coming soon)** — Model dashboard and monitoring
* **Model Mapping** — Map names like `gpt-4o` → `qwen2.5`
* **Secure** — Optional API key usage
* **No Timeout Limits** — Long-running tasks supported

## Quick Start

### 1. Install Ollama

Make sure Ollama is installed and running:

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

*Note: GUI version is not yet functional.*

## Setting Up Cursor

### 1. Install Cursor

Download from [https://cursor.sh](https://cursor.sh)

### 2. Start Linx

You will see two endpoints:

* **Local URL:** `http://localhost:8080/v1`
* **Tunnel URL:** `https://randomsubdomain.localhost.run/v1`

Use:

* **Local URL** → when Cursor runs on the same machine
* **Tunnel URL** → for remote/cloud access

### 3. Configure in Cursor

1. Open **Settings → Models**
2. In “Override OpenAI Base URL” paste your Linx URL (must end with `/v1`)
3. Enter API key (if configured)
4. Press **Verify**
5. Add mapped models (e.g. `qwen2.5`, `llama3`)

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

## OpenRouter Setup

1. Get an API key at [https://openrouter.ai](https://openrouter.ai)
2. Enable OpenRouter in `config.json`:

   ```json
   "openrouter": {
       "enabled": true,
       "api_key": "sk-or-xxxx",
       "endpoint": "https://openrouter.ai/api/v1"
   }
   ```
3. Run Linx:

   ```sh
   python run_cli.py
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

Map commercial model names to local or remote ones. Example:

```json
"model_mappings": {
  "gpt-4o": "qwen2.5-coder",
  "claude-3.5": "llama3"
}
```

Fuzzy matching resolves variants automatically.

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

* Unified support for **Ollama** and **OpenRouter**
* **OpenAI-compatible API** (`/v1`)
* **Provider routing**, **fallback**, and **health monitoring**
* **GUI** in development

## License

MIT
