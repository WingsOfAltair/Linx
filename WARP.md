# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Environment Setup

### Required Dependencies
1. Python environment
2. Install dependencies:
```sh
pip install -e .  # Using pyproject.toml
# or
pip install -r requirements.txt
```
3. Ollama must be running:
```sh
ollama serve
```

### Running the Application
Currently, only the CLI version is functional:
```sh
python run_cli.py  # Basic usage
python run_cli.py --port 8080 --host 127.0.0.1  # With specific host/port
python run_cli.py --direct  # Without tunnel
```

## Project Architecture

### Core Components

1. **Router System (`core/router.py`)**
   - Central component managing requests between different model providers
   - Implements provider selection, fallback logic, and health checks
   - Supports Ollama, OpenRouter, and Llama.cpp integrations

2. **Client Abstractions (`core/clients/`)**
   - `base_client.py`: Abstract base class for provider clients
   - `ollama_client.py`: Local Ollama model integration
   - `openrouter_client.py`: Cloud model provider integration
   - `llamacpp_client.py`: Local Llama.cpp integration

3. **Request/Response Handlers (`core/handlers/`)**
   - Provider-specific request/response formatting
   - Message processing and streaming support
   - Error handling and recovery logic

4. **Configuration System**
   - Uses `config.json` for provider settings, model mappings, and server config
   - Supports dynamic reconfiguration without restart
   - Includes fallback logic and provider priorities

### Key Features

1. **Intelligent Model Routing**
   - Maps client model requests to available providers
   - Implements fallback mechanisms between providers
   - Health monitoring and automatic failover

2. **Streaming Support**
   - Asynchronous streaming for all providers
   - Token-based progress tracking
   - Timeout and error handling

3. **Provider Integration**
   - Unified interface for local and cloud models
   - Provider-specific optimizations
   - Health monitoring and availability checks

## Testing

Use pytest to run tests:
```sh
pytest  # Run all tests
pytest test_specific_file.py  # Run specific test file
pytest -k "test_name"  # Run specific test by name
pytest -v  # Verbose output
```

## Build Process

### macOS:
```sh
python setup.py py2app --cli  # CLI version only
```

### Windows:
```sh
pyinstaller --name OllamaLink-CLI --onefile --console --icon=icon.ico --add-data "config.json;." run_cli.py  # CLI version only
```