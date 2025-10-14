# Kilocode Compatibility Changes

This document outlines the changes made to Linx to ensure compatibility with Kilocode's chat interface.

## Changes Overview

1. Streaming Response Format
2. Error Handling
3. Type Safety Improvements
4. Resource Management

### 1. Streaming Response Format

#### Raw NDJSON for Ollama Endpoints
For Ollama native endpoints (`/api/chat`, `/api/generate`), we now:
- Use `application/x-ndjson` content type
- Pass through raw NDJSON lines from Ollama
- Skip SSE framing for these endpoints
- Preserve Ollama's streaming format

```python
return StreamingResponse(
    content=generate_stream(),
    media_type=("application/x-ndjson" if raw_passthrough else "text/event-stream")
)
```

#### SSE for OpenAI Compatibility
For OpenAI-compatible endpoints (`/v1/chat/completions`), we:
- Use `text/event-stream` content type
- Format responses as OpenAI-style SSE chunks
- Add proper SSE framing (`data: ` prefix)
- Include `[DONE]` markers

### 2. Error Handling

Enhanced error handling for client disconnections:

```python
try:
    async for line in response.aiter_lines():
        if time.time() - last_heartbeat >= 5.0:
            try:
                yield ": ping\\n\\n"
                last_heartbeat = time.time()
            except GeneratorExit:
                logger.info("Client disconnected during heartbeat; stopping stream")
                return

        # ... content processing ...

except GeneratorExit:
    logger.info("Client disconnected, cleaning up stream without error")
    return
finally:
    if client:
        await client.aclose()
```

### 3. Type Safety Improvements

Added proper type hints and fixed type-related issues:

```python
from typing import (
    Any, AsyncGenerator, AsyncIterable, 
    Callable, Dict, Optional, Union, 
    MutableMapping, Awaitable
)

async def api_key_middleware(
    request: Request, 
    call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    ...

class ResponseInterceptor(StreamingResponse):
    def __init__(
        self, 
        content: AsyncIterable[Union[str, bytes]], 
        **kwargs: Any
    ) -> None:
        ...
```

### 4. Resource Management

Improved resource cleanup and connection handling:

```python
# Configure connection limits
limits = httpx.Limits(
    max_keepalive_connections=5,
    max_connections=10,
    keepalive_expiry=30.0
)

# Configure generous timeouts
timeout = httpx.Timeout(
    timeout=300.0,  # 5 minutes total timeout
    read=None,    # No read timeout for streaming
    write=30.0,   # 30 seconds for sending request
    connect=30.0  # 30 seconds for connection
)
```

## Testing

To verify Kilocode compatibility:

1. Configure Kilocode to use your Linx instance:
   - Use `/api/chat` for direct Ollama mode
   - Use `/v1/chat/completions` for OpenAI compatibility mode

2. Test with both streaming modes:
   ```bash
   # Test Ollama native streaming (NDJSON)
   curl -N http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"model":"llama2","messages":[{"role":"user","content":"Hello"}]}'

   # Test OpenAI compatibility (SSE)
   curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"llama2","messages":[{"role":"user","content":"Hello"}]}'
   ```
