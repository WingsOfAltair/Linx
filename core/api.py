import argparse
import asyncio
import json
import logging
import sys
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterable, Callable, Dict, Optional, Union, MutableMapping, Awaitable

from .handlers import (
    LlamaCppRequestHandler, LlamaCppResponseHandler,
    OllamaRequestHandler, OllamaResponseHandler,
    OpenRouterRequestHandler, OpenRouterResponseHandler,
)
from .router import Router
from .util import load_config, start_localhost_run_tunnel

# Constants
DEFAULT_VERIFICATION_PROMPT_TOKENS = 10
DEFAULT_VERIFICATION_COMPLETION_TOKENS = 8
CURSOR_VERIFICATION_KEYWORDS = ["test", "hello", "hi", "ping", "verify", "check", "connection"]
MAX_VERIFICATION_MESSAGE_LENGTH = 20

logger = logging.getLogger(__name__)
 
tunnel_process: Optional[Any] = None
tunnel_url: Optional[str] = None
tunnel_port: Optional[int] = None
        
def create_api(
    ollama_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    request_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> FastAPI:
    """Create a new FastAPI instance with all routes configured"""
    app = FastAPI(title="OllamaLink")

    # Initialize components with proper error handling
    router = None
    config = None
    request_handlers: Dict[str, Optional[Union[OllamaRequestHandler, OpenRouterRequestHandler, LlamaCppRequestHandler]]] = {
        'ollama': None,
        'openrouter': None,
        'llamacpp': None
    }
    response_handlers: Dict[str, Optional[Union[OllamaResponseHandler, OpenRouterResponseHandler, LlamaCppResponseHandler]]] = {
        'ollama': None,
        'openrouter': None,
        'llamacpp': None
    }
    
    try:
        router = Router(ollama_endpoint=ollama_endpoint or "http://localhost:11434", config_path=str(Path("config.json")))
        logger.info("Router initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize router: {str(e)}")
        logger.error("Server will start with limited functionality")
    
    try:
        config = load_config(Path("config.json"))
        
        # Initialize Ollama handlers
        ollama_endpoint_url = ollama_endpoint or config.get("ollama", {}).get("endpoint", "http://localhost:11434")
        request_handlers['ollama'] = OllamaRequestHandler(endpoint=ollama_endpoint_url)
        response_handlers['ollama'] = OllamaResponseHandler()
        
        # Initialize OpenRouter handlers
        openrouter_config = config.get("openrouter", {})
        openrouter_api_key = openrouter_config.get("api_key", "")
        if openrouter_api_key:
            request_handlers['openrouter'] = OpenRouterRequestHandler(
                endpoint=openrouter_config.get("endpoint", "https://openrouter.ai/api/v1"),
                api_key=openrouter_api_key
            )
            response_handlers['openrouter'] = OpenRouterResponseHandler()
        
        # Initialize LlamaCpp handlers
        llamacpp_config = config.get("llamacpp", {})
        request_handlers['llamacpp'] = LlamaCppRequestHandler(
            endpoint=llamacpp_config.get("endpoint", "http://localhost:8080")
        )
        response_handlers['llamacpp'] = LlamaCppResponseHandler()
        
        logger.info("All handlers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize handlers: {str(e)}")
        logger.error("Server will start with limited functionality")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if api_key:
        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            if request.url.path == "/":
                return await call_next(request)
                
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Missing API key", "code": "missing_api_key"}}
                )
                
            try:
                token_type, token = auth_header.split()
                if token_type.lower() != "bearer":
                    raise ValueError("Invalid token type")
            except (ValueError, IndexError):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid Authorization header", "code": "invalid_auth_header"}}
                )
                
            if token != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "code": "invalid_api_key"}}
                )
                
            return await call_next(request)

    if request_callback:
        @app.middleware("http")
        async def request_tracking_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            if not request.url.path.startswith("/v1/chat/completions"):
                return await call_next(request)
                
            body = await request.body()
            request_data = None
            
            if body:
                try:
                    request_data = json.loads(body)
                    request_callback({
                        "type": "request",
                        "request": request_data
                    })
                except (json.JSONDecodeError, ValueError):
                    pass
                    
            class ResponseInterceptor(StreamingResponse):
                
                def __init__(self, content: AsyncIterable[Union[str, bytes]], **kwargs: Any) -> None:
                    self.is_streaming: bool = bool(kwargs.pop("is_streaming", False))
                    self.original_content: AsyncIterable[Union[str, bytes]] = content
                    self.chunk_count: int = 0
                    self.start_time: float = time.time()
                    
                    if self.is_streaming:
                        async def wrapped_content() -> AsyncGenerator[str, None]:
                            if request_callback is not None and request_data is not None:
                                request_callback({
                                    "type": "stream_start",
                                    "request": request_data
                                })
                                
                            try:
                                async for chunk in self.original_content:
                                    self.chunk_count += 1
                                    # Normalize chunk to string
                                    if isinstance(chunk, (bytes, bytearray)):
                                        chunk_str = bytes(chunk).decode("utf-8", errors="ignore")
                                    else:
                                        chunk_str = str(chunk)
                                    
                                    if self.chunk_count % 10 == 0:
                                        if request_callback is not None:
                                            request_callback({
                                                "type": "stream_chunk",
                                                "request": request_data,
                                                "chunk_count": self.chunk_count,
                                                "elapsed": time.time() - self.start_time
                                            })
                                    
                                    yield chunk_str
                                    
                                if request_callback is not None and request_data is not None:
                                    request_callback({
                                        "type": "stream_end",
                                        "request": request_data,
                                        "chunk_count": self.chunk_count,
                                        "elapsed": time.time() - self.start_time
                                    })
                                    
                            except Exception as e:
                                logger.error(f"Error in stream processing: {str(e)}")
                                if request_callback is not None and request_data is not None:
                                    request_callback({
                                        "type": "error",
                                        "request": request_data,
                                        "error": str(e)
                                    })
                                    
                        content = wrapped_content()
                        
                    super().__init__(content, **kwargs)
                    
            async def _intercept_response(response_body: str) -> str:
                if request_callback is not None and request_data is not None:
                    try:
                        if response_body.strip().startswith("data:"):
                            request_callback({
                                "type": "response",
                                "request": request_data,
                                "response": {"message": "SSE streaming response"}
                            })
                        else:
                            response_body = response_body.strip()
                            if response_body:
                                response_data = json.loads(response_body)
                                request_callback({
                                    "type": "response",
                                    "request": request_data,
                                    "response": response_data
                                })
                            else:
                                request_callback({
                                    "type": "error",
                                    "request": request_data,
                                    "error": "Empty response received"
                                })
                    except json.JSONDecodeError as json_error:
                        logger.error(f"Error parsing JSON response: {json_error} - Response body: '{response_body[:100]}...'")
                        request_callback({
                            "type": "error",
                            "request": request_data,
                            "error": f"Invalid JSON in response: {str(json_error)}"
                        })
                    except Exception as e:
                        logger.error(f"Error tracking response: {str(e)}")
                        
                return response_body
                
            modified_request = Request(
                scope=request.scope,
                receive=request._receive
            )
            
            async def modified_receive() -> MutableMapping[str, Any]:
                data: MutableMapping[str, Any] = await request._receive()
                if data["type"] == "http.request":
                    data["body"] = body
                return data
                
            modified_request._receive = modified_receive
            
            try:
                response = await call_next(modified_request)
                
                if isinstance(response, StreamingResponse):
                    is_streaming = True
                    
                    interceptor = ResponseInterceptor(
                        response.body_iterator,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type,
                        background=response.background,
                        is_streaming=is_streaming
                    )
                    
                    return interceptor
                    
                else:
                    body = b""
                    async for chunk in response.body_iterator:
                        body += chunk
                        
                    try:
                        body_str = body.decode('utf-8')
                        processed_body = await _intercept_response(body_str)
                        body = processed_body.encode('utf-8')
                    except Exception as e:
                        logger.error(f"Error processing response: {str(e)}")
                        
                    return Response(
                        content=body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type,
                        background=response.background
                    )
            except Exception as e:
                logger.error(f"Error in request processing: {str(e)}")
                if request_callback is not None and request_data is not None:
                    request_callback({
                        "type": "error",
                        "request": request_data,
                        "error": str(e)
                    })
                raise

    @app.get("/v1")
    async def api_info() -> Response:
        """API root - provides basic info"""
        return JSONResponse(content={
            "info": "OllamaLink API Bridge",
            "ollama_endpoint": router.ollama_endpoint if router is not None else None,
            "version": "0.1.0"
        })

    @app.get("/v1/models", response_model=None)
    async def list_models() -> Any:
        """List available models"""
        try:
            if router is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                )
            models = await router.get_available_models()
            return JSONResponse(content={"data": models, "object": "list"})
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "internal_error"}}
            )

    # Proxy Ollama endpoints for compatibility (e.g., Kilocode)
    @app.get("/v1/api/tags", response_model=None)
    @app.get("/api/tags", response_model=None)  # Alias without /v1 prefix
    async def proxy_ollama_tags() -> Response:
        try:
            if router is None or not getattr(router, "ollama_endpoint", None):
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Ollama endpoint not configured", "code": "service_unavailable"}}
                )
            url = f"{router.ollama_endpoint}/api/tags"
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return JSONResponse(content=resp.json(), status_code=200)
                else:
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": f"Upstream returned {resp.status_code}", "code": resp.status_code}}
                    )
        except Exception as e:
            logger.error(f"Error proxying Ollama tags: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "proxy_error"}}
            )
    
    async def _proxy_ollama_request(request: Request, endpoint: str, stream: bool = False, raw_passthrough: bool = False) -> Response:
        """Generic handler to proxy requests to Ollama endpoints"""
        if router is None or not getattr(router, "ollama_endpoint", None):
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Ollama endpoint not configured", "code": "service_unavailable"}}
            )

        try:
            body = await request.json()
            url = f"{router.ollama_endpoint}{endpoint}"
            logger.info(f"Proxying to {url} (stream={stream})")
            logger.debug(f"Request body: {json.dumps(body)[:200]}...")
            
            # Configure generous timeouts for streaming responses
            timeout = httpx.Timeout(
                timeout=300.0,  # 5 minutes total timeout
                read=None,    # No read timeout for streaming
                write=30.0,   # 30 seconds for sending request
                connect=30.0  # 30 seconds for connection
            )
            
            # Configure connection limits
            limits = httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            )
            if stream:
                response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
                last_heartbeat = time.time()
                
                async def generate_stream() -> AsyncGenerator[str, None]:
                    nonlocal last_heartbeat
                    client: Optional[httpx.AsyncClient] = None
                    response = None
                    
                    try:
                        # For raw passthrough (Ollama NDJSON), do not send an initial SSE chunk
                        if not raw_passthrough:
                            # Send initial OpenAI-style SSE chunk
                            initial_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": body.get("model", "unknown"),
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }]
                        }
                        if not raw_passthrough:
                            yield f"data: {json.dumps(initial_chunk)}\n\n"
                        
                        client = httpx.AsyncClient(timeout=timeout, limits=limits)
                        async with client.stream('POST', url, json=body) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                error_msg = f"Upstream returned {response.status_code}: {error_text.decode()}"
                                if raw_passthrough:
                                    # Emit raw JSON error line
                                    yield json.dumps({"error": {"message": error_msg, "code": response.status_code}}) + "\n"
                                else:
                                    error_data = json.dumps({"error": {"message": error_msg, "code": response.status_code}})
                                    yield f"data: {error_data}\n\n"
                                    yield "data: [DONE]\n\n"
                                return

                            async for line in response.aiter_lines():
                                # Send periodic heartbeat for SSE only
                                if not raw_passthrough and time.time() - last_heartbeat >= 5.0:
                                    try:
                                        yield ": ping\n\n"
                                        last_heartbeat = time.time()
                                    except GeneratorExit:
                                        logger.info("Client disconnected during heartbeat; stopping stream")
                                        return
                                
                                if not line or not line.strip():
                                    await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                                    continue

                                if raw_passthrough:
                                    # Forward Ollama NDJSON exactly as-is
                                    try:
                                        # Validate it's JSON; if not, pass through anyway
                                        _ = json.loads(line)
                                    except json.JSONDecodeError:
                                        pass
                                    yield line + "\n"
                                    continue

                                try:
                                    data = json.loads(line)
                                    if "error" in data:
                                        logger.error(f"Error from Ollama: {data['error']}")
                                        error_data = json.dumps({"error": {"message": str(data['error']), "code": "model_error"}})
                                        yield f"data: {error_data}\n\n"
                                        yield "data: [DONE]\n\n"
                                        return

                                    content = ""
                                    if isinstance(data.get("message"), dict):
                                        msg = data["message"]
                                        content = msg.get("content", "") or msg.get("thinking", "")
                                    elif isinstance(data.get("message"), str):
                                        content = data["message"]
                                    else:
                                        content = data.get("content", "")

                                    if content:
                                        chunk = {
                                            "id": response_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": body.get("model", "unknown"),
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": content},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk)}\n\n"

                                    if data.get("done", False):
                                        yield "data: [DONE]\n\n"
                                        return

                                except json.JSONDecodeError as e:
                                    logger.warning(f"Failed to parse JSON: {e}")
                                    continue

                            # Send final DONE if we exit the loop
                            if not raw_passthrough:
                                yield "data: [DONE]\n\n"

                    except GeneratorExit:
                        logger.info("Client disconnected, cleaning up stream without error")
                        return
                    except Exception as e:
                        logger.error(f"Stream error: {str(e)}")
                        try:
                            error_data = json.dumps({"error": {"message": str(e), "code": "stream_error"}})
                            yield f"data: {error_data}\n\n"
                            yield "data: [DONE]\n\n"
                        except GeneratorExit:
                            logger.info("Client disconnected during error response; stopping stream")
                            return
                    finally:
                        # Clean up resources
                        if client:
                            await client.aclose()

                return StreamingResponse(
                    content=generate_stream(),
                    media_type=("application/x-ndjson" if raw_passthrough else "text/event-stream"),
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )

            # Handle regular non-streaming response
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body)
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_msg = f"Upstream returned {resp.status_code}: {error_text.decode()}"
                    logger.error(f"Error from {url}: {error_msg}")
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": error_msg, "code": resp.status_code}}
                    )
                
                response_json = resp.json()
                logger.debug(f"Success response from {url}: {json.dumps(response_json)[:200]}...")
                return JSONResponse(content=response_json, status_code=200)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error on {endpoint}: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON in request body", "code": "invalid_json"}}
            )
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout while proxying to {endpoint}: {type(e).__name__}: {str(e)}")
            return JSONResponse(
                status_code=504,  # Gateway Timeout
                content={"error": {"message": "Request timed out waiting for model response", "code": "timeout"}}
            )
        except Exception as e:
            logger.error(f"Error proxying to {endpoint}: {type(e).__name__}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"{type(e).__name__}: {str(e)}", "code": "proxy_error"}}
            )

    
    @app.post("/v1/api/show", response_model=None)
    @app.post("/api/show", response_model=None)  # Alias without /v1 prefix
    async def proxy_ollama_show(request: Request) -> Response:
        """Proxy Ollama's model show endpoint"""
        return await _proxy_ollama_request(request, "/api/show")

    @app.post("/v1/api/chat", response_model=None)
    @app.post("/api/chat", response_model=None)
    async def proxy_ollama_chat(request: Request) -> Response:
        """Proxy Ollama's chat endpoint (Ollama native NDJSON)"""
        return await _proxy_ollama_request(request, "/api/chat", stream=True, raw_passthrough=True)

    # OpenAI-compatible Chat Completions endpoint
    @app.post("/v1/chat/completions", response_model=None)
    async def openai_chat_completions(request: Request) -> Response:
        """OpenAI-compatible chat completions that maps to Ollama chat with streaming (SSE)."""
        return await _proxy_ollama_request(request, "/api/chat", stream=True, raw_passthrough=False)

    @app.post("/v1/api/generate", response_model=None)
    @app.post("/api/generate", response_model=None)
    async def proxy_ollama_generate(request: Request) -> Response:
        """Proxy Ollama's generate endpoint (Ollama native NDJSON)"""
        return await _proxy_ollama_request(request, "/api/generate", stream=True, raw_passthrough=True)

    @app.post("/v1/api/embeddings", response_model=None)
    @app.post("/api/embeddings", response_model=None)
    async def proxy_ollama_embeddings(request: Request) -> Response:
        """Proxy Ollama's embeddings endpoint"""
        return await _proxy_ollama_request(request, "/api/embeddings")

    @app.post("/v1/api/pull", response_model=None)
    @app.post("/api/pull", response_model=None)
    async def proxy_ollama_pull(request: Request) -> Response:
        """Proxy Ollama's model pull endpoint"""
        return await _proxy_ollama_request(request, "/api/pull", stream=True)

    @app.post("/v1/api/copy", response_model=None)
    @app.post("/api/copy", response_model=None)
    async def proxy_ollama_copy(request: Request) -> Response:
        """Proxy Ollama's model copy endpoint"""
        return await _proxy_ollama_request(request, "/api/copy")

    @app.post("/v1/api/delete", response_model=None)
    @app.post("/api/delete", response_model=None)
    async def proxy_ollama_delete(request: Request) -> Response:
        """Proxy Ollama's model delete endpoint"""
        return await _proxy_ollama_request(request, "/api/delete")
    
    @app.get("/v1/providers/status", response_model=None)
    async def provider_status() -> Response:
        """Get status of all providers"""
        try:
            if router is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                )
            status = router.get_provider_status()
            return status
        except Exception as e:
            logger.error(f"Error getting provider status: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "internal_error"}}
            )
    
    @app.get("/api/providers/status", response_model=None)
    async def gui_provider_status() -> Response:
        """Get provider status formatted for GUI"""
        try:
            if router is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                )
            
            status = router.get_provider_status()
            
            # Format for GUI consumption
            gui_status = {
                "providers": {
                    "ollama": {
                        "name": "Ollama",
                        "enabled": status["ollama"]["enabled"],
                        "healthy": status["ollama"]["healthy"],
                        "models": status["ollama"]["models"],
                        "endpoint": status["ollama"]["endpoint"],
                        "status": "connected" if status["ollama"]["healthy"] else "disconnected",
                        "error": None if status["ollama"]["healthy"] else "Connection failed"
                    },
                    "openrouter": {
                        "name": "OpenRouter",
                        "enabled": status["openrouter"]["enabled"],
                        "healthy": status["openrouter"]["healthy"],
                        "models": status["openrouter"]["models"],
                        "endpoint": status["openrouter"]["endpoint"],
                        "status": "connected" if status["openrouter"]["healthy"] else "disconnected",
                        "error": None if status["openrouter"]["healthy"] else "API key or connection issue"
                    },
                    "llamacpp": {
                        "name": "Llama.cpp",
                        "enabled": status["llamacpp"]["enabled"],
                        "healthy": status["llamacpp"]["healthy"],
                        "models": status["llamacpp"]["models"],
                        "endpoint": status["llamacpp"]["endpoint"],
                        "status": "connected" if status["llamacpp"]["healthy"] else "disconnected",
                        "error": None if status["llamacpp"]["healthy"] else "Server not available"
                    }
                },
                "routing": status["routing"],
                "timestamp": int(time.time())
            }
            
            return gui_status
            
        except Exception as e:
            logger.error(f"Error getting GUI provider status: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "internal_error"}}
            )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Response:
        """Handle chat completions"""
        try:
            body = await request.json()
            logger.info(f"Received chat completion request: {json.dumps(body, indent=2)}")
            
            user_agent = request.headers.get("User-Agent", "")
            is_cursor = "Cursor" in user_agent
            logger.info(f"User-Agent: {user_agent}")
            
            # Detect Cursor verification requests
            model = body.get("model")
            messages = body.get("messages", [])
            is_cursor_verification = False
            
            if is_cursor and model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"] and len(messages) == 1:
                # Check if it's a short test message (typical verification)
                user_message = messages[0].get("content", "").lower().strip()
                if any(keyword in user_message for keyword in CURSOR_VERIFICATION_KEYWORDS) or len(user_message) < MAX_VERIFICATION_MESSAGE_LENGTH:
                    is_cursor_verification = True
                    logger.info(f"Detected Cursor verification request for {model}: '{user_message}'")
            
            # Don't force streaming for verification requests
            if is_cursor and not is_cursor_verification and not body.get("stream", False):
                logger.info("Forcing streaming mode for Cursor client")
                body["stream"] = True
                
            stream = body.get("stream", False)
            temperature = body.get("temperature", 0.7)
            provider = body.get("provider", None) 
            
            max_tokens = body.get("max_tokens", None)
            if max_tokens is None:
                max_tokens = body.get("max_new_tokens", None)
                if max_tokens is None:
                    max_tokens = body.get("maxOutputTokens", None)
                    
            if max_tokens is not None:
                max_tokens = int(max_tokens)
            
            # Handle Cursor verification requests with direct response
            if is_cursor_verification:
                logger.info("Responding to Cursor verification with direct response")
                verification_response = {
                    "id": f"chatcmpl-{str(uuid.uuid4())[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Connection verified. OllamaLink is ready!"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": DEFAULT_VERIFICATION_PROMPT_TOKENS,
                        "completion_tokens": DEFAULT_VERIFICATION_COMPLETION_TOKENS,
                        "total_tokens": DEFAULT_VERIFICATION_PROMPT_TOKENS + DEFAULT_VERIFICATION_COMPLETION_TOKENS
                    },
                    "system_fingerprint": "ollamalink-server"
                }
                
                if stream:
                    # Return streaming response for verification
                    async def generate_verification_stream() -> AsyncGenerator[str, None]:
                        chunk = {
                            "id": verification_response["id"],
                            "object": "chat.completion.chunk",
                            "created": verification_response["created"],
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "Connection verified. OllamaLink is ready!"},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    
                    return StreamingResponse(
                        generate_verification_stream(),
                        media_type="text/event-stream"
                    )
                else:
                    # Return proper OpenAI format for non-streaming
                    return JSONResponse(
                        content=verification_response,
                        status_code=200
                    )
            
            # Check if explicit provider selection is requested
            if provider:
                # Use explicit provider selection
                if router is None:
                    return JSONResponse(
                        status_code=503,
                        content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                    )
                
                route_result = await router.make_request_with_provider(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            else:
                if router is None:
                    return JSONResponse(
                        status_code=503,
                        content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                    )
                
                route_result = await router.make_request(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            
            # Check if router returned an error that should be passed through
            if route_result.get("error"):
                return JSONResponse(
                    status_code=route_result["error"].get("code", 500),
                    content={"error": route_result["error"]}
                )
            
            provider = route_result.get("provider")
            display_model = route_result.get("display_model", model)
            
            if route_result.get("fallback"):
                logger.info(f"Using fallback provider: {provider}")
            
            # Use router's result directly instead of separate handlers
            if route_result.get("stream"):
                # Router returned streaming result
                response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
                logger.info(f"Starting stream with ID: {response_id}")
                
                async def wrapped_stream() -> AsyncGenerator[str, None]:
                    try:
                        chunk: Optional[Any] = None
                        got_provider_chunk = False
                        sent_initial_tick = False
                        
                        # Send a tiny initial tick to ensure clients recognize an active stream
                        initial_tick = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": display_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }]
                        }
                        logger.info("Streaming: sending initial tick")
                        yield f"data: {json.dumps(initial_tick)}\n\n"
                        sent_initial_tick = True
                        
                        async for chunk in route_result["stream_generator"]:
                            got_provider_chunk = True
                            
                            # If router already produced SSE strings, pass-through directly
                            if isinstance(chunk, str):
                                # Ensure proper line termination
                                if not chunk.endswith("\n\n"):
                                    chunk = chunk + "\n\n"
                                logger.info("Streaming: pass-through SSE line from router")
                                yield chunk
                                # If this is the [DONE] line, stop
                                if chunk.strip() == "data: [DONE]":
                                    logger.info("Streaming: received [DONE] from router")
                                    return
                                continue
                            
                            # Otherwise, handle dict chunks and format as OpenAI-compatible SSE
                            if isinstance(chunk, dict):
                                if "error" in chunk:
                                    logger.error(f"Error from provider: {chunk['error']}")
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                                
                                # Extract message content or thinking text
                                content = ""
                                msg = chunk.get("message")
                                if isinstance(msg, dict):
                                    content = msg.get("content", "") or msg.get("thinking", "")
                                elif isinstance(msg, str):
                                    content = msg
                                else:
                                    # Try direct content access if message structure is different
                                    content = chunk.get("content", "")
                                
                                # Some providers may send keep-alive chunks; tolerate empty content
                                # Check if we need to include role in delta (first chunk should have role)
                                delta = {"content": content}
                                if not sent_initial_tick and msg:
                                    delta["role"] = "assistant"
                                
                                is_done = bool(chunk.get("done"))
                                if is_done and (not content or content.strip() == ""):
                                    # Finalizer with no content: just send [DONE]
                                    yield "data: [DONE]\n\n"
                                    return
                                
                                response = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": display_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": "stop" if is_done else None
                                    }]
                                }
                                
                                yield f"data: {json.dumps(response)}\n\n"
                                
                                if is_done:
                                    yield "data: [DONE]\n\n"
                                    return
                        
                        # If we get here with no provider chunks at all, send an error
                        if not got_provider_chunk:
                            error_json = json.dumps({"error": {"message": "Provider did not send back a response", "type": "empty_response"}})
                            yield f"data: {error_json}\n\n"
                            yield "data: [DONE]\n\n"
                            
                    except Exception as e:
                        logger.error(f"Stream error: {str(e)}")
                        error_json = json.dumps({"error": {"message": str(e), "type": "stream_error"}})
                        yield f"data: {error_json}\n\n"
                        yield "data: [DONE]\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    content=wrapped_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Content-Type": "text/event-stream"
                    }
                )
            elif route_result.get("result"):
                # Router returned non-streaming result
                result = route_result["result"]
                if isinstance(result, dict) and "error" in result:
                    return JSONResponse(
                        status_code=result["error"].get("code", 500),
                        content={"error": result["error"]}
                    )
                else:
                    # Format as OpenAI response
                    return JSONResponse(content=result)
            
            # If we reach here, router failed to handle request properly
            logger.error(f"Router failed to handle request for model {model}, provider {provider}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "Internal routing error", "code": "routing_error"}}
            )
                
        except Exception as e:
            logger.error(f"Error processing chat completion request: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(e), "code": "invalid_request"}}
            )
    
    @app.post("/api/tunnel/start", response_model=None)
    async def start_tunnel(request: Request) -> Any:
        """Start a localhost.run tunnel"""
        global tunnel_process, tunnel_url, tunnel_port
        
        try:
            body = await request.json()
            port = body.get("port", 8000)
            
            if tunnel_process is not None:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "Tunnel already running", "code": "tunnel_active"}}
                )
            
            logger.info(f"Starting localhost.run tunnel for port {port}...")
            
            result = await start_localhost_run_tunnel(port)
            
            if not result:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": "Failed to start tunnel", "code": "tunnel_start_failed"}}
                )
            
            tunnel_url, tunnel_process = result
            tunnel_port = port
            
            cursor_url = f"{tunnel_url}/v1"
            
            logger.info(f"Tunnel started successfully: {tunnel_url}")
            
            return JSONResponse(content={
                "success": True,
                "tunnel_url": tunnel_url,
                "cursor_url": cursor_url,
                "port": port,
                "status": "running"
            })
            
        except Exception as e:
            logger.error(f"Error starting tunnel: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "tunnel_error"}}
            )
    
    @app.post("/api/tunnel/stop", response_model=None)
    async def stop_tunnel() -> Any:
        """Stop the localhost.run tunnel"""
        global tunnel_process, tunnel_url, tunnel_port
        
        try:
            if tunnel_process is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "No tunnel running", "code": "no_tunnel"}}
                )
            
            logger.info("Stopping tunnel...")
            
            if hasattr(tunnel_process, 'terminate'):
                tunnel_process.terminate()
            elif hasattr(tunnel_process, 'kill'):
                tunnel_process.kill()
            
            # Reset global state
            tunnel_process = None
            tunnel_url = None
            tunnel_port = None
            
            logger.info("Tunnel stopped successfully")
            
            return JSONResponse(content={
                "success": True,
                "status": "stopped"
            })
            
        except Exception as e:
            logger.error(f"Error stopping tunnel: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "tunnel_stop_error"}}
            )
    
    @app.get("/api/tunnel/status")
    async def get_tunnel_status() -> Response:
        """Get current tunnel status"""
        global tunnel_process, tunnel_url, tunnel_port
        
        is_running = tunnel_process is not None
        cursor_url = f"{tunnel_url}/v1" if tunnel_url else None
        
        return JSONResponse(content={
            "running": is_running,
            "tunnel_url": tunnel_url,
            "cursor_url": cursor_url,
            "port": tunnel_port,
            "status": "running" if is_running else "stopped"
        })

    return app


def setup_logging() -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> None:
    """Main function to run the API server"""
    parser = argparse.ArgumentParser(description='OllamaLink API Server')
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to (default: 8000)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--log-level',
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Log level (default: info)'
    )
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        app = create_api()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
