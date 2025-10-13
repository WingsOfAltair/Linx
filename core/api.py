            # Use router's result directly instead of separate handlers
            if route_result.get("stream"):
                # Router returned streaming result
                async def wrapped_stream():
                    response_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"
                    logger.info(f"Starting stream with ID: {response_id}")
                    
                    try:
                        async for chunk in route_result["stream_generator"]:
                            if isinstance(chunk, dict):
                                if "error" in chunk:
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                                
                                if "message" in chunk:
                                    msg = chunk["message"]
                                    # Get content or thinking mode message
                                    content = ""
                                    if isinstance(msg, dict):
                                        content = msg.get("content", "") or msg.get("thinking", "")
                                    
                                    # Format as OpenAI-compatible chunk
                                    response = {
                                        "id": response_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": display_model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": content},
                                            "finish_reason": "stop" if chunk.get("done") else None
                                        }]
                                    }
                                    
                                    yield f"data: {json.dumps(response)}\n\n"
                                    
                                    if chunk.get("done"):
                                        yield "data: [DONE]\n\n"
                                        logger.info("Stream completed")
                                    
                    except Exception as e:
                        logger.error(f"Stream error: {str(e)}")
                        error_json = json.dumps({"error": {"message": str(e), "type": "stream_error"}})
                        yield f"data: {error_json}\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    content=wrapped_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )