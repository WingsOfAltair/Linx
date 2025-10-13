    async def stream_chat_completion(self, model: str, messages: List[Dict[str, Any]],
                                   temperature: float = 0.7, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion from Ollama."""
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens:
            request_data["max_tokens"] = max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/api/chat",
                    json=request_data
                ) as response:
                    if response.status_code != 200:
                        yield {
                            "error": {"message": f"Ollama returned status: {response.status_code}", "code": response.status_code}
                        }
                        return
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON: {line[:100]}... Error: {str(e)}")
                                continue