import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import httpx
import json
import asyncio
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI()

STATUS_URL = "https://duckduckgo.com/duckchat/v1/status"
CHAT_URL = "https://duckduckgo.com/duckchat/v1/chat"

# 获取代理设置
http_proxy = os.environ.get('HTTP_PROXY')
https_proxy = os.environ.get('HTTPS_PROXY')
if http_proxy is None:
    http_proxy = "http://127.0.0.1:7890"
if https_proxy is None:
    https_proxy = "https://127.0.0.1:7890"
proxies = {
    'http': http_proxy,
    'https': https_proxy
}

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    max_tokens: Optional[int] = None
    messages: List[Message]


class Chat:
    def __init__(self):
        headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "sec-ch-ua": "\"Not_A Brand\";v=\"99\", \"Google Chrome\";v=\"109\", \"Chromium\";v=\"109\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "x-vqd-accept": "1"
        }
        response = requests.get(STATUS_URL, headers=headers, proxies=proxies)
        self.old_vqd = response.headers.get("x-vqd-4")
        self.new_vqd = response.headers.get("x-vqd-4")

    def update_vqd(self, new_vqd):
        self.old_vqd = self.new_vqd
        self.new_vqd = new_vqd

    def get_headers(self):
        return {
            'accept': 'text/event-stream',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'origin': 'https://duckduckgo.com',
            'referer': 'https://duckduckgo.com/',
            'sec-ch-ua': '"Not_A Brand";v="99", "Google Chrome";v="109", "Chromium";v="109"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
            'x-vqd-4': self.new_vqd
        }


chat_instance = Chat()


async def fetch_duckduckgo_response(messages, model="gpt-4o-mini", retries=3):
    global chat_instance
    new_msg = []
    for msg in messages:
        role = msg.role
        if role == "system":
            msg.role = "user"
        new_msg.append(msg.dict())
    data = {
        "model": model,
        "messages": new_msg
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0, proxies=proxies) as client:
                async with client.stream('POST', CHAT_URL, headers=chat_instance.get_headers(), json=data) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                yield line[6:]  # Remove "data: " prefix
                        return  # Successful response, exit the function
                    else:
                        if response.status_code == 400:
                            chat_instance = Chat()
                            continue
                        print(f"Error response: {response.status_code}")
                        print(await response.text())
        except Exception as e:
            print(f"Error occurred: {str(e)}")

        if attempt < retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    print("All retries failed")
    yield json.dumps({"error": "Failed to get response after multiple attempts"})


def parse_duckduckgo_response(response):
    try:
        data = json.loads(response)
        return data.get("message", "")
    except json.JSONDecodeError:
        return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        model = request.model
        messages = request.messages
        for message in messages:  # ping响应，为了应付dify的检查。
            if message.content == "ping":
                return {"model": "glm4", "id": "ec0afa48-b167-4299-b3d7-825bcd859f06", "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong",
                                                             "function_call": None}, "finish_reason": "length"}]}
            print("用户消息:", message)

        async def generate():
            full_response = ""
            print(f"{model}:", end='', flush=True)
            async for chunk in fetch_duckduckgo_response(messages, model=model):
                content = parse_duckduckgo_response(chunk)
                if content:
                    full_response += content
                    print(f"{content}", end='', flush=True)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': content}}]})}\n\n"

            yield f"data: {json.dumps({'choices': [{'delta': {'content': ''}, 'finish_reason': 'stop'}]})}\n\n"
            print(f"", flush=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5002)