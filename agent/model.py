"""
Deep-Think-Agent 模型调用模块

封装智谱 GLM 模型的异步调用，支持 JSON 输出提示与多模态消息。
"""

import asyncio
import copy
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

MODEL_NAME = "glm-4.6v-flashx"
JSON_INSTRUCTION = "只输出严格 JSON，不要有其他文字"
MAX_ATTEMPTS = 3

client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))


def _is_image_part(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "image_url"
        and isinstance(item.get("image_url"), dict)
        and item["image_url"].get("url")
    )


def _to_text_part(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """将含图片的消息转为 GLM 多模态 content 列表格式。"""
    msg = copy.deepcopy(message)
    content = msg.get("content")

    if _is_image_part(content):
        msg["content"] = [content]
        return msg

    if isinstance(content, list):
        parts: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append(_to_text_part(item))
            elif _is_image_part(item):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(_to_text_part(str(item["text"])))
            else:
                parts.append(_to_text_part(str(item)))

        if any(_is_image_part(part) for part in parts):
            msg["content"] = parts
        elif len(parts) == 1 and parts[0].get("type") == "text":
            msg["content"] = parts[0]["text"]
        else:
            msg["content"] = parts
        return msg

    return msg


def _append_json_instruction(messages: List[Dict[str, Any]]) -> None:
    """在 system prompt 后追加 JSON 输出约束。"""
    for msg in messages:
        if msg.get("role") != "system":
            continue

        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = f"{content}\n{JSON_INSTRUCTION}"
        elif isinstance(content, list):
            content.append(_to_text_part(JSON_INSTRUCTION))
        else:
            msg["content"] = JSON_INSTRUCTION
        return

    messages.insert(0, {"role": "system", "content": JSON_INSTRUCTION})


def _prepare_messages(
    messages: List[Dict[str, Any]],
    response_format: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prepared = [_normalize_message(message) for message in messages]
    if response_format == "json":
        _append_json_instruction(prepared)
    return prepared


def _call_glm_sync(
    messages: List[Dict[str, Any]],
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content or ""


async def call_glm(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    response_format: Optional[str] = None,
) -> str:
    """调用 GLM 模型并返回文本结果。

    Args:
        messages: OpenAI 风格消息列表，content 可含多模态片段。
        temperature: 采样温度，默认 0.2。
        response_format: 为 ``"json"`` 时在 system prompt 追加 JSON 约束。

    Returns:
        模型生成的文本内容。
    """
    prepared = _prepare_messages(messages, response_format)
    last_error: Optional[Exception] = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            return await asyncio.to_thread(
                _call_glm_sync,
                prepared,
                temperature,
            )
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt)

    assert last_error is not None
    raise last_error


if __name__ == "__main__":
    test_messages: List[Dict[str, Union[str, List[Dict[str, Any]]]]] = [
        {"role": "user", "content": "用一句话介绍 Python 是什么。"},
    ]

    result = asyncio.run(call_glm(test_messages))
    print("模型返回：")
    print(result)
