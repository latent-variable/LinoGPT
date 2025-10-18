#!/usr/bin/env python3
"""
LM Studio MCP Proxy
Translates OpenAI API requests from Open WebUI to LM Studio's /v1/responses format with MCP tools
Supports both streaming and non-streaming responses
"""
import json
import copy
import os
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
import logging
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# LM Studio endpoint
LM_STUDIO_BASE = "http://192.168.50.142:5002"

# MCP tools configuration

MCP_CONFIG_PATH = Path("/app/mcp.json")


def load_mcp_tools_from_file() -> list[dict]:
    """
    Load MCP tools from LM Studio's mcp.json configuration file.
    Converts LM Studio's format to the /v1/responses API format.
    """
    tools: list[dict] = []

    if not MCP_CONFIG_PATH.exists():
        logger.warning(f"MCP config file not found at {MCP_CONFIG_PATH}")
        return tools

    try:
        with open(MCP_CONFIG_PATH, 'r') as f:
            mcp_config = json.load(f)

        mcp_servers = mcp_config.get("mcpServers", {})

        for server_label, server_config in mcp_servers.items():
            # Skip Docker command-based servers (only support URL-based)
            if "command" in server_config:
                logger.info(f"Skipping command-based MCP server: {server_label}")
                continue

            tool_entry: dict = {
                "type": "mcp",
                "server_label": server_label,
                "allowed_tools": []  # Allow all tools by default
            }

            # Add server URL
            if "url" in server_config:
                tool_entry["server_url"] = server_config["url"]

            # Add headers if present
            if "headers" in server_config and isinstance(server_config["headers"], dict):
                tool_entry["headers"] = server_config["headers"]

            # Add environment variables if needed
            if "env" in server_config and isinstance(server_config["env"], dict):
                # Some MCP servers might need env vars passed as headers
                # We'll skip this for now unless needed
                pass

            tools.append(tool_entry)
            logger.info(f"Loaded MCP server: {server_label} from mcp.json")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse mcp.json: {e}")
    except Exception as e:
        logger.error(f"Error loading MCP tools: {e}")

    return tools


MCP_TOOLS = load_mcp_tools_from_file()


def redact_url(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url)
    if not parts.query:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "***", parts.fragment))


def sanitized_mcp_tools(tools: list[dict]) -> list[dict]:
    safe_tools: list[dict] = []
    for tool in tools:
        sanitized = copy.deepcopy(tool)
        server_url = sanitized.get("server_url")
        if server_url:
            sanitized["server_url"] = redact_url(server_url)
        safe_tools.append(sanitized)
    return safe_tools


@app.route('/v1/models', methods=['GET'])
def list_models():
    """Forward model listing to LM Studio"""
    try:
        response = requests.get(f"{LM_STUDIO_BASE}/v1/models")
        return Response(response.content, status=response.status_code, content_type='application/json')
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Convert OpenAI chat completions request to LM Studio responses format with MCP tools
    Supports both streaming and non-streaming
    """
    try:
        openai_request = request.json

        def summarize_messages(msgs):
            summary = []
            for idx, msg in enumerate(msgs):
                entry = {"role": msg.get("role"), "index": idx}
                content = msg.get("content")
                if isinstance(content, str):
                    entry["content_len"] = len(content)
                elif isinstance(content, list):
                    entry["content_parts"] = [
                        part.get("type", type(part).__name__) for part in content
                    ]
                entry["tool_calls"] = len(msg.get("tool_calls", []) or [])
                if "tool_call_id" in msg:
                    entry["tool_call_id"] = msg["tool_call_id"]
                if "name" in msg:
                    entry["name"] = msg["name"]
                summary.append(entry)
            return summary

        logger.debug(f"Incoming request summary: {summarize_messages(openai_request.get('messages', []))}")
        logger.info(f"Received request from Open WebUI (stream={openai_request.get('stream', False)})")

        # Extract messages and create input text
        messages = openai_request.get("messages", [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        def normalize_content(content):
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    value = item.get("text") if isinstance(item, dict) else item
                    if value:
                        parts.append(str(value))
                return "\n".join(parts)
            return str(content)

        transcript_lines: list[str] = []
        last_user_found = False
        system_prompt = None
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                text = normalize_content(msg.get("content"))
                if text:
                    system_prompt = text if system_prompt is None else f"{system_prompt}\n{text}"
            elif role == "user":
                text = normalize_content(msg.get("content"))
                transcript_lines.append(f"User: {text}")
                last_user_found = True
            elif role == "assistant":
                text = normalize_content(msg.get("content"))
                if text:
                    transcript_lines.append(f"Assistant: {text}")
                for call in msg.get("tool_calls", []):
                    function = call.get("function", {})
                    name = function.get("name", "tool")
                    arguments = function.get("arguments", "")
                    transcript_lines.append(
                        f"Assistant called tool '{name}' with arguments: {arguments}"
                    )
            elif role == "tool":
                name = msg.get("name") or "tool"
                text = normalize_content(msg.get("content"))
                transcript_lines.append(f"Tool '{name}' response: {text}")

        if not last_user_found:
            return jsonify({"error": "No user message found"}), 400

        conversation_text = "\n".join(transcript_lines)
        if system_prompt:
            input_text = f"{system_prompt}\n\n{conversation_text}\nAssistant:"
        else:
            input_text = f"{conversation_text}\nAssistant:"

        # Build LM Studio request
        is_streaming = openai_request.get('stream', False)

        lm_studio_request: dict = {
            "model": openai_request.get("model", "openai/gpt-oss-120b"),
            "input": input_text,
            "max_tokens": openai_request.get("max_tokens", 2000),
            "temperature": openai_request.get("temperature", 0.7),
            "stream": is_streaming,
        }

        # Pass MCP tools to LM Studio
        if MCP_TOOLS:
            lm_studio_request["tools"] = MCP_TOOLS

        logger.info(f"Sending to LM Studio /v1/responses (streaming={is_streaming})")
        logger.debug(f"MCP tools being sent: {sanitized_mcp_tools(MCP_TOOLS)}")

        if is_streaming:
            # Handle streaming response
            return Response(
                stream_with_context(stream_lm_studio_response(lm_studio_request)),
                mimetype='text/event-stream'
            )
        else:
            # Handle non-streaming response
            response = requests.post(
                f"{LM_STUDIO_BASE}/v1/responses",
                json=lm_studio_request,
                headers={"Content-Type": "application/json"},
                timeout=120
            )

            if response.status_code != 200:
                logger.error(f"LM Studio error: {response.text}")
                return jsonify({"error": f"LM Studio returned {response.status_code}: {response.text}"}), response.status_code

            lm_studio_response = response.json()
            openai_response = convert_lm_studio_to_openai(lm_studio_response)
            return jsonify(openai_response)

    except requests.Timeout:
        logger.error("LM Studio request timed out")
        return jsonify({"error": "Request to LM Studio timed out"}), 504
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def stream_lm_studio_response(lm_studio_request):
    """
    Stream LM Studio response and convert to OpenAI streaming format
    """
    try:
        response = requests.post(
            f"{LM_STUDIO_BASE}/v1/responses",
            json=lm_studio_request,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=(10, 300)  # (connect timeout, read timeout) - 5 min for tool execution
        )

        if response.status_code != 200:
            error_data = {
                "id": "error",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": lm_studio_request.get('model'),
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": f"Error: {response.status_code}"},
                    "finish_reason": "error"
                }]
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Track state across chunks
        collected_content = []
        reasoning_chunks = []
        response_id = None
        model = lm_studio_request.get("model")
        reasoning_started = False
        reasoning_sent = False

        current_event = None
        reasoning_ready = False
        active_tool_messages: set[str] = set()
        search_banner_displayed = False
        completed_output_sent = False
        streamed_output_ids: set[str] = set()
        function_calls: dict[str, dict] = {}
        tool_call_order: list[str] = []
        final_chunk_sent = False
        final_finish_reason = "stop"

        def maybe_send_reasoning():
            nonlocal reasoning_sent
            if reasoning_sent or not reasoning_chunks:
                return None
            thinking_text = "".join(reasoning_chunks)
            if not thinking_text.strip():
                reasoning_sent = True
                logger.info("Skipping empty thinking chunk")
                return None
            logger.info(f"Sending thinking chunk ({len(reasoning_chunks)} pieces, {len(thinking_text)} chars)")
            think_chunk = {
                "id": response_id or "chatcmpl-proxy",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": thinking_text,
                        },
                        "finish_reason": None,
                    }
                ],
            }
            reasoning_sent = True
            logger.info("Reasoning sent flag set to True")
            return f"data: {json.dumps(think_chunk)}\n\n"
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # Parse SSE format: "event: <type>" followed by "data: <json>"
            if line.startswith('event: '):
                current_event = line[7:]  # Remove 'event: ' prefix
                continue

            if line.startswith('data: '):
                data_line = line[6:]  # Remove 'data: ' prefix

                if data_line == "[DONE]":
                    # Send final chunk if not already sent
                    if not final_chunk_sent:
                        final_chunk = {
                            "id": response_id or "chatcmpl-proxy",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": final_finish_reason,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        final_chunk_sent = True
                    yield "data: [DONE]\n\n"
                    break

                try:
                    chunk_data = json.loads(data_line)

                    # Extract response ID
                    if 'response' in chunk_data and 'id' in chunk_data['response']:
                        response_id = chunk_data['response']['id']

                    # Capture reasoning/thinking tokens
                    if current_event == "response.reasoning_text.delta":
                        delta = chunk_data.get("delta", "")
                        if delta:
                            reasoning_chunks.append(delta)
                            if not reasoning_started:
                                logger.info("Started collecting reasoning tokens")
                                reasoning_started = True
                    elif current_event in ("response.reasoning_text.done", "response.content_part.done"):
                        reasoning_ready = True

                    # Detect MCP tool calls and show search indicator
                    if current_event == "response.mcp_call.in_progress":
                        item_id = chunk_data.get("item_id")
                        if item_id and item_id in active_tool_messages:
                            pass
                        elif not search_banner_displayed:
                            tool_name = chunk_data.get("name") or chunk_data.get("tool_name") or chunk_data.get("label") or "tool"
                            if item_id:
                                active_tool_messages.add(item_id)
                            search_indicator = {
                                "id": response_id or "chatcmpl-proxy",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": f"\n\n🔍 *Searching with {tool_name}...*\n\n"},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(search_indicator)}\n\n"
                            search_banner_displayed = True

                    if current_event == "response.output_item.added":
                        item = chunk_data.get("item", {})
                        if item.get("type") == "function_call":
                            item_id = item.get("id")
                            if item_id:
                                reasoning_payload = maybe_send_reasoning()
                                if reasoning_payload:
                                    yield reasoning_payload
                                idx = len(tool_call_order)
                                tool_call_order.append(item_id)
                                function_calls[item_id] = {
                                    "id": item.get("call_id") or item_id,
                                    "name": item.get("name"),
                                    "arguments": "",
                                }
                                tool_chunk = {
                                    "id": response_id or "chatcmpl-proxy",
                                    "object": "chat.completion.chunk",
                                    "created": 0,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [
                                                    {
                                                        "index": idx,
                                                        "id": function_calls[item_id]["id"],
                                                        "type": "function",
                                                        "function": {
                                                            "name": function_calls[item_id]["name"] or ""
                                                        },
                                                    }
                                                ]
                                            },
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(tool_chunk)}\n\n"

                    if current_event == "response.function_call_arguments.delta":
                        item_id = chunk_data.get("item_id")
                        delta = chunk_data.get("delta", "")
                        if item_id and item_id in function_calls and delta:
                            reasoning_payload = maybe_send_reasoning()
                            if reasoning_payload:
                                yield reasoning_payload
                            function_calls[item_id]["arguments"] += delta
                            idx = tool_call_order.index(item_id)
                            arg_chunk = {
                                "id": response_id or "chatcmpl-proxy",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": idx,
                                                    "id": function_calls[item_id]["id"],
                                                    "type": "function",
                                                    "function": {
                                                        "arguments": delta
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(arg_chunk)}\n\n"

                    if current_event == "response.function_call_arguments.done":
                        item_id = chunk_data.get("item_id")
                        arguments = chunk_data.get("arguments", "")
                        if item_id and item_id in function_calls:
                            function_calls[item_id]["arguments"] = arguments or function_calls[item_id]["arguments"]

                    # Look for output text delta events (actual content)
                    if current_event == "response.output_text.delta":
                        delta = chunk_data.get("delta", "")
                        if delta:
                            # Send thinking ONCE before first output (if we have any reasoning)
                            reasoning_payload = maybe_send_reasoning()
                            if reasoning_payload:
                                yield reasoning_payload

                            # Send actual output chunk
                            openai_chunk = {
                                "id": response_id or "chatcmpl-proxy",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                            collected_content.append(delta)
                            item_id = chunk_data.get("item_id")
                            if item_id:
                                streamed_output_ids.add(item_id)

                    # Handle completed responses that may contain aggregated output
                    if current_event == "response.completed" and not completed_output_sent:
                        response_payload = chunk_data.get("response", {})
                        output_items = response_payload.get("output", []) or []

                        if output_items:
                            reasoning_payload = maybe_send_reasoning()
                            if reasoning_payload:
                                yield reasoning_payload

                        for item in output_items:
                            item_id = item.get("id")
                            if item_id and item_id in streamed_output_ids:
                                continue
                            if item.get("type") == "message":
                                message_content = item.get("content", [])
                                for content_item in message_content:
                                    if content_item.get("type") == "output_text":
                                        text = content_item.get("text", "")
                                        if text:
                                            openai_chunk = {
                                                "id": response_id or "chatcmpl-proxy",
                                                "object": "chat.completion.chunk",
                                                "created": 0,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"content": text},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                                            collected_content.append(text)
                                if item_id:
                                    streamed_output_ids.add(item_id)
                        completed_output_sent = True

                    if current_event == "response.output_item.done":
                        item = chunk_data.get("item", {})
                        if item.get("type") == "function_call":
                            item_id = item.get("id")
                            if item_id and item_id in function_calls and not final_chunk_sent:
                                reasoning_payload = maybe_send_reasoning()
                                if reasoning_payload:
                                    yield reasoning_payload
                                final_finish_reason = "tool_calls"
                                finish_chunk = {
                                    "id": response_id or "chatcmpl-proxy",
                                    "object": "chat.completion.chunk",
                                    "created": 0,
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "tool_calls",
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(finish_chunk)}\n\n"
                                final_chunk_sent = True

                except json.JSONDecodeError:
                    logger.warning(f"Could not parse data: {data_line}")
                    continue

    except requests.exceptions.ChunkedEncodingError as e:
        logger.error(f"Stream interrupted (likely MCP tool execution): {e}")
        # Stream was interrupted - send what we have and finish gracefully
        if collected_content:
            logger.info(f"Collected {len(collected_content)} content chunks before interruption")
        else:
            # Inform client about interruption if no content was streamed
            interruption_message = {
                "id": response_id or "chatcmpl-proxy",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": lm_studio_request.get('model'),
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n\n⚠️ Search interrupted before results were returned. Please try again or adjust the query.\n\n"},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(interruption_message)}\n\n"

        if not final_chunk_sent:
            final_chunk = {
                "id": response_id or "chatcmpl-proxy",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": lm_studio_request.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": final_finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            final_chunk_sent = True
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error streaming from LM Studio: {e}", exc_info=True)
        error_data = {
            "id": "error",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": lm_studio_request.get('model'),
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": f"Error: {str(e)}"},
                "finish_reason": "error"
            }]
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


def convert_lm_studio_to_openai(lm_response):
    """
    Convert LM Studio /v1/responses format to OpenAI chat completions format
    """
    # Extract the final assistant message from LM Studio output
    output = lm_response.get('output', [])

    # Find the last message item
    content = ""
    for item in output:
        if item.get('type') == 'message' and item.get('role') == 'assistant':
            # Get the text content from the message
            message_content = item.get('content', [])
            for content_item in message_content:
                if content_item.get('type') == 'output_text':
                    content = content_item.get('text', '')
                    break

    # Build OpenAI-compatible response
    openai_response = {
        "id": lm_response.get('id', 'chatcmpl-proxy'),
        "object": "chat.completion",
        "created": lm_response.get('created_at', 0),
        "model": lm_response.get('model', 'openai/gpt-oss-120b'),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": lm_response.get('usage', {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })
    }

    return openai_response


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "lm_studio": LM_STUDIO_BASE})


if __name__ == '__main__':
    logger.info("Starting LM Studio MCP Proxy...")
    logger.info(f"LM Studio endpoint: {LM_STUDIO_BASE}")
    logger.info(f"MCP tools configured: {sanitized_mcp_tools(MCP_TOOLS)}")
    logger.info("Streaming: ENABLED")
    app.run(host='0.0.0.0', port=5003, debug=True)
