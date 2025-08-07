import json
import httpx
import inspect
from functools import partial
import re


async def _initialize_connection(mcp_url: str) -> dict:
    """
    Initialize the MCP server, and return the headers.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "python", "version": "3.12"},
        },
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        r = await client.post(
            mcp_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        r.raise_for_status()
        session_id = r.headers.get("mcp-session-id") or r.headers.get(
            "x-mcp-session-Id"
        )
        # The endpoint streams Server‑Sent Events (SSE).  Each logical reply is on
        # a `data:` line; grab the first JSON payload:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                try:
                    obj = json.loads(line.removeprefix("data:").strip())
                    server_name = obj["result"]["serverInfo"]["name"]
                except Exception:
                    server_name = "unknown_server"

        # strip any non alphanumeric and underscores
        server_name = re.sub(r"[^a-zA-Z0-9_]", "", server_name)
        return session_id, server_name


async def _initialize_notification(mcp_url: str, headers: dict) -> dict:
    """
    Initialize the MCP server.
    """
    payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "application/json, text/event-stream"
    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        r = await client.post(mcp_url, headers=headers, json=payload)
        r.raise_for_status()
        return


async def _discover_tools(mcp_url: str, headers: dict) -> list[dict]:
    """
    Hit `tools/list` and return the decoded tool records.

    Args:
        mcp_url: The URL of the MCP server

    Returns:
        A list of tool records
    """
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    }
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "application/json, text/event-stream"
    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        r = await client.post(
            mcp_url,
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        # The endpoint streams Server‑Sent Events (SSE).  Each logical reply is on
        # a `data:` line; grab the first JSON payload:
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                obj = json.loads(line.removeprefix("data:").strip())
                return obj["result"]["tools"]
        raise RuntimeError("tools/list returned no data lines")


async def get_mcp_tools(mcp_url: str):
    """
    Dynamically create methods for every tool, so that the tool can be called as a python function.

    Args:
        mcp_url: The URL of the MCP server

    Returns:
        A list of tool functions
    """
    # 1. initialize the connection and get the headers
    session_id, server_name = await _initialize_connection(mcp_url)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "mcp-session-id": session_id,
    }

    # 2. initialize the notification
    await _initialize_notification(mcp_url, headers)

    # 3. list all tools from the server
    tools = await _discover_tools(mcp_url, headers)

    tools_to_return = []

    # helper shared by all generated methods
    async def _call_tool(tool_name, **kwargs):
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "python", "version": "3.12"},
                "name": tool_name,
                "arguments": kwargs,
            },
        }
        async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
            r = await client.post(
                mcp_url,
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    reply = json.loads(line.removeprefix("data:").strip())
                    if reply.get("error"):
                        raise RuntimeError(reply["error"])
                    return (
                        reply["result"]["content"][0]["text"]
                        if reply["result"]["content"]
                        else None
                    )
            raise RuntimeError("tools/call returned no data lines")

    # generate one method per tool
    for tool in tools:
        name = tool["name"]
        # bind the helper with the concrete tool name baked in
        func = partial(_call_tool, tool_name=name)
        # make the function signature reflect the tool's schema (optional)
        props = tool["inputSchema"]["properties"]
        params = [
            inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY) for k in props.keys()
        ]
        sig = inspect.Signature(parameters=params, return_annotation=object)
        func.__signature__ = sig
        func.__doc__ = tool["description"]
        func.__name__ = f"mcp__{server_name}__{name}"
        tools_to_return.append(func)

    # return an object masquerading as a simple namespace
    return tools_to_return
