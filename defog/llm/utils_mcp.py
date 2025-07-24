import json
import httpx
import inspect
from functools import partial


async def _discover_tools(mcp_url: str) -> list[dict]:
    """
    Hit `tools/list` and return the decoded tool records.

    Args:
        mcp_url: The URL of the MCP server

    Returns:
        A list of tool records
    """
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.post(
            mcp_url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
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
    # 1. list all tools from the server
    tools = await _discover_tools(mcp_url)
    tools_to_return = []

    # helper shared by all generated methods
    async def _call_tool(tool_name, **kwargs):
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": kwargs},
        }
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.post(
                mcp_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
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
        tools_to_return.append(func)

    # return an object masquerading as a simple namespace
    return tools_to_return
