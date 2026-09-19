"""
stdio MCP server exposing petd's tools to the Claude CLI.

The CLI spawns MCP servers as its own child processes, so this cannot
share memory with the running petd; it forwards everything to petd's
local HTTP API instead (GET /tools, POST /tool/{name}), keeping one
source of truth for the tool set.

    python -m petd.mcp_shim.robot_mcp --api http://127.0.0.1:8765

Started automatically via runtime/brain/mcp.json (see brain/claude_cli.py).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.environ.get("PETD_API", "http://127.0.0.1:8765"))
    args = parser.parse_args()

    client = httpx.AsyncClient(base_url=args.api, timeout=60.0)

    async def list_tools(_ctx, _params) -> types.ListToolsResult:
        resp = await client.get("/tools")
        resp.raise_for_status()
        return types.ListToolsResult(tools=[
            types.Tool(name=t["name"], description=t["description"], inputSchema=t["schema"])
            for t in resp.json()
        ])

    async def call_tool(_ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            resp = await client.post(f"/tool/{params.name}", json=params.arguments or {})
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:  # noqa: BLE001 - the model should see the failure
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"petd unreachable: {exc}")],
                isError=True)
        content: list = []
        if result.get("text"):
            content.append(types.TextContent(type="text", text=result["text"]))
        if result.get("image_b64"):
            content.append(types.ImageContent(type="image", data=result["image_b64"],
                                              mimeType=result.get("mime", "image/jpeg")))
        if not content:
            content.append(types.TextContent(type="text", text="done"))
        return types.CallToolResult(content=content, isError=bool(result.get("is_error")))

    server = Server("robot", on_list_tools=list_tools, on_call_tool=call_tool)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
