from . import tools

try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("logtriage")


@mcp.tool()
def triage_log(log_text: str, backend: str = "offline", prompt_version: str = "v2") -> dict:
    r = tools.triage_log(log_text, backend, prompt_version)
    return {k: v for k, v in r.items() if k != "report_md"}


@mcp.tool()
def search_defects(query: str, k: int = 3) -> list[dict]:
    return tools.search_defects(query, k)


@mcp.tool()
def list_error_events(log_text: str, only_suspects: bool = True) -> list[dict]:
    return tools.list_error_events(log_text, only_suspects)


@mcp.tool()
def get_log_context(log_text: str, line: int, radius: int = 5) -> str:
    return tools.get_log_context(log_text, line, radius)


if __name__ == "__main__":
    mcp.run()
