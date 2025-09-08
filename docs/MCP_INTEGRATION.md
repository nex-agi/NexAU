# MCP (Model Context Protocol) Integration

This document describes the MCP integration implementation in the Northau framework.

## Overview

The Northau framework now supports MCP servers, allowing agents to dynamically discover and use tools from MCP-compliant servers. This enables integration with a wide variety of external services and tools.

## Implementation

The MCP integration consists of several components:

### 1. MCPServerConfig
Configuration class for MCP servers supporting both stdio and HTTP protocols:

```python
@dataclass
class MCPServerConfig:
    name: str
    type: str = "stdio"  # "stdio" or "http"
    # For stdio servers
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # For HTTP servers
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[float] = 30
```

### 2. MCPTool
Wrapper class that adapts MCP tools to the Northau Tool interface:

```python
class MCPTool(Tool):
    """Wrapper for MCP tools to conform to Northau Tool interface."""

    def __init__(self, mcp_tool: MCPToolType, client_session: ClientSession):
        self.mcp_tool = mcp_tool
        self.client_session = client_session
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            input_schema=mcp_tool.inputSchema,
            implementation=self._execute_sync
        )
```

### 3. MCPClient
Core client for managing connections to MCP servers:

```python
class MCPClient:
    """Client for connecting to and managing MCP servers."""

    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
```

### 4. MCPManager
High-level manager for MCP operations:

```python
class MCPManager:
    """High-level manager for MCP operations."""

    def add_server(self, name: str, server_type: str = "stdio", ...):
        """Add an MCP server configuration."""

    async def initialize_servers(self) -> Dict[str, List[MCPTool]]:
        """Initialize all configured servers and discover their tools."""
```

## Usage

### Basic Usage

```python
from northau.archs.main_sub.agent import create_agent

# Configure MCP servers
mcp_servers = [
    {
        "name": "filesystem",
        "type": "stdio",
        "command": "mcp-server-filesystem",
        "args": ["--base-path", "/tmp"]
    },
    {
        "name": "amap-maps",
        "type": "http",
        "url": "https://mcp.amap.com/mcp?key=your-api-key"
    }
]

# Create agent with MCP tools
agent = create_agent(
    name="mcp_agent",
    mcp_servers=mcp_servers,
    system_prompt="You are an agent with access to various MCP tools..."
)

# Use the agent
response = agent.run("List files in the /tmp directory")
```

### Amap Maps Example

```python
mcp_servers = [
    {
        "name": "amap-maps-streamableHTTP",
        "type": "http",
        "url": "https://mcp.amap.com/mcp?key=4a1f6a2bb045e3d2e05461265bc8ead8",
        "headers": {
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        "timeout": 30
    }
]

agent = create_agent(
    name="amap_agent",
    mcp_servers=mcp_servers,
    system_prompt="You are an AI agent with access to Amap Maps services..."
)

# Example queries:
# "Find restaurants near Beijing Central Business District"
# "Get directions from Beijing Capital Airport to Tian'anmen Square"
# "How long does it take to drive from Shanghai to Hangzhou?"
```

## Current Limitations

1. **Session Management**: The current implementation has challenges with keeping MCP sessions alive outside their async context managers. This is a limitation of the MCP Python library design.

2. **Error Handling**: Connection failures are logged but don't prevent agent creation, which is by design for robustness.

3. **Async/Sync Bridge**: The implementation uses event loop management to bridge async MCP calls with the synchronous Tool interface.

## Future Improvements

1. **Connection Pooling**: Implement proper connection pooling and session management.
2. **Retry Logic**: Add automatic retry mechanisms for failed connections.
3. **Health Monitoring**: Monitor MCP server health and automatically reconnect.
4. **Tool Caching**: Cache discovered tools to reduce startup time.
5. **Dynamic Discovery**: Support runtime discovery and addition of new MCP servers.

## Integration Points

The MCP integration plugs into the Northau framework at several points:

1. **Agent Constructor**: The `mcp_servers` parameter in `create_agent()` and `Agent.__init__()`
2. **Tool System**: MCP tools are converted to Northau tools and added to the agent's tool registry
3. **Builtin Tools**: MCP client components are exported from `northau.archs.tool.builtin`

## Testing

The integration includes comprehensive tests:

- `test_mcp_integration.py`: Basic functionality tests
- `examples/mcp_agent_example.py`: General MCP usage example
- `examples/mcp_amap_example.py`: Amap Maps specific example

## Dependencies

- `mcp`: The official MCP Python library
- `asyncio`: For async/await support
- `httpx`: For HTTP-based MCP servers (via mcp library)

## Configuration Schema

Server configurations follow this schema:

```json
{
  "name": "server-name",
  "type": "stdio|http",
  "command": "command-for-stdio",
  "args": ["arg1", "arg2"],
  "env": {"ENV_VAR": "value"},
  "url": "https://server.com/mcp",
  "headers": {"Header": "value"},
  "timeout": 30
}
```

This MCP integration makes the Northau framework compatible with the growing ecosystem of MCP servers and tools.
