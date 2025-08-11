"""MCP client implementation for Northau framework."""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as MCPToolType, CallToolRequest

from ..tool import Tool

logger = logging.getLogger(__name__)


class HTTPMCPSession:
    """HTTP MCP session that handles Connection: close servers with direct HTTP requests."""
    
    def __init__(self, config: 'MCPServerConfig', headers: Dict[str, str], timeout: float):
        self.config = config
        self.headers = headers
        self.timeout = timeout
        self._request_id = 0
    
    def _get_next_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id
        
    async def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a direct HTTP request to the MCP server."""
        import httpx
        
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._get_next_id()
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.config.url,
                json=request_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        
    async def list_tools(self):
        """List tools by making a direct HTTP request."""
        response = await self._make_request("tools/list")
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
        
        # Convert the response to match the expected format
        tools_data = response.get("result", {}).get("tools", [])
        
        # Create a simple object structure that matches what MCPTool expects
        class SimpleToolList:
            def __init__(self, tools_data):
                self.tools = [SimpleTool(tool_data) for tool_data in tools_data]
        
        class SimpleTool:
            def __init__(self, tool_data):
                self.name = tool_data["name"]
                self.description = tool_data.get("description", "")
                self.inputSchema = tool_data.get("inputSchema", {})
        
        return SimpleToolList(tools_data)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a tool by making a direct HTTP request."""
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        
        response = await self._make_request("tools/call", params)
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
        
        # Create a simple result object that matches what MCPTool expects
        class SimpleResult:
            def __init__(self, result_data):
                self.content = result_data.get("content", [])
        
        return SimpleResult(response.get("result", {}))


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
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


class MCPTool(Tool):
    """Wrapper for MCP tools to conform to Northau Tool interface."""
    
    def __init__(self, mcp_tool: MCPToolType, client_session: Union[ClientSession, HTTPMCPSession], server_config: Optional[MCPServerConfig] = None):
        self.mcp_tool = mcp_tool
        self.client_session = client_session
        self.server_config = server_config  # Store config for recreating sessions
        self._session_type = type(client_session).__name__
        
        # Store session parameters for thread-safe recreation
        if isinstance(client_session, HTTPMCPSession):
            self._session_params = {
                'config': client_session.config,
                'headers': client_session.headers,
                'timeout': client_session.timeout
            }
        else:
            # For stdio sessions, store the server config for recreation
            self._session_params = server_config
        
        # Convert MCP tool to Northau tool format
        super().__init__(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            input_schema=mcp_tool.inputSchema,
            implementation=self._execute_sync
        )
    
    async def _get_thread_local_session(self) -> Union[ClientSession, HTTPMCPSession]:
        """Get or create a thread-local session to avoid event loop conflicts."""
        # For HTTPMCPSession, we can safely create a new instance in each thread
        if self._session_type == 'HTTPMCPSession' and isinstance(self._session_params, dict):
            return HTTPMCPSession(
                self._session_params['config'],
                self._session_params['headers'],
                self._session_params['timeout']
            )
        
        # For stdio sessions, we need to create a new DirectMCPSession
        elif self._session_type == 'DirectMCPSession' and isinstance(self._session_params, MCPServerConfig):
            config = self._session_params
            if config.type == "stdio" and config.command:
                # Create a new DirectMCPSession
                import os
                merged_env = os.environ.copy()
                if config.env:
                    merged_env.update(config.env)
                
                # Use the DirectMCPSession class definition from the original code
                class DirectMCPSession:
                    def __init__(self, command, args, env):
                        self.command = command
                        self.args = args
                        self.env = env
                        self.process = None
                        self._request_id = 0
                        
                    async def initialize(self):
                        import subprocess
                        self.process = await asyncio.create_subprocess_exec(
                            self.command,
                            *self.args,
                            env=self.env,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        if not self.process.stdout or not self.process.stdin:
                            raise RuntimeError("Failed to get process streams")
                            
                        # Initialize the MCP connection properly
                        await self._initialize_connection()
                        return self
                    
                    def _get_next_id(self):
                        self._request_id += 1
                        return self._request_id
                        
                    async def _initialize_connection(self):
                        """Initialize the MCP connection with proper handshake."""
                        import json
                        # Send initialize request
                        init_request = {
                            "jsonrpc": "2.0",
                            "id": self._get_next_id(),
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {
                                    "roots": {
                                        "listChanged": True
                                    },
                                    "sampling": {}
                                },
                                "clientInfo": {
                                    "name": "northau-mcp-client",
                                    "version": "1.0.0"
                                }
                            }
                        }
                        
                        # Send the initialize request
                        init_str = json.dumps(init_request) + "\n"
                        self.process.stdin.write(init_str.encode())
                        await self.process.stdin.drain()
                        
                        # Read the initialize response
                        response_line = await self.process.stdout.readline()
                        response = json.loads(response_line.decode().strip())
                        
                        if "error" in response:
                            raise Exception(f"MCP initialization error: {response['error']}")
                            
                        # Send initialized notification
                        initialized_notification = {
                            "jsonrpc": "2.0",
                            "method": "notifications/initialized",
                            "params": {}
                        }
                        
                        notif_str = json.dumps(initialized_notification) + "\n"
                        self.process.stdin.write(notif_str.encode())
                        await self.process.stdin.drain()
                        
                        # Test with tools/list to verify connection
                        await self._make_request("tools/list")
                        
                    async def _make_request(self, method, params=None):
                        if not self.process or not self.process.stdin or not self.process.stdout:
                            raise RuntimeError("Process not initialized")
                            
                        import json
                        request = {
                            "jsonrpc": "2.0",
                            "id": self._get_next_id(),
                            "method": method,
                            "params": params or {}
                        }
                        
                        # Send request
                        request_str = json.dumps(request) + "\n"
                        self.process.stdin.write(request_str.encode())
                        await self.process.stdin.drain()
                        
                        # Read response
                        response_line = await self.process.stdout.readline()
                        response = json.loads(response_line.decode().strip())
                        
                        if "error" in response:
                            raise Exception(f"MCP error: {response['error']}")
                            
                        return response
                        
                    async def call_tool(self, name, arguments):
                        response = await self._make_request("tools/call", {"name": name, "arguments": arguments})
                        result_data = response.get("result", {})
                        
                        class ToolCallResult:
                            def __init__(self, data):
                                self.content = data.get("content", [])
                                
                        return ToolCallResult(result_data)
                
                # Create and initialize new session
                session = DirectMCPSession(config.command, config.args or [], merged_env)
                await session.initialize()
                return session
        
        # For other session types, try to use the original session
        # This is a fallback - ideally all MCP sessions should be recreatable
        return self.client_session
    
    def _execute_sync(self, **kwargs) -> Dict[str, Any]:
        """Execute the MCP tool synchronously."""
        # Always create a new event loop to avoid "Future attached to a different loop" errors
        # This is necessary when running in multi-threaded environments like ThreadPoolExecutor
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._execute_async(**kwargs))
        finally:
            # Clean up the event loop
            try:
                loop.close()
            except Exception:
                pass
            # Reset to previous loop if it exists
            try:
                asyncio.get_event_loop()
            except RuntimeError:
                # No existing loop, that's fine
                pass
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the MCP tool synchronously (for backward compatibility)."""
        return self._execute_sync(**kwargs)
    
    async def _execute_async(self, **kwargs) -> Dict[str, Any]:
        """Execute the MCP tool asynchronously."""
        try:
            # Create a thread-local session to avoid event loop conflicts
            session = await self._get_thread_local_session()
            result = await session.call_tool(self.name, kwargs)
            
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    # Handle list of content items
                    content_items = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            content_items.append(item.text)
                        else:
                            content_items.append(str(item))
                    return {"result": "\n".join(content_items)}
                else:
                    return {"result": str(result.content)}
            else:
                return {"result": str(result)}
                
        except Exception as e:
            logger.error(f"Error executing MCP tool '{self.name}': {e}")
            return {"error": str(e)}


class MCPClient:
    """Client for connecting to and managing MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, MCPServerConfig] = {}
        self.sessions: Dict[str, Union[ClientSession, HTTPMCPSession]] = {}
        self.tools: Dict[str, MCPTool] = {}
        
    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self.servers[config.name] = config
        logger.info(f"Added MCP server configuration: {config.name}")
    
    async def connect_to_server(self, server_name: str) -> bool:
        """Connect to an MCP server and initialize the session."""
        if server_name not in self.servers:
            logger.error(f"Server '{server_name}' not found in configurations")
            return False
            
        config = self.servers[server_name]
        
        try:
            if config.type == "stdio":
                # Stdio server
                if not config.command:
                    logger.error(f"Command required for stdio server '{server_name}'")
                    return False
                
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env or {}
                )
                
                # Use asyncio.wait_for for timeout protection
                timeout_duration = config.timeout or 30
                
                # Use direct subprocess approach with proper JSON-RPC protocol
                logger.info(f"Starting stdio MCP server: {config.command} {' '.join(config.args or [])}")
                
                # Override the environment to preserve PATH
                import os
                merged_env = os.environ.copy()
                if server_params.env:
                    merged_env.update(server_params.env)
                
                # Create a direct subprocess-based MCP session
                class DirectMCPSession:
                    def __init__(self, command, args, env):
                        self.command = command
                        self.args = args
                        self.env = env
                        self.process = None
                        self._request_id = 0
                        
                    async def initialize(self):
                        import subprocess
                        self.process = await asyncio.create_subprocess_exec(
                            self.command,
                            *self.args,
                            env=self.env,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        if not self.process.stdout or not self.process.stdin:
                            raise RuntimeError("Failed to get process streams")
                            
                        # Initialize the MCP connection properly
                        await self._initialize_connection()
                        return self
                    
                    def _get_next_id(self):
                        self._request_id += 1
                        return self._request_id
                        
                    async def _initialize_connection(self):
                        """Initialize the MCP connection with proper handshake."""
                        import json
                        # Send initialize request
                        init_request = {
                            "jsonrpc": "2.0",
                            "id": self._get_next_id(),
                            "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {
                                    "roots": {
                                        "listChanged": True
                                    },
                                    "sampling": {}
                                },
                                "clientInfo": {
                                    "name": "northau-mcp-client",
                                    "version": "1.0.0"
                                }
                            }
                        }
                        
                        # Send the initialize request
                        init_str = json.dumps(init_request) + "\n"
                        self.process.stdin.write(init_str.encode())
                        await self.process.stdin.drain()
                        
                        # Read the initialize response
                        response_line = await self.process.stdout.readline()
                        response = json.loads(response_line.decode().strip())
                        
                        if "error" in response:
                            raise Exception(f"MCP initialization error: {response['error']}")
                            
                        # Send initialized notification
                        initialized_notification = {
                            "jsonrpc": "2.0",
                            "method": "notifications/initialized",
                            "params": {}
                        }
                        
                        notif_str = json.dumps(initialized_notification) + "\n"
                        self.process.stdin.write(notif_str.encode())
                        await self.process.stdin.drain()
                        
                        # Test with tools/list to verify connection
                        await self._make_request("tools/list")
                        
                    async def _make_request(self, method, params=None):
                        if not self.process or not self.process.stdin or not self.process.stdout:
                            raise RuntimeError("Process not initialized")
                            
                        import json
                        request = {
                            "jsonrpc": "2.0",
                            "id": self._get_next_id(),
                            "method": method,
                            "params": params or {}
                        }
                        
                        # Send request
                        request_str = json.dumps(request) + "\n"
                        self.process.stdin.write(request_str.encode())
                        await self.process.stdin.drain()
                        
                        # Read response
                        response_line = await self.process.stdout.readline()
                        response = json.loads(response_line.decode().strip())
                        
                        if "error" in response:
                            raise Exception(f"MCP error: {response['error']}")
                            
                        return response
                        
                    async def list_tools(self):
                        response = await self._make_request("tools/list")
                        tools_data = response.get("result", {}).get("tools", [])
                        
                        # Create tool objects
                        class SimpleTool:
                            def __init__(self, data):
                                self.name = data["name"] 
                                self.description = data.get("description", "")
                                self.inputSchema = data.get("inputSchema", {})
                                
                        class ToolResult:
                            def __init__(self, tools):
                                self.tools = [SimpleTool(tool) for tool in tools]
                                
                        return ToolResult(tools_data)
                        
                    async def call_tool(self, name, arguments):
                        response = await self._make_request("tools/call", {"name": name, "arguments": arguments})
                        result_data = response.get("result", {})
                        
                        class ToolCallResult:
                            def __init__(self, data):
                                self.content = data.get("content", [])
                                
                        return ToolCallResult(result_data)
                
                # Create and initialize the session
                session = DirectMCPSession(server_params.command, server_params.args, merged_env)
                await asyncio.wait_for(session.initialize(), timeout=timeout_duration)
                self.sessions[server_name] = session
                logger.info(f"Successfully connected to stdio MCP server: {server_name}")
                return True
                    
            elif config.type == "http":
                # HTTP server
                if not config.url:
                    logger.error(f"URL required for HTTP server '{server_name}'")
                    return False
                
                # Use reasonable timeout for HTTP connections
                connection_timeout = config.timeout or 30
                
                try:
                    # Test server accessibility first with a shorter timeout
                    logger.info(f"Testing HTTP MCP server '{server_name}' accessibility...")
                    
                    async def connect_http():
                        # Ensure proper headers for MCP streamable HTTP protocol
                        headers = config.headers or {}
                        # Add required Accept header if not present
                        if "Accept" not in headers:
                            headers["Accept"] = "application/json, text/event-stream"
                        if "Content-Type" not in headers:
                            headers["Content-Type"] = "application/json"
                        
                        logger.info(f"Connecting to HTTP MCP server '{server_name}' with timeout {connection_timeout}s")
                        logger.debug(f"URL: {config.url}")
                        logger.debug(f"Headers: {headers}")
                        
                        try:
                            if not config.url:
                                raise ValueError("URL is required for HTTP server")
                            
                            # For HTTP servers, we'll test the connection and then store the config
                            # The actual persistent connection will be established per-request
                            logger.info(f"Testing HTTP MCP server connection to '{server_name}'...")
                            
                            # Test connection with a short timeout first
                            test_timeout = min(5, connection_timeout)
                            
                            # Test connection with direct HTTP approach
                            logger.info(f"Testing HTTP MCP server with direct requests to '{server_name}'...")
                            
                            # Create test session and try to list tools
                            test_session = HTTPMCPSession(config, headers, test_timeout)
                            
                            # Test basic connectivity and tool listing
                            tools_result = await test_session.list_tools()
                            logger.info(f"Successfully retrieved {len(tools_result.tools)} tools from '{server_name}'")
                            
                            # Store a session that will create real connections on demand
                            self.sessions[server_name] = HTTPMCPSession(config, headers, connection_timeout)
                            logger.info(f"Successfully configured HTTP MCP server: {server_name}")
                            return True
                        except Exception as e:
                            logger.warning(f"HTTP client connection failed for '{server_name}': {e}")
                            raise
                    
                    await asyncio.wait_for(connect_http(), timeout=connection_timeout)
                    logger.info(f"Successfully connected to HTTP MCP server: {server_name}")
                    return True
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Connection to HTTP MCP server '{server_name}' timed out after {connection_timeout}s")
                    logger.warning(f"  URL: {config.url}")
                    logger.warning(f"  This may indicate the server is unresponsive or the URL/API key is incorrect")
                    return False
                    
            else:
                logger.error(f"Unknown server type '{config.type}' for server '{server_name}'")
                return False
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection to MCP server '{server_name}' timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
            # Log more detailed error information for debugging
            import traceback
            logger.debug(f"Detailed error for '{server_name}': {traceback.format_exc()}")
            return False
    
    async def discover_tools(self, server_name: str) -> List[MCPTool]:
        """Discover tools from an MCP server."""
        if server_name not in self.sessions:
            logger.error(f"No active session for server '{server_name}'")
            return []
            
        session = self.sessions[server_name]
        
        try:
            # List available tools
            tools_result = await session.list_tools()
            
            # Convert MCP tools to Northau tools
            discovered_tools = []
            server_config = self.servers.get(server_name)
            
            for mcp_tool in tools_result.tools:
                # For HTTPMCPSession, we need to pass the session directly
                # For regular ClientSession, we also pass it directly
                # Convert SimpleTool to MCPToolType-like object for compatibility
                if hasattr(mcp_tool, 'inputSchema'):
                    tool = MCPTool(mcp_tool, session, server_config)
                else:
                    # Handle SimpleTool case by converting to proper format
                    tool = MCPTool(mcp_tool, session, server_config)
                discovered_tools.append(tool)
                
                # Store in registry with server prefix to avoid conflicts
                tool_key = f"{server_name}.{mcp_tool.name}"
                self.tools[tool_key] = tool
                
            logger.info(f"Discovered {len(discovered_tools)} tools from server '{server_name}'")
            return discovered_tools
            
        except Exception as e:
            logger.error(f"Failed to discover tools from server '{server_name}': {e}")
            # Log more details for debugging
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return []
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> List[MCPTool]:
        """Get all discovered tools."""
        return list(self.tools.values())
    
    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from an MCP server."""
        if server_name in self.sessions:
            try:
                session = self.sessions[server_name]
                # The session should handle cleanup automatically
                # when the context manager exits
                del self.sessions[server_name]
                
                # Remove tools from this server
                tools_to_remove = [k for k in self.tools.keys() if k.startswith(f"{server_name}.")]
                for tool_key in tools_to_remove:
                    del self.tools[tool_key]
                    
                logger.info(f"Disconnected from MCP server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from server '{server_name}': {e}")
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server_name in list(self.sessions.keys()):
            await self.disconnect_server(server_name)


class MCPManager:
    """High-level manager for MCP operations."""
    
    def __init__(self):
        self.client = MCPClient()
        self.auto_connect = True
    
    def add_server(self, name: str, server_type: str = "stdio", 
                   command: Optional[str] = None, args: Optional[List[str]] = None,
                   env: Optional[Dict[str, str]] = None, url: Optional[str] = None,
                   headers: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> None:
        """Add an MCP server configuration."""
        config = MCPServerConfig(
            name=name, 
            type=server_type,
            command=command, 
            args=args, 
            env=env,
            url=url,
            headers=headers,
            timeout=timeout
        )
        self.client.add_server(config)
    
    async def initialize_servers(self) -> Dict[str, List[MCPTool]]:
        """Initialize all configured servers and discover their tools."""
        all_tools = {}
        
        for server_name in self.client.servers.keys():
            if await self.client.connect_to_server(server_name):
                tools = await self.client.discover_tools(server_name)
                all_tools[server_name] = tools
            else:
                all_tools[server_name] = []
                
        return all_tools
    
    def get_available_tools(self) -> List[MCPTool]:
        """Get all available MCP tools."""
        return self.client.get_all_tools()
    
    async def shutdown(self) -> None:
        """Shutdown the MCP manager and disconnect all servers."""
        await self.client.disconnect_all()


# Global MCP manager instance
_mcp_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    """Get the global MCP manager instance."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager


async def initialize_mcp_tools(server_configs: List[Dict[str, Any]]) -> List[Tool]:
    """Initialize MCP tools from server configurations.
    
    Args:
        server_configs: List of server configuration dictionaries with keys:
            - name: Server name
            - type: Server type ("stdio" or "http") 
            For stdio servers:
            - command: Command to run
            - args: Command arguments
            - env: Optional environment variables
            For HTTP servers:
            - url: Server URL
            - headers: Optional HTTP headers
            - timeout: Optional timeout in seconds
    
    Returns:
        List of initialized MCP tools
    """
    manager = get_mcp_manager()
    
    # Add server configurations
    for config in server_configs:
        manager.add_server(
            name=config['name'],
            server_type=config.get('type', 'stdio'),
            command=config.get('command'),
            args=config.get('args'),
            env=config.get('env'),
            url=config.get('url'),
            headers=config.get('headers'),
            timeout=config.get('timeout')
        )
    
    # Initialize servers and discover tools
    await manager.initialize_servers()
    
    # Return all available tools
    return manager.get_available_tools()


def sync_initialize_mcp_tools(server_configs: List[Dict[str, Any]]) -> List[Tool]:
    """Synchronous wrapper for initialize_mcp_tools."""
    # Always create a new event loop to avoid "Future attached to a different loop" errors
    # This matches the pattern used in MCPTool._execute_sync
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(initialize_mcp_tools(server_configs))
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except Exception:
            pass
        # Reset to previous loop if it exists
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # No existing loop, that's fine
            pass