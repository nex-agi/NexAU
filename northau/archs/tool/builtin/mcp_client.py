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
    """HTTP MCP session that handles MCP streamable HTTP with session management."""
    
    def __init__(self, config: 'MCPServerConfig', headers: Dict[str, str], timeout: float):
        self.config = config
        self.headers = headers
        self.timeout = timeout
        self._request_id = 0
        self._session_id: Optional[str] = None
        self._initialized = False
    
    def _get_next_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _initialize_session(self) -> None:
        """Initialize the MCP session."""
        if self._initialized:
            return
            
        import httpx
        
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "northau-mcp-client",
                    "version": "1.0.0"
                }
            },
            "id": self._get_next_id()
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if not self.config.url:
                raise ValueError("Server URL is required")
            
            response = await client.post(
                self.config.url,
                json=init_request,
                headers=self.headers
            )
            response.raise_for_status()
            
            # Extract session ID from response headers
            session_id = response.headers.get('mcp-session-id')
            if session_id:
                self._session_id = session_id
                logger.debug(f"Captured session ID: {session_id}")
            
            # Parse response - handle both SSE and standard JSON formats
            response_text = response.text
            logger.debug(f"Initialize response: {response_text}")
            
            result = None
            
            if "event: message" in response_text and "data: " in response_text:
                # Handle SSE format (FastMCP style)
                lines = response_text.strip().split('\n')
                json_data = None
                for line in lines:
                    if line.startswith("data: "):
                        json_data = line[6:]  # Remove "data: " prefix
                        break
                
                if json_data:
                    import json
                    result = json.loads(json_data)
            else:
                # Handle standard JSON format (GitHub MCP style)
                try:
                    import json
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise Exception(f"Could not parse initialization response as JSON: {response_text}")
            
            if not result:
                raise Exception(f"No valid response data found in initialization response: {response_text}")
            
            if "error" in result:
                raise Exception(f"MCP initialization error: {result['error']}")
            
            # Check if we got a proper initialization result
            if "result" in result:
                logger.debug(f"Initialization successful, server info: {result.get('result', {}).get('serverInfo', 'Unknown')}")
                # Send initialized notification as per MCP protocol
                await self._send_initialized_notification()
                self._initialized = True
                logger.info(f"MCP HTTP session initialized successfully with session ID: {self._session_id}")
                return
                
            # If we get here, the response format was unexpected
            raise Exception(f"Unexpected initialization response format - missing 'result' field: {response_text}")
    
    async def _send_initialized_notification(self) -> None:
        """Send the initialized notification as per MCP protocol."""
        import httpx
        
        notification_data = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        # Prepare headers with session ID
        request_headers = self.headers.copy()
        if self._session_id:
            request_headers["mcp-session-id"] = self._session_id
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if not self.config.url:
                raise ValueError("Server URL is required")
            response = await client.post(
                self.config.url,
                json=notification_data,
                headers=request_headers
            )
            # Notifications don't expect a response, but check for errors
            if response.status_code >= 400:
                logger.warning(f"Initialized notification returned status {response.status_code}")
            else:
                logger.debug("Initialized notification sent successfully")
        
    async def _make_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a direct HTTP request to the MCP server."""
        import httpx
        import json
        
        # Ensure session is initialized
        if not self._initialized:
            await self._initialize_session()
        
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._get_next_id()
        }
        
        # Only include params if they are provided and not empty
        if params:
            request_data["params"] = params
        
        # Prepare headers with session ID if available
        request_headers = self.headers.copy()
        if self._session_id:
            request_headers["mcp-session-id"] = self._session_id
            logger.debug(f"Including session ID in request: {self._session_id}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if not self.config.url:
                raise ValueError("Server URL is required")
            response = await client.post(
                self.config.url,
                json=request_data,
                headers=request_headers
            )
            response.raise_for_status()
            
            # Handle SSE response format
            response_text = response.text
            
            if "event: message" in response_text and "data: " in response_text:
                # Extract JSON from SSE format - need to find the message with matching ID
                lines = response_text.strip().split('\n')
                request_id = request_data.get("id")
                
                # Parse all SSE messages and find the one with matching request ID
                for line in lines:
                    if line.startswith("data: "):
                        try:
                            json_data = line[6:]  # Remove "data: " prefix
                            parsed_message = json.loads(json_data)
                            
                            # Check if this message has the matching request ID (for responses)
                            # or if it's a notification (no ID required)
                            if "id" in parsed_message and parsed_message["id"] == request_id:
                                logger.debug(f"Found matching response for request ID {request_id}")
                                return parsed_message
                            elif request_id is None and "method" not in parsed_message:
                                # For requests without ID (notifications), return first non-notification
                                return parsed_message
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse SSE line as JSON: {line[:100]}...")
                            continue
                
                # If no matching message found, log the issue and try to return the last valid message
                logger.warning(f"No matching response found for request ID {request_id} in SSE stream")
                logger.debug(f"Full SSE response: {response_text}")
                
                # As fallback, try to return the last message that looks like a response
                for line in reversed(lines):
                    if line.startswith("data: "):
                        try:
                            json_data = line[6:]
                            parsed_message = json.loads(json_data)
                            if "result" in parsed_message or "error" in parsed_message:
                                logger.debug(f"Using fallback response: {parsed_message}")
                                return parsed_message
                        except json.JSONDecodeError:
                            continue
            
            # Fallback to regular JSON parsing
            try:
                return response.json()
            except:
                # If JSON parsing fails, return the raw response
                raise Exception(f"Could not parse response: {response_text}")
        
    async def list_tools(self):
        """List tools by making a direct HTTP request."""
        # Ensure session is initialized
        if not self._initialized:
            await self._initialize_session()
        
        response = await self._make_request("tools/list")
        logger.debug(f"Tools list response: {response}")
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
        
        # Convert the response to match the expected format
        tools_data = response.get("result", {}).get("tools", [])
        logger.debug(f"Extracted tools data: {tools_data}")
        
        # Create a simple object structure that matches what MCPTool expects
        class SimpleToolList:
            def __init__(self, tools_data):
                self.tools = [SimpleTool(tool_data) for tool_data in tools_data]
        
        class SimpleTool:
            def __init__(self, tool_data):
                self.name = tool_data["name"]
                self.description = tool_data.get("description", "")
                self.inputSchema = tool_data.get("inputSchema", {})
        
        result = SimpleToolList(tools_data)
        logger.debug(f"Created tool list with {len(result.tools)} tools")
        return result
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a tool by making a direct HTTP request."""
        # Ensure session is initialized
        if not self._initialized:
            await self._initialize_session()
        
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
            config = self._session_params['config']
            headers = self._session_params['headers']
            timeout = self._session_params['timeout']
            if isinstance(config, MCPServerConfig) and isinstance(headers, dict) and isinstance(timeout, (int, float)):
                return HTTPMCPSession(config, headers, timeout)
        
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
                            limit=1024*128, # set a large buffer size
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
                        request_id = self._get_next_id()
                        request = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "method": method,
                            "params": params or {}
                        }
                        
                        # Send request
                        request_str = json.dumps(request) + "\n"
                        self.process.stdin.write(request_str.encode())
                        await self.process.stdin.drain()
                        
                        # Read responses until we get the one matching our request ID
                        max_attempts = 10  # Prevent infinite loops
                        attempts = 0
                        
                        while attempts < max_attempts:
                            response_line = await self.process.stdout.readline()
                            
                            if not response_line:
                                raise Exception("MCP server closed connection")
                                
                            try:
                                response = json.loads(response_line.decode().strip())
                                
                                # Check if this is a response to our request
                                if "id" in response and response["id"] == request_id:
                                    if "error" in response:
                                        raise Exception(f"MCP error: {response['error']}")
                                    return response
                                # Ignore notifications and responses to other requests
                                    
                            except json.JSONDecodeError:
                                # Ignore malformed JSON
                                pass
                                
                            attempts += 1
                        
                        raise Exception(f"No matching response received for request ID {request_id} after {max_attempts} attempts")
                        
                    async def call_tool(self, name, arguments):
                        try:
                            response = await self._make_request("tools/call", {"name": name, "arguments": arguments})
                            result_data = response.get("result", {})
                            
                            class ToolCallResult:
                                def __init__(self, data):
                                    self.content = data.get("content", [])
                                    
                            return ToolCallResult(result_data)
                        except Exception as e:
                            logger.error(f"MCP error: {e}")
                            logger.error(f"Name: {name}")
                            logger.error(f"Request: {arguments}")
                            raise Exception(f"MCP error: {e}")
                
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
        # Filter out global_storage parameter as it's not needed for MCP tools
        # and causes JSON serialization errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'global_storage'}
        return self._execute_sync(**filtered_kwargs)
    
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
                        elif isinstance(item, dict) and 'text' in item:
                            content_items.append(item['text'])
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
                            limit=1024*128, # set a large buffer size
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
                # HTTP server with FastMCP protocol
                if not config.url:
                    logger.error(f"URL required for HTTP server '{server_name}'")
                    return False
                
                # Use reasonable timeout for HTTP connections
                connection_timeout = config.timeout or 30
                
                try:
                    logger.info(f"Connecting to HTTP MCP server '{server_name}'...")
                    
                    # Ensure proper headers for FastMCP streamable HTTP protocol
                    headers = config.headers or {}
                    if "Accept" not in headers:
                        headers["Accept"] = "application/json, text/event-stream"
                    if "Content-Type" not in headers:
                        headers["Content-Type"] = "application/json"
                    
                    # Test connection with a simple session
                    test_session = HTTPMCPSession(config, headers, connection_timeout)
                    
                    # Test basic connectivity by trying to initialize
                    await test_session._initialize_session()
                    logger.info(f"Successfully tested HTTP MCP server connection: {server_name}")
                    
                    # Store the session configuration for later use
                    self.sessions[server_name] = HTTPMCPSession(config, headers, connection_timeout)
                    logger.info(f"Successfully connected to HTTP MCP server: {server_name}")
                    return True
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Connection to HTTP MCP server '{server_name}' timed out after {connection_timeout}s")
                    logger.warning(f"  URL: {config.url}")
                    return False
                except Exception as e:
                    logger.error(f"Failed to connect to HTTP MCP server '{server_name}': {e}")
                    logger.debug(f"Error details: {e}")
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