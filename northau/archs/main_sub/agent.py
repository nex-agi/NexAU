"""Refactored Agent implementation for the Northau framework."""
import logging
import uuid
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

try:
    from langfuse import Langfuse

    try:
        from langfuse import openai
    except ImportError:
        import openai
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    try:
        import openai
    except ImportError:
        openai = None

from northau.archs.main_sub.config import AgentConfig, ExecutionConfig
from northau.archs.main_sub.execution.executor import Executor
from northau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from northau.archs.main_sub.agent_state import AgentState
from northau.archs.main_sub.utils.cleanup_manager import cleanup_manager
from northau.archs.main_sub.utils.token_counter import TokenCounter
from northau.archs.main_sub.prompt_builder import PromptBuilder
from northau.archs.llm.llm_config import LLMConfig

# Setup logger for agent execution
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Agent:
    """Lightweight agent container focusing on configuration and delegation."""

    def __init__(
        self,
        config: AgentConfig,
        global_storage: Optional[GlobalStorage] = None,
        exec_config: Optional[ExecutionConfig] = None,
    ):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration
            global_storage: Optional global storage instance
            exec_config: Optional execution configuration
        """
        self.config = config
        self.global_storage = (
            global_storage if global_storage is not None else GlobalStorage()
        )
        self.exec_config = exec_config if exec_config is not None else ExecutionConfig()

        # Initialize services
        self.openai_client = self._initialize_openai_client()
        self.langfuse_client = self._initialize_langfuse_client()

        # Initialize MCP tools if configured
        if self.config.mcp_servers:
            self._initialize_mcp_tools()

        # Build tool registry for quick lookup
        self.tool_registry = {tool.name: tool for tool in self.config.tools}
        self.serial_tool_name = [
            tool.name for tool in self.config.tools if tool.disable_parallel
        ]

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()

        # Initialize execution components
        self._initialize_execution_components()

        # Conversation history
        self.history = []

        # Queue for messages to be processed in the next execution cycle
        self.queued_messages = []

        # Register for cleanup
        cleanup_manager.register_agent(self)

    def _initialize_openai_client(self) -> Any:
        """Initialize OpenAI client from LLM config."""
        if openai is None:
            logger.warning('⚠️ OpenAI package not available')
            return None

        try:
            client_kwargs = self.config.llm_config.to_client_kwargs()
            return openai.OpenAI(**client_kwargs)
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            return None

    def _initialize_langfuse_client(self) -> Any:
        """Initialize Langfuse client if available."""
        if not LANGFUSE_AVAILABLE:
            return None

        try:
            import os

            public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
            secret_key = os.getenv('LANGFUSE_SECRET_KEY')
            host = os.getenv('LANGFUSE_HOST')

            if public_key and secret_key:
                client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info(
                    f"✅ Langfuse client initialized with host: {host or 'default'}",
                )
                return client
            else:
                logger.warning(
                    '⚠️ Langfuse environment variables not found, tracing disabled',
                )
                return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Langfuse client: {e}")
            return None

    def _initialize_mcp_tools(self) -> None:
        """Initialize tools from MCP servers."""
        try:
            from ..tool.builtin import sync_initialize_mcp_tools

            logger.info(
                f"Initializing MCP tools from {len(self.config.mcp_servers)} servers",
            )

            mcp_tools = sync_initialize_mcp_tools(self.config.mcp_servers)
            self.config.tools.extend(mcp_tools)

            logger.info(f"Successfully initialized {len(mcp_tools)} MCP tools")

        except ImportError:
            logger.error(
                'MCP client not available. Please install the mcp package.',
            )
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")

    def _initialize_execution_components(self) -> None:
        """Initialize execution components."""
        # Initialize the Executor
        self.executor = Executor(
            agent_name=self.config.name,
            agent_id=self.config.agent_id,
            tool_registry=self.tool_registry,
            serial_tool_name=self.serial_tool_name,
            sub_agent_factories=self.config.sub_agent_factories,
            stop_tools=self.config.stop_tools,
            openai_client=self.openai_client,
            llm_config=self.config.llm_config,
            max_iterations=self.exec_config.max_iterations,
            max_context_tokens=self.exec_config.max_context_tokens,
            max_running_subagents=self.exec_config.max_running_subagents,
            retry_attempts=self.exec_config.retry_attempts,
            token_counter=self.config.token_counter or TokenCounter(),
            langfuse_client=self.langfuse_client,
            after_model_hooks=self.config.after_model_hooks,
            after_tool_hooks=self.config.after_tool_hooks,
            global_storage=self.global_storage,
            custom_llm_generator=self.config.custom_llm_generator,
        )

    def run(
        self,
        message: str,
        history: Optional[list[dict]] = None,
        context: Optional[dict] = None,
        state: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
        dump_trace_path: Optional[str] = None,
        parent_agent_state: Optional[AgentState] = None,
    ) -> Union[str, tuple[str, dict[str, Any]]]:
        """Run agent with a message and return response."""
        logger.info(f"🤖 Agent '{self.config.name}' starting execution")
        logger.info(f"📝 User message: {message}")

        # Merge initial state/config/context with provided ones
        merged_state = {**(self.config.initial_state or {})}
        if state:
            merged_state.update(state)

        merged_config = {**(self.config.initial_config or {})}
        if config:
            merged_config.update(config)

        merged_context = {**(self.config.initial_context or {})}
        if context:
            merged_context.update(context)

        # Create agent context
        with AgentContext(context=merged_context) as ctx:
            if history:
                self.history = history.copy()
            else:
                # Build and add system prompt to history
                system_prompt = self.prompt_builder.build_system_prompt(
                    agent_config=self.config,
                    tools=self.config.tools,
                    sub_agent_factories=self.config.sub_agent_factories,
                    runtime_context=merged_context,
                )
                self.history = [{'role': 'system', 'content': system_prompt}]

            # Add user message to history
            self.history.append({'role': 'user', 'content': message})

            # Create the AgentState instance
            agent_state = AgentState(
                agent_name=self.config.name,
                agent_id=self.config.agent_id,
                context=ctx,
                global_storage=self.global_storage,
                parent_agent_state=parent_agent_state,
            )

            try:
                # Execute using the executor
                if self.langfuse_client:
                    try:
                        from datetime import datetime

                        with self.langfuse_client.start_as_current_span(
                            name=f"agent_{self.config.name}",
                            input=message,
                            metadata={
                                'agent_name': self.config.name,
                                'max_iterations': self.exec_config.max_iterations,
                                'model': self.config.llm_config.model,
                                'system_prompt_type': self.config.system_prompt_type,
                                'timestamp': datetime.now().isoformat(),
                            },
                        ):
                            logger.info(
                                f"📊 Created Langfuse span for agent: {self.config.name}",
                            )

                            response, updated_messages = self.executor.execute(
                                self.history,
                                agent_state,
                                dump_trace_path,
                            )
                            self.history = updated_messages

                            self.langfuse_client.update_current_span(
                                output=response,
                                metadata={
                                    'response_length': len(response),
                                    'history_length': len(self.history),
                                    'execution_completed': True,
                                },
                            )

                            logger.info(
                                f"📤 Langfuse span completed for agent: {self.config.name}",
                            )

                        self.langfuse_client.flush()

                    except Exception as langfuse_error:
                        logger.warning(
                            f"⚠️ Langfuse tracing failed: {langfuse_error}",
                        )
                        response, updated_messages = self.executor.execute(
                            self.history,
                            agent_state,
                            dump_trace_path,
                        )
                        self.history = updated_messages
                else:
                    response, updated_messages = self.executor.execute(
                        self.history,
                        agent_state,
                        dump_trace_path,
                    )
                    self.history = updated_messages

                # Add final assistant response to history if not already included
                if (
                    not self.history
                    or self.history[-1]['role'] != 'assistant'
                    or self.history[-1]['content'] != response
                ):
                    self.history.append(
                        {'role': 'assistant', 'content': response},
                    )

                logger.info(
                    f"✅ Agent '{self.config.name}' completed execution",
                )
                return response

            except Exception as e:
                logger.error(
                    f"❌ Agent '{self.config.name}' encountered error: {e}",
                )

                if self.config.error_handler:
                    error_response = self.config.error_handler(
                        e,
                        self,
                        merged_context,
                    )
                    self.history.append(
                        {'role': 'assistant', 'content': error_response},
                    )
                    return error_response
                else:
                    error_message = f"Error: {str(e)}"
                    self.history.append(
                        {'role': 'assistant', 'content': error_message},
                    )
                    raise

    def add_tool(self, tool) -> None:
        """Add a tool to the agent."""
        self.config.tools.append(tool)
        self.tool_registry[tool.name] = tool
        self.executor.add_tool(tool)

    def add_sub_agent(self, name: str, agent_factory: Callable[[], 'Agent']) -> None:
        """Add a sub-agent factory for delegation."""
        self.config.sub_agent_factories[name] = agent_factory
        self.executor.add_sub_agent(name, agent_factory)

    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be added to the history."""
        self.executor.enqueue_message(message)

    def stop(self) -> None:
        """Clean up this agent and all its running sub-agents."""
        logger.info(
            f"🧹 Cleaning up agent '{self.config.name}' and its sub-agents...",
        )
        self.executor.cleanup()
        logger.info(f"✅ Agent '{self.config.name}' cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup when agent is garbage collected."""
        try:
            self.stop()
        except Exception:
            pass  # Avoid exceptions during garbage collection


# Factory function for agent creation
def create_agent(
    name: Optional[str] = None,
    agent_id: Optional[str] = None,
    tools: Optional[list] = None,
    sub_agents: Optional[list[tuple[str, Callable[[], 'Agent']]]] = None,
    system_prompt: Optional[str] = None,
    system_prompt_type: str = 'string',
    llm_config: Optional[Union[LLMConfig, dict[str, Any]]] = None,
    max_iterations: int = 100,
    max_context_tokens: int = 128000,
    max_running_subagents: int = 5,
    error_handler: Optional[Callable] = None,
    retry_attempts: int = 5,
    timeout: int = 300,
    # Token counting parameters
    token_counter: Optional[Callable[[list[dict[str, str]]], int]] = None,
    # Context parameters
    initial_state: Optional[dict[str, Any]] = None,
    initial_config: Optional[dict[str, Any]] = None,
    initial_context: Optional[dict[str, Any]] = None,
    # MCP parameters
    mcp_servers: Optional[list[dict[str, Any]]] = None,
    # Stop tools parameters
    stop_tools: Optional[list[str]] = None,
    # Hook parameters
    after_model_hooks: Optional[list[Callable]] = None,
    after_tool_hooks: Optional[list[Callable]] = None,
    # Global storage parameter
    global_storage: Optional[GlobalStorage] = None,
    # Custom LLM generator parameter
    custom_llm_generator: Optional[
        Callable[
            [
                Any,
                dict[str, Any],
            ],
            Any,
        ]
    ] = None,
    **llm_kwargs,
) -> Agent:
    """Create a new agent with specified configuration."""
    # Handle llm_config creation with backward compatibility
    if llm_config is None and llm_kwargs:
        llm_config = LLMConfig(**llm_kwargs)
    elif llm_config is None:
        raise ValueError('llm_config is required')

    # Create agent configuration
    agent_config = AgentConfig(
        name=name,
        agent_id=agent_id if agent_id else str(uuid.uuid4()),
        system_prompt=system_prompt,
        system_prompt_type=system_prompt_type,
        tools=tools or [],
        sub_agents=sub_agents,
        llm_config=llm_config,
        stop_tools=stop_tools or [],
        initial_state=initial_state,
        initial_config=initial_config,
        initial_context=initial_context,
        mcp_servers=mcp_servers,
        after_model_hooks=after_model_hooks,
        after_tool_hooks=after_tool_hooks,
        error_handler=error_handler,
        token_counter=token_counter,
        custom_llm_generator=custom_llm_generator,
    )

    # Create execution configuration
    exec_config = ExecutionConfig(
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        max_running_subagents=max_running_subagents,
        retry_attempts=retry_attempts,
        timeout=timeout,
    )

    return Agent(agent_config, global_storage, exec_config)
