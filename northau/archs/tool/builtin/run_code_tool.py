"""Jupyter notebook execution tool implementation with Python and Bash kernel support."""
import logging
import time
import json
import re
from typing import Any, Optional, Dict, TYPE_CHECKING, Literal
from queue import Empty
import os

try:
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
except ImportError:
    raise ImportError("jupyter_client is required. Install with: pip install jupyter_client")

if TYPE_CHECKING:
    from ...main_sub.agent_state import AgentState

logger = logging.getLogger(__name__)

# Maximum output length to prevent overwhelming responses
MAX_OUTPUT_LENGTH = 30000
DEFAULT_TIMEOUT = 120000  # 2 minutes in milliseconds
MAX_TIMEOUT = 600000      # 10 minutes in milliseconds

# Key for storing kernel managers in agent_state global storage
KERNEL_MANAGERS_KEY = 'jupyter_kernel_managers'

# Supported kernel types and their kernel names
SUPPORTED_KERNELS = {
    'python': 'python3',
    'bash': 'bash'
}

# ANSI color code regex pattern
ANSI_COLOR_PATTERN = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\\]*\x1b\\|\x1b\][^\x07]*\x07|\x1b[=>]|\x1b[()][AB0]')


def _strip_ansi_colors(text: str) -> str:
    """Remove ANSI color codes from text."""
    return ANSI_COLOR_PATTERN.sub('', text)


def _get_kernel_managers(agent_state: 'AgentState') -> Dict[str, KernelManager]:
    """
    Get kernel managers from agent_state global storage.
    
    Args:
        agent_state: AgentState instance (required)
        
    Returns:
        Dictionary of kernel managers
    """
    kernel_managers = agent_state.get_global_value(KERNEL_MANAGERS_KEY, None)
    if kernel_managers is None:
        kernel_managers = {}
        agent_state.set_global_value(KERNEL_MANAGERS_KEY, kernel_managers)
    
    return kernel_managers


def _get_or_create_kernel(
    kernel_type: str,
    workspace: Optional[str],
    agent_state: 'AgentState',
    extra_env: Optional[Dict[str, str]] = None
) -> KernelManager:
    """
    Get existing kernel or create a new one with environment variables.
    
    Args:
        kernel_type: Type of kernel ('python' or 'bash')
        workspace: Working directory for the kernel
        agent_state: AgentState containing global storage
        extra_env: Additional environment variables to set
    """
    kernel_name = SUPPORTED_KERNELS[kernel_type]
    kernel_managers = _get_kernel_managers(agent_state)
    request_id = agent_state.context.get_context_value('request_id', None)
    kernel_id = f"{kernel_type}_{request_id or 'default'}"
    
    if kernel_id in kernel_managers:
        km = kernel_managers[kernel_id]
        if km.is_alive():
            return km
        else:
            # Cleanup dead kernel
            try:
                km.shutdown_kernel(now=True)
            except Exception:
                pass
            del kernel_managers[kernel_id]
    
    # Create new kernel with environment variables
    logger.info(f"Creating new {kernel_type} kernel: {kernel_id}")
    km = KernelManager(kernel_name=kernel_name)
    
    # Set working directory if provided
    if workspace:
        km.kernel_spec_manager = KernelSpecManager()
        km.cwd = workspace
    
    # Prepare environment variables
    kernel_env = os.environ.copy()  # Start with current environment
    
    # Add or override with extra environment variables
    if extra_env:
        kernel_env.update(extra_env)
    
    # Add some useful default variables
    kernel_env.update({
        'JUPYTER_KERNEL_TYPE': kernel_type,
        'JUPYTER_WORKSPACE': workspace or os.getcwd(),
    })
    
    # Start kernel with environment
    km.start_kernel(env=kernel_env)
    kernel_managers[kernel_id] = km
    
    # Update storage
    agent_state.set_global_value(KERNEL_MANAGERS_KEY, kernel_managers)
    
    # Wait for kernel to be ready and initialize
    client = km.client()
    client.start_channels()
    client.wait_for_ready(timeout=10)
    
    # Set working directory and any initialization
    if workspace:
        if kernel_type == 'python':
            init_code = f"import os; os.chdir('{workspace}')"
        else:  # bash
            init_code = f"cd '{workspace}'"
        
        client.execute(init_code, silent=True, store_history=False,
                      user_expressions={}, allow_stdin=False, stop_on_error=True)
        time.sleep(0.5)
    
    client.stop_channels()
    
    return km


def _cleanup_kernels(agent_state: 'AgentState', kernel_type: Optional[str] = None):
    """
    Cleanup kernels managed by this agent.
    
    Args:
        agent_state: AgentState containing kernel managers (required)
        kernel_type: Optional, cleanup only kernels of this type. If None, cleanup all.
    """
    kernel_managers = _get_kernel_managers(agent_state)
    
    for kernel_id, km in list(kernel_managers.items()):
        # If kernel_type is specified, only cleanup matching kernels
        if kernel_type and not kernel_id.startswith(f"{kernel_type}_"):
            continue
            
        try:
            if km.is_alive():
                logger.info(f"Shutting down kernel: {kernel_id}")
                km.shutdown_kernel(now=True)
        except Exception as e:
            logger.error(f"Error shutting down kernel {kernel_id}: {e}")
        finally:
            # Always remove from managers
            del kernel_managers[kernel_id]
    
    # Update storage
    agent_state.set_global_value(KERNEL_MANAGERS_KEY, kernel_managers)


def _process_output_message(msg: Dict[str, Any], kernel_type: str) -> Dict[str, Any]:
    """
    Process a Jupyter output message into a standardized format.
    
    Args:
        msg: Raw message from Jupyter kernel
        kernel_type: Type of kernel for context-specific processing
        
    Returns:
        Processed output dictionary
    """
    msg_type = msg['header']['msg_type']
    content = msg.get('content', {})
    
    if msg_type == 'stream':
        text = content.get('text', '')
        # Strip ANSI color codes
        text = _strip_ansi_colors(text)
        return {
            'type': 'stream',
            'name': content.get('name', 'stdout'),
            'text': text
        }
    
    elif msg_type == 'execute_result':
        result = {
            'type': 'execute_result'
        }
        
        # For bash kernel, extract text output if present and strip colors
        if kernel_type == 'bash' and 'text/plain' in content.get('data', {}):
            result['text'] = _strip_ansi_colors(content.get('data', {})['text/plain'])
            
        return result
    
    elif msg_type == 'display_data':
        result = {
            'type': 'display_data',
        }
        # Strip colors from text/plain if present
        if 'text/plain' in content.get('data', {}):
            result['text'] = _strip_ansi_colors(content.get('data', {})['text/plain'])
        
        return result
    
    elif msg_type == 'error':
        # Strip colors from traceback
        traceback = content.get('traceback', [])
        clean_traceback = [_strip_ansi_colors(line) for line in traceback]
        
        error_result = {
            'type': 'error',
            'ename': content.get('ename', 'Unknown'),
            'evalue': _strip_ansi_colors(content.get('evalue', '')),
            'traceback': clean_traceback
        }
        
        # Simplify bash errors if possible
        if kernel_type == 'bash' and error_result['traceback']:
            # Bash errors often have verbose tracebacks, try to extract the key error
            error_result['error_summary'] = error_result['evalue'] or 'Command failed'
            
        return error_result
    
    elif msg_type == 'execute_input':
        # Skip input echo
        return None
    
    elif msg_type in ['execute_reply', 'status']:
        # Skip status messages
        return None
    
    else:
        # Unknown message type
        return {
            'type': 'unknown',
            'msg_type': msg_type,
            'content': content
        }


def _aggregate_outputs(raw_outputs: list) -> list:
    """
    Aggregate consecutive stream outputs with the same name to reduce fragmentation.
    
    Args:
        raw_outputs: List of raw output messages
        
    Returns:
        List of aggregated outputs
    """
    if not raw_outputs:
        return []
    
    aggregated = []
    current_stream = None
    
    for output in raw_outputs:
        if output is None:
            continue
            
        if output['type'] == 'stream':
            stream_name = output['name']
            stream_text = output['text']
            
            # If this is the same stream as the current one, aggregate
            if (current_stream and 
                current_stream['type'] == 'stream' and 
                current_stream['name'] == stream_name):
                current_stream['text'] += stream_text
            else:
                # Start a new stream or finish the current one
                if current_stream:
                    aggregated.append(current_stream)
                current_stream = {
                    'type': 'stream',
                    'name': stream_name,
                    'text': stream_text
                }
        else:
            # Non-stream output, finish current stream if any
            if current_stream:
                aggregated.append(current_stream)
                current_stream = None
            aggregated.append(output)
    
    # Don't forget the last stream if any
    if current_stream:
        aggregated.append(current_stream)
    
    return aggregated


def run_code_tool(
    code_block: str,
    kernel_type: Literal['python', 'bash'] = 'python',
    timeout: Optional[int] = None,
    description: Optional[str] = None,
    reset_kernel: bool = False,
    shutdown_after: bool = False,
    agent_state: 'AgentState' = None,
) -> dict[str, Any]:
    """
    Execute code in a Jupyter notebook kernel (Python or Bash) with proper handling and output capture.

    Args:
        code_block: The code block to execute (required)
        kernel_type: Type of kernel to use ('python' or 'bash', default: 'python')
        timeout: Optional timeout in milliseconds (max 600000ms / 10 minutes)
        description: Clear, concise description of what this code does in 5-10 words
        reset_kernel: Whether to restart the kernel before execution
        shutdown_after: Whether to shutdown all kernels after execution
        agent_state: AgentState containing agent context and global storage (required)

    Returns:
        Dict containing execution results including outputs, errors, and metadata
        
    Examples:
        # Python kernel (default)
        result = jupyter_notebook_tool(
            code_block="import pandas as pd\\nprint('Hello')",
            kernel_type='python',
            agent_state=agent_state
        )
        
        # Bash kernel
        result = jupyter_notebook_tool(
            code_block="ls -la | grep '.txt'",
            kernel_type='bash',
            agent_state=agent_state
        )
    """
    if not agent_state:
        raise ValueError("agent_state is required for jupyter_notebook_tool")
    
    # Validate kernel type
    if kernel_type not in SUPPORTED_KERNELS:
        return {
            'status': 'error',
            'error': f"Unsupported kernel type: {kernel_type}. Supported types: {list(SUPPORTED_KERNELS.keys())}",
            'duration_ms': 0,
        }
    
    start_time = time.time()
    
    workspace = agent_state.get_global_value('workspace', None)
    
    # Validate timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    elif timeout > MAX_TIMEOUT:
        return {
            'status': 'error',
            'error': f"Timeout cannot exceed {MAX_TIMEOUT}ms (10 minutes)",
            'duration_ms': 0,
        }
    
    # Convert timeout to seconds
    timeout_seconds = timeout / 1000.0
    
    # Validate code
    if not code_block or not code_block.strip():
        return {
            'status': 'error',
            'error': 'Code block cannot be empty',
            'duration_ms': int((time.time() - start_time) * 1000),
        }
    
    try:
        # Get or create kernel
        km = _get_or_create_kernel(kernel_type, workspace, agent_state)
        
        # Reset kernel if requested
        if reset_kernel:
            logger.info(f"Restarting {kernel_type} kernel")
            km.restart_kernel(now=False)
            # Wait for kernel to be ready after restart
            client = km.client()
            client.start_channels()
            client.wait_for_ready(timeout=10)
            
            # Re-set working directory after restart if needed
            if workspace:
                if kernel_type == 'python':
                    init_code = f"import os; os.chdir('{workspace}')"
                else:  # bash
                    init_code = f"cd '{workspace}'"
                
                client.execute(
                    init_code,
                    silent=True,
                    store_history=False,
                    user_expressions={},
                    allow_stdin=False,
                    stop_on_error=True,
                )
                time.sleep(0.5)
        else:
            client = km.client()
            client.start_channels()
        
        # Execute the code
        msg_id = client.execute(
            code_block,
            silent=False,
            store_history=True,
            user_expressions={},
            allow_stdin=False,
            stop_on_error=True,
        )
        
        # Collect outputs
        raw_outputs = []
        status = 'success'
        error_info = None
        
        # Wait for execution to complete
        deadline = time.time() + timeout_seconds
        
        while True:
            try:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    # Timeout occurred
                    client.stop_channels()
                    duration_ms = int((time.time() - start_time) * 1000)
                    
                    # Cleanup if requested
                    if shutdown_after:
                        _cleanup_kernels(agent_state)
                    
                    return {
                        'status': 'timeout',
                        'error': f"Code execution timed out after {timeout}ms",
                        'kernel_type': kernel_type,
                        'duration_ms': duration_ms,
                        'outputs': _aggregate_outputs(raw_outputs),
                    }
                
                # Get message with timeout
                msg = client.get_iopub_msg(timeout=min(1, remaining_time))
                
                # Check if this message is for our execution
                if msg['parent_header'].get('msg_id') != msg_id:
                    continue
                
                # Process the message
                processed = _process_output_message(msg, kernel_type)
                if processed:
                    if processed['type'] == 'error':
                        status = 'error'
                        error_info = processed
                    
                    raw_outputs.append(processed)
                
                # Check if execution is complete
                if msg['header']['msg_type'] == 'status' and \
                   msg['content']['execution_state'] == 'idle':
                    break
                    
            except Empty:
                # No message available, continue waiting
                continue
            
        client.stop_channels()
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Aggregate outputs to reduce fragmentation
        outputs = _aggregate_outputs(raw_outputs)
        
        # Prepare result
        result = {
            'status': status,
            'kernel_type': kernel_type,
            'duration_ms': duration_ms
        }
        
        # Add description if provided
        if description:
            result['description'] = description
        
        # Add workspace if set
        if workspace:
            result['working_directory'] = workspace
        
        # Process outputs
        if outputs:
            # Truncate if needed
            outputs_str = json.dumps(outputs, default=str, ensure_ascii=False)
            if len(outputs_str) > MAX_OUTPUT_LENGTH:
                result['outputs'] = outputs
                result['outputs_truncated'] = True
                result['outputs_original_length'] = len(outputs_str)
            else:
                result['outputs'] = outputs
                result['outputs_truncated'] = False
        else:
            result['outputs'] = []
            result['outputs_truncated'] = False
        
        # Add error details if present
        if error_info:
            result['error_type'] = error_info.get('ename', 'Unknown')
            result['error_value'] = error_info.get('evalue', '')
            result['traceback'] = error_info.get('traceback', [])
            if 'error_summary' in error_info:
                result['error_summary'] = error_info['error_summary']
        
        # Cleanup if requested
        if shutdown_after:
            _cleanup_kernels(agent_state)
            result['kernels_cleaned'] = True
        
        # Log execution
        logger.info(
            f"Jupyter {kernel_type} code executed (status={status}, duration={duration_ms}ms)",
        )
        
        return result
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Cleanup if requested even on error
        if shutdown_after:
            _cleanup_kernels(agent_state)
        
        return {
            'status': 'error',
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__,
            'kernel_type': kernel_type,
            'duration_ms': duration_ms,
        }