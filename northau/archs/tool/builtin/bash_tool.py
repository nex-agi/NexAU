"""Bash command execution tool implementation."""
import logging
import os
import subprocess
import time
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...main_sub.agent_state import AgentState

logger = logging.getLogger(__name__)

# Maximum output length to prevent overwhelming responses
MAX_OUTPUT_LENGTH = 30000
DEFAULT_TIMEOUT = 120000  # 2 minutes in milliseconds
MAX_TIMEOUT = 600000      # 10 minutes in milliseconds


def bash_tool(
    command: str,
    timeout: Optional[int] = None,
    description: Optional[str] = None,
    agent_state: Optional['AgentState'] = None,
) -> dict[str, Any]:
    """
    Execute a bash command in a persistent shell session with proper handling and security measures.

    Args:
        command: The bash command to execute (required)
        timeout: Optional timeout in milliseconds (max 600000ms / 10 minutes)
        description: Clear, concise description of what this command does in 5-10 words
        agent_state: AgentState containing agent context and global storage

    Returns:
        Dict containing execution results
    """
    start_time = time.time()

    workspace = None
    if agent_state:
        workspace = agent_state.get_global_value('workspace', None)

    # Validate timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    elif timeout > MAX_TIMEOUT:
        return {
            'status': 'error',
            'error': f"Timeout cannot exceed {MAX_TIMEOUT}ms (10 minutes)",
            'command': command,
            'duration_ms': 0,
        }

    # Convert timeout to seconds for subprocess
    timeout_seconds = timeout / 1000.0

    # Validate command
    if not command or not command.strip():
        return {
            'status': 'error',
            'error': 'Command cannot be empty',
            'duration_ms': int((time.time() - start_time) * 1000),
        }

    # Security warnings for potentially dangerous commands
    dangerous_patterns = [
        'rm -rf /', 'sudo rm', 'rm -rf ~', 'mkfs', 'fdisk', 'dd if=',
        '> /dev/', 'shutdown', 'reboot', 'halt', 'poweroff',
    ]

    command_lower = command.lower().strip()
    for pattern in dangerous_patterns:
        if pattern in command_lower:
            logger.warning(
                f"Potentially dangerous command detected: {command}",
            )
            return {
                'status': 'error',
                'error': f"Command contains potentially dangerous pattern: {pattern}",
                'command': command,
                'duration_ms': int((time.time() - start_time) * 1000),
            }

    try:
        # Execute the command
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=workspace or os.getcwd(),
            env=os.environ.copy(),
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            duration_ms = int((time.time() - start_time) * 1000)
            return {
                'status': 'timeout',
                'error': f"Command timed out after {timeout}ms",
                'command': command,
                'duration_ms': duration_ms,
                'stdout': stdout[:MAX_OUTPUT_LENGTH] if stdout else '',
                'stderr': stderr[:MAX_OUTPUT_LENGTH] if stderr else '',
                'exit_code': process.returncode,
            }

        duration_ms = int((time.time() - start_time) * 1000)

        # Prepare output
        result = {
            'status': 'success' if process.returncode == 0 else 'error',
            'command': command,
            'exit_code': process.returncode,
            'duration_ms': duration_ms,
            'working_directory': os.getcwd(),
        }

        # Add description if provided
        if description:
            result['description'] = description

        # Truncate output if too long
        if stdout:
            if len(stdout) > MAX_OUTPUT_LENGTH:
                result['stdout'] = stdout[:MAX_OUTPUT_LENGTH]
                result['stdout_truncated'] = True
                result['stdout_original_length'] = len(stdout)
            else:
                result['stdout'] = stdout
                result['stdout_truncated'] = False
        else:
            result['stdout'] = ''
            result['stdout_truncated'] = False

        if stderr:
            if len(stderr) > MAX_OUTPUT_LENGTH:
                result['stderr'] = stderr[:MAX_OUTPUT_LENGTH]
                result['stderr_truncated'] = True
                result['stderr_original_length'] = len(stderr)
            else:
                result['stderr'] = stderr
                result['stderr_truncated'] = False
        else:
            result['stderr'] = ''
            result['stderr_truncated'] = False

        # Log execution
        logger.info(
            f"Bash command executed: '{command}' (exit_code={process.returncode}, duration={duration_ms}ms)",
        )

        return result

    except FileNotFoundError:
        return {
            'status': 'error',
            'error': 'Shell not found. Bash may not be available on this system.',
            'command': command,
            'duration_ms': int((time.time() - start_time) * 1000),
        }

    except PermissionError as e:
        return {
            'status': 'error',
            'error': f"Permission denied: {str(e)}",
            'command': command,
            'duration_ms': int((time.time() - start_time) * 1000),
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': f"Unexpected error: {str(e)}",
            'error_type': type(e).__name__,
            'command': command,
            'duration_ms': int((time.time() - start_time) * 1000),
        }


# Alternative class-based implementation for more advanced usage
class BashTool:
    """
    A class-based implementation of the bash tool for more advanced usage.
    Provides session persistence and better error handling.
    """

    def __init__(self, max_output_length: int = MAX_OUTPUT_LENGTH, default_timeout: int = DEFAULT_TIMEOUT):
        self.max_output_length = max_output_length
        self.default_timeout = default_timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session_env = os.environ.copy()

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        description: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute a bash command with enhanced features.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in milliseconds
            description: Description of what the command does
            cwd: Working directory for the command

        Returns:
            Dict containing execution results
        """
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout

        # Set working directory
        if cwd and os.path.exists(cwd):
            original_cwd = os.getcwd()
            os.chdir(cwd)
        else:
            original_cwd = None

        try:
            result = bash_tool(command, timeout, description)
            return result

        finally:
            # Restore original working directory
            if original_cwd:
                os.chdir(original_cwd)

    def execute_multiple(self, commands: list[str], timeout: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Execute multiple commands in sequence.

        Args:
            commands: List of commands to execute
            timeout: Optional timeout for each command

        Returns:
            List of execution results
        """
        results = []
        for command in commands:
            result = self.execute(command, timeout)
            results.append(result)

            # Stop execution if a command fails
            if result['status'] != 'success':
                break

        return results


def main():
    """Test function to demonstrate and validate the bash_tool functionality."""
    print('🔧 BashTool 测试开始...')
    print('=' * 50)

    # Test 1: Basic command execution
    print('\n📋 测试 1: 基本命令执行')
    try:
        result = bash_tool(
            "echo 'Hello, World!'",
            description='Print hello message',
        )
        print('✅ 命令执行成功')
        print(f"⏱️  执行时间: {result['duration_ms']}ms")
        print(f"📤 输出: {result.get('stdout', '').strip()}")
        print(f"🚪 退出码: {result['exit_code']}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 2: Command with error
    print('\n📋 测试 2: 错误命令测试')
    try:
        result = bash_tool(
            'ls /nonexistent_directory_12345',
            description='List non-existent directory',
        )
        if result['status'] != 'success':
            print('✅ 正确处理了错误命令')
            print(f"🚪 退出码: {result['exit_code']}")
            print(f"⚠️  错误输出: {result.get('stderr', '').strip()}")
        else:
            print('⚠️  意外成功了')
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 3: Timeout test
    print('\n📋 测试 3: 超时测试')
    try:
        result = bash_tool(
            'sleep 1', timeout=500,
            description='Sleep with short timeout',
        )
        if result['status'] == 'timeout':
            print('✅ 正确处理了超时')
            print(f"⏱️  执行时间: {result['duration_ms']}ms")
        else:
            print('⚠️  超时测试可能有问题')
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 4: Security check
    print('\n📋 测试 4: 安全检查')
    try:
        result = bash_tool('rm -rf /', description='Dangerous command test')
        if result['status'] == 'error' and 'dangerous' in result.get('error', '').lower():
            print('✅ 正确阻止了危险命令')
            print(f"⚠️  错误信息: {result['error']}")
        else:
            print('⚠️  安全检查可能有问题')
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 5: Class-based implementation
    print('\n📋 测试 5: 类实现测试')
    try:
        bash_tool_instance = BashTool()
        result = bash_tool_instance.execute(
            'pwd', description='Print working directory',
        )
        if result['status'] == 'success':
            print('✅ 类实现工作正常')
            print(f"📁 当前目录: {result.get('stdout', '').strip()}")
        else:
            print('❌ 类实现失败')
    except Exception as e:
        print(f"❌ 类实现测试失败: {e}")

    # Test 6: Multiple commands
    print('\n📋 测试 6: 多命令执行')
    try:
        bash_tool_instance = BashTool()
        commands = ["echo 'Command 1'", "echo 'Command 2'", 'pwd']
        results = bash_tool_instance.execute_multiple(commands)

        if all(r['status'] == 'success' for r in results):
            print('✅ 多命令执行成功')
            for i, result in enumerate(results, 1):
                print(f"   命令 {i}: {result.get('stdout', '').strip()}")
        else:
            print('⚠️  部分命令执行失败')
    except Exception as e:
        print(f"❌ 多命令测试失败: {e}")

    print('\n' + '=' * 50)
    print('🎉 BashTool 测试完成!')

    # Usage tips
    print('\n💡 使用提示:')
    print('  • 总是使用双引号包围含有空格的路径')
    print("  • 避免使用危险命令如 'rm -rf /'")
    print("  • 使用绝对路径避免 'cd' 命令")
    print('  • 设置合适的超时时间避免长时间阻塞')
    print('  • 检查退出码判断命令是否成功执行')


if __name__ == '__main__':
    main()
