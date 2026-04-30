@echo off
REM RFC-0019: Windows run-agent compatibility wrapper
uv run python -m nexau.cli.run_agent %*
