# Windows Support

NexAU supports Windows 10 and Windows 11 with the default Windows shell backend.

The default backend is PowerShell, selected in this order:

1. `pwsh.exe`
2. `powershell.exe`
3. `cmd.exe`

Git Bash is optional. NexAU only requires Git Bash when you explicitly select the
bash-compatible backend or when a command needs bash-only syntax.

## Support Matrix

| Area | Supported | Notes |
| --- | --- | --- |
| Operating system | Windows 10, Windows 11 | Other Windows versions are not part of the public support statement. |
| Default backend | PowerShell | NexAU probes `pwsh.exe`, then `powershell.exe`, then `cmd.exe`. |
| Optional backend | Git Bash | Use only for explicit bash-compatible mode or bash-only commands. |
| Python CLI | `nexau` | Validated through Windows parser and runtime smoke checks. |
| Script entrypoint | `python -m nexau.cli.run_agent`, `run-agent.cmd` | `run-agent.cmd` is the native Windows wrapper. |
| npm entrypoint | `npm run agent`, `npm run run-agent` | Wraps the Python run-agent module. |
| Legacy wrapper | `nexau-cli` | Kept validated while it remains supported. |

This page does not define a detailed capability matrix for external tools such as
`rg`, `ffmpeg`, or Node.js. Those dependencies are handled by their owning
features and tests.

## Install and Run

Install from source with the same Python workflow used on other platforms:

```powershell
git clone git@github.com:nex-agi/NexAU.git
cd NexAU
pip install uv
uv sync
```

Install the JavaScript dependencies if you use npm or the legacy CLI wrapper:

```powershell
npm install
cd cli
npm install
cd ..
```

Run the main Python CLI:

```powershell
uv run nexau --help
uv run nexau chat examples/simple_research/main_agent.yaml --query "hello"
```

Run an agent through the Windows script wrapper:

```powershell
.\run-agent.cmd examples/code_agent/code_agent.yaml
```

Run through npm:

```powershell
npm run agent -- examples/code_agent/code_agent.yaml
```

## Shell Backends

### Default PowerShell Backend

On Windows, NexAU does not rely on the host shell through `shell=True`. It
creates an explicit backend and launches commands through the selected shell.
The default path keeps Windows-native paths such as `C:\Users\Public\demo` when
commands run through PowerShell or `cmd.exe`.

Use PowerShell syntax for commands sent to the default backend:

```powershell
Write-Output "hello from NexAU"
Get-Location
Select-String -Path .\README.md -Pattern NexAU
```

### Optional Git Bash Backend

Use Git Bash when you need bash-compatible syntax, POSIX-style paths, or
bash-only commands. Select it explicitly with `NEXAU_WINDOWS_SHELL_BACKEND`:

```powershell
$env:NEXAU_WINDOWS_SHELL_BACKEND = "git-bash"
uv run python -c "from nexau.archs.platform.shell_backend import create_shell_backend; print(type(create_shell_backend()).__name__)"
Remove-Item Env:\NEXAU_WINDOWS_SHELL_BACKEND
```

Accepted values include `default`, `powershell`, `pwsh`, `cmd`, and `git-bash`.

When Git Bash is selected, NexAU converts native Windows paths to Git
Bash-compatible paths, for example:

```text
C:\Users\Public\test path -> /c/Users/Public/test path
```

## Bash-Only Commands

The default PowerShell backend does not silently emulate bash syntax. If a
command uses bash-only constructs such as bash heredoc syntax, rewrite it for
PowerShell or select the optional Git Bash backend.

Examples that should use Git Bash or be rewritten:

```bash
cat <<EOF
hello
EOF
```

```bash
echo "out" && echo "err" >&2
```

PowerShell equivalents should be written with PowerShell syntax:

```powershell
@"
hello
"@
```

```powershell
Write-Output "out"; Write-Error "err"
```

## Self Checks

Check that the default Windows backend is available:

```powershell
uv run python -c "from nexau.cli.entrypoint_checks import ensure_default_windows_shell_for_entrypoint; print(ensure_default_windows_shell_for_entrypoint())"
```

Check which backend NexAU will use:

```powershell
uv run python -c "from nexau.archs.platform.shell_backend import create_shell_backend; print(type(create_shell_backend()).__name__)"
```

Check the optional Git Bash backend:

```powershell
$env:NEXAU_WINDOWS_SHELL_BACKEND = "git-bash"
uv run python -c "from nexau.archs.platform.shell_backend import create_shell_backend; print(type(create_shell_backend()).__name__)"
Remove-Item Env:\NEXAU_WINDOWS_SHELL_BACKEND
```

Run the focused platform tests:

```powershell
uv run pytest tests/unit/archs/platform tests/unit/test_cli_entrypoint_checks.py tests/unit/test_cli_run_agent.py tests/unit/test_cli_wrapper.py -q
```

## Troubleshooting

### Default startup fails before the agent starts

Run the default backend self check:

```powershell
uv run python -c "from nexau.cli.entrypoint_checks import ensure_default_windows_shell_for_entrypoint; print(ensure_default_windows_shell_for_entrypoint())"
```

If no Windows shell can be found, install PowerShell or repair the system
`PATH`. `cmd.exe` is only a last-resort diagnostic backend, not the recommended
daily shell.

### A bash command fails under the default backend

The command may use bash-only syntax. Rewrite it for PowerShell or set:

```powershell
$env:NEXAU_WINDOWS_SHELL_BACKEND = "git-bash"
```

Then retry the command. Remove the variable when you want to return to the
default backend:

```powershell
Remove-Item Env:\NEXAU_WINDOWS_SHELL_BACKEND
```

### Explicit Git Bash mode reports a missing dependency

Install Git for Windows, make sure `bash.exe` is discoverable, and retry the same
NexAU command. NexAU fails fast in explicit Git Bash mode when Git Bash is
missing or fails its health check; it does not run an interactive installer.

### A command works in PowerShell but fails from a one-line test command

PowerShell expands `$...` expressions in double-quoted strings before Python or
the shell backend sees them. When a self-check embeds Python code that contains
PowerShell variables, prefer a single-quoted here-string piped to `uv run python
-` so the command is not expanded too early.
