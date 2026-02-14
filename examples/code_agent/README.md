# Code Agent Examples

提供两种 code agent 配置：`code_agent`（完整版）和 `code_agent_sync`（同步版）。

## 对比

| 特性 | code_agent | code_agent_sync |
|------|------------|-----------------|
| **Shell 执行模式** | 支持前台 + 后台 | 仅前台（同步） |
| **is_background 参数** | 有，可传 `true` 后台运行 | 无，所有命令同步执行 |
| **BackgroundTaskManage** | 有，可查询/终止后台任务 | 无 |
| **适用场景** | 开发服务器、watch、长跑任务 | build、test、脚本、一次性命令 |

## 使用建议

- **code_agent**：需要启动 dev server、file watcher、或其它长跑进程时使用。
- **code_agent_sync**：纯 build/test、一次性脚本、CI 类任务时使用，更简单、无后台任务管理。

## 运行方式

```bash
# 完整版（支持后台）
nexau run code_agent

# 同步版（仅前台）
nexau run code_agent_sync
```

## 环境变量

需设置 `LLM_MODEL`、`LLM_BASE_URL`、`LLM_API_KEY`。本地运行建议设置 `SANDBOX_WORK_DIR`：

```bash
export SANDBOX_WORK_DIR=/tmp/code_agent_work
```
