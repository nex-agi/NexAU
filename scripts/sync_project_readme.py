#!/usr/bin/env python3
import json
import subprocess
import sys
from typing import Any

OWNER = "china-qijizhifeng"
PROJECT_NUM = "5"
PROJECT_ID = "PVT_kwDOCmnEdM4BQARn"

REPO_META = {
    "nexau": {
        "role": "NexAU 核心框架",
        "dependency": "核心框架，其他两个 repo 依赖它",
    },
    "nexau-cloud-runtime": {
        "role": "Cloud 平台",
        "dependency": "通过 git submodule 引用 `nexau`",
    },
    "north-coder": {
        "role": "桌面端编程助手",
        "dependency": "backend 通过 pip 依赖引用 `nexau`",
    },
}

VIEW_PURPOSE = {
    "Backlog": "待排期 issue（`status:Backlog, no:iteration`）",
    "Iteration board": "当前迭代看板，按 Status × Repo 看进展",
    "Iteration table": "当前迭代列表，适合批量编辑",
    "Roadmap": "按模块查看整体排期",
    "My items": "个人任务视图",
    "PR": "PR 追踪视图",
    "数据局项目 MAP 平台融合": "特定项目路线图",
}

LAYOUT_NAME = {
    "TABLE_LAYOUT": "Table",
    "BOARD_LAYOUT": "Board",
    "ROADMAP_LAYOUT": "Roadmap",
}

WORKFLOW_NOTE = {
    "Auto-add nexau to project": "nexau 新 issue 自动入表",
    "Auto-add nexau-cloud-runtime to project": "cloud-runtime 新 issue 自动入表",
    "Auto-add nexau-coder to project": "north-coder 新 issue 自动入表",
    "Auto-add Gitagents to project": "Gitagents 新 issue 自动入表",
    "Auto-add sub-issues to project": "子 issue 自动入表",
    "Item added to project": "新 item 默认设为 Backlog",
    "Item reopened": "reopen 后自动设为 In progress",
    "Pull request merged": "PR 合并后自动联动状态",
    "Auto-close issue": "父 issue 自动关闭",
    "Item closed": "关闭状态由 repo workflow 接管，避免把 `not_planned/duplicate` 误写成 Done",
}

WORKFLOW_ORDER = [
    "Auto-add nexau to project",
    "Auto-add nexau-cloud-runtime to project",
    "Auto-add nexau-coder to project",
    "Auto-add Gitagents to project",
    "Auto-add sub-issues to project",
    "Item added to project",
    "Item reopened",
    "Pull request merged",
    "Auto-close issue",
    "Item closed",
]


def run(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout


def gh_json(*args: str) -> Any:
    return json.loads(run("gh", *args))


def get_project_data() -> dict[str, Any]:
    query = f'''
    {{
      node(id: "{PROJECT_ID}") {{
        ... on ProjectV2 {{
          title
          url
          views(first: 20) {{
            nodes {{
              ... on ProjectV2View {{
                name
                number
                layout
              }}
            }}
          }}
          workflows(first: 20) {{
            nodes {{
              id
              name
              enabled
              number
            }}
          }}
          fields(first: 30) {{
            nodes {{
              ... on ProjectV2IterationField {{
                id
                name
                configuration {{
                  iterations {{ id title startDate duration }}
                  completedIterations {{ id title startDate duration }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}'''
    data = gh_json("api", "graphql", "-f", f"query={query}")
    return data["data"]["node"]


def latest_tag(repo: str) -> str:
    data = gh_json("api", f"repos/{OWNER}/{repo}/tags?per_page=1")
    return data[0]["name"] if data else "-"


def normalize_sprint_title(title: str) -> str:
    if title.startswith("Sprint") and not title.startswith("Sprint "):
        return title.replace("Sprint", "Sprint ", 1)
    return title


def render() -> str:
    project = get_project_data()
    views = sorted(project["views"]["nodes"], key=lambda x: x["number"])
    workflows = {wf["name"]: wf for wf in project["workflows"]["nodes"]}

    iteration_field = None
    for field in project["fields"]["nodes"]:
        if field and field.get("name") == "Iteration":
            iteration_field = field
            break
    if not iteration_field:
        raise RuntimeError("Iteration field not found")

    current_iterations = iteration_field["configuration"]["iterations"]
    current_sprint = normalize_sprint_title(current_iterations[0]["title"]) if current_iterations else "-"
    next_sprint = normalize_sprint_title(current_iterations[1]["title"]) if len(current_iterations) > 1 else "-"

    repo_rows = []
    for repo, meta in REPO_META.items():
        repo_rows.append(f"| [{repo}](https://github.com/{OWNER}/{repo}) | {meta['role']} | `{latest_tag(repo)}` | {meta['dependency']} |")

    view_rows = []
    for view in views:
        name = view["name"]
        layout = LAYOUT_NAME.get(view["layout"], view["layout"])
        purpose = VIEW_PURPOSE.get(name, "-")
        view_rows.append(f"| {view['number']} | **{name}** | {layout} | {purpose} |")

    workflow_rows = []
    for name in WORKFLOW_ORDER:
        wf = workflows.get(name)
        if not wf:
            continue
        status = "Enabled" if wf["enabled"] else "Disabled"
        note = WORKFLOW_NOTE.get(name, "-")
        workflow_rows.append(f"| {name} | {status} | {note} |")

    return f"""# {project["title"]}

NexAU 产品线统一研发看板，覆盖 `nexau` / `nexau-cloud-runtime` / `north-coder` 三个 repo 的 backlog、iteration、PR、roadmap 管理。

项目地址：<{project["url"]}>

## 当前节奏

- **当前 Sprint**：`{current_sprint}`
- **下一个 Sprint**：`{next_sprint}`
- **迭代节奏**：每周一个 Sprint（**周一 ~ 周日**）

## 关联仓库

| Repo | 角色 | 最新 Tag | 依赖关系 |
|------|------|----------|----------|
{chr(10).join(repo_rows)}

## 主要视图

| # | View | Layout | 用途 |
|---|------|--------|------|
{chr(10).join(view_rows)}

## 字段约定

### Status

`Backlog → Todo → In progress → Done / Blocked / Paused / Cancelled`

- **Backlog**：待排期
- **Todo**：已排入迭代，待开始
- **In progress**：进行中
- **Done**：已完成
- **Blocked**：被依赖阻塞
- **Paused**：主动暂停
- **Cancelled**：明确不做 / 重复关闭

### Close reason → Status 映射

- `completed` → **Done**
- `not_planned` → **Cancelled**
- `duplicate` → **Cancelled**
- `reopened` → **In progress**

### 其他字段

- **Iteration**：周维度 Sprint 字段
- **Priority**：P0 / P1 / P2 / P3
- **Module**：按模块聚合 Roadmap 与统计

## 自动化规则

### Project 内置 Workflows

| Workflow | 状态 | 动作 |
|----------|------|------|
{chr(10).join(workflow_rows)}

### Repo Actions Workflows

#### 1. Backlog / Todo / Done 状态同步
- 文件：`.github/workflows/project-auto-status.yml`
- 触发：每 10 分钟轮询
- 逻辑：
  - issue 已分配 Iteration 且仍为 Backlog → 自动改为 Todo
  - Todo 但 Iteration 被清空 → 自动改回 Backlog
  - Project Status 已被手动设为 Done，但 issue 仍是 open → 自动 `close as completed`

#### 2. Close reason 自动映射状态
- 文件：`.github/workflows/project-auto-status-on-close.yml`
- 触发：`issues.closed`
- 逻辑：
  - `completed` → Done
  - `not_planned` / `duplicate` → Cancelled

#### 3. Milestone 自动管理
- 文件：`.github/workflows/milestone-auto-manage.yml`
- 触发：推送版本 tag
- 逻辑：自动创建 / 关联 / 关闭 Milestone，并补下一个版本的 Milestone

## README 维护策略

- **每周五 22:00（Asia/Shanghai / UTC+8）**检查一次 Project README
- 若 repo/tag、views、iteration 节奏、workflow 规则或模板发生变化，则自动更新 Project Settings 中的 README
- README sync workflow 托管在 repo Actions 中执行

## Secrets / 运维注意事项

三个 repo 都需要配置：

| Secret | 说明 |
|--------|------|
| `PROJECT_PAT` | 需要 `project` read/write 权限，用于 Project 字段更新与 README 同步 |

如果发现「关闭 issue 后状态不对」或「自动化没生效」，优先检查：

1. Project Settings → Workflows 中 **`Item closed`** 是否保持 **Disabled**
2. 三个 repo 的 **`PROJECT_PAT`** 是否过期 / 权限不足
3. 对应 workflow run 是否报 `Could not resolve to a node` / project permission 错误
"""


if __name__ == "__main__":
    sys.stdout.write(render())
