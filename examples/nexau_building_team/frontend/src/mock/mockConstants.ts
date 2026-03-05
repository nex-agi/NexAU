import type { TaskInfo } from "../types";

// ---------------------------------------------------------------------------
// Agent IDs & Roles
// ---------------------------------------------------------------------------
export const TEAM_ID = "demo-team";
export const RUN_ID = "run-demo";

export const LEADER_ID = "leader-001";
export const LEADER_ROLE = "team_leader";

export const RFC_WRITER_ID = "rfc-writer-001";
export const RFC_WRITER_ROLE = "rfc_writer";

export const BUILDER_DB_ID = "builder-db-001";
export const BUILDER_DB_ROLE = "builder_engineer";

export const BUILDER_SKILL_ID = "builder-skill-001";
export const BUILDER_SKILL_ROLE = "builder_engineer";

export const BUILDER_CFG_ID = "builder-cfg-001";
export const BUILDER_CFG_ROLE = "builder_engineer";

// ---------------------------------------------------------------------------
// Phase 枚举
// ---------------------------------------------------------------------------
export enum Phase {
  /** Leader 分析需求，创建 TODO，ask_user 需求澄清 */
  REQUIREMENTS = 0,
  /** 暂停：等待用户回答需求问题 */
  ASK_USER_1 = 1,
  /** Leader 创建任务 + spawn RFC Writer → RFC Writer 写设计文档 → ask_user 审批 */
  DESIGN = 2,
  /** 暂停：等待用户审批设计文档 */
  ASK_USER_2 = 3,
  /** Leader spawn 3 Builders → 并行探索 DB + 设计 Skills + 写配置 */
  IMPLEMENTATION = 4,
  /** Leader 总结，整个流程完成 */
  COMPLETE = 5,
}

// ---------------------------------------------------------------------------
// Helper: 构造 TaskInfo
// ---------------------------------------------------------------------------
export function makeTask(
  overrides: Partial<TaskInfo> &
    Pick<TaskInfo, "task_id" | "title" | "status" | "created_by">,
): TaskInfo {
  return {
    description: "",
    priority: 1,
    dependencies: [],
    assignee_agent_id: null,
    result_summary: null,
    deliverable_path: null,
    is_blocked: false,
    created_at: "2026-03-02T10:00:00Z",
    updated_at: "2026-03-02T10:00:00Z",
    ...overrides,
  };
}
