import type { TaskInfo } from "../types";
import {
  LEADER_ID, RFC_WRITER_ID,
  BUILDER_DB_ID, BUILDER_SKILL_ID, BUILDER_CFG_ID,
  Phase, makeTask,
} from "./mockConstants";

// ---------------------------------------------------------------------------
// 各阶段的 Task 快照
// ---------------------------------------------------------------------------
export const MOCK_TASKS: Record<Phase, TaskInfo[]> = {
  // ── Phase 0: 需求确认 ────────────────────────────────────────────
  [Phase.REQUIREMENTS]: [
    makeTask({
      task_id: "REQ-001", title: "需求确认与澄清",
      description: "确认数据库类型、分析需求范围、数据字典",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: LEADER_ID,
    }),
  ],

  // ── Phase 1: ask_user 暂停（与 Phase 0 相同）────────────────────
  [Phase.ASK_USER_1]: [
    makeTask({
      task_id: "REQ-001", title: "需求确认与澄清",
      description: "确认数据库类型、分析需求范围、数据字典",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: LEADER_ID,
    }),
  ],

  // ── Phase 2: 设计文档 ────────────────────────────────────────────
  [Phase.DESIGN]: [
    makeTask({
      task_id: "REQ-001", title: "需求确认与澄清",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: LEADER_ID,
      result_summary: "PostgreSQL 数据库, 4 类分析需求已确认",
    }),
    makeTask({
      task_id: "RFC-001", title: "经信委智能体设计文档",
      description: "编写智能体架构和 Skills 设计方案 (RFC-0001)",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: RFC_WRITER_ID,
      deliverable_path: ".nexau/tasks/RFC-001-design-doc.md",
    }),
  ],

  // ── Phase 3: ask_user 审批暂停 ────────────────────────────────
  [Phase.ASK_USER_2]: [
    makeTask({
      task_id: "REQ-001", title: "需求确认与澄清",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: LEADER_ID,
      result_summary: "PostgreSQL 数据库, 4 类分析需求已确认",
    }),
    makeTask({
      task_id: "RFC-001", title: "经信委智能体设计文档",
      description: "等待用户审批设计方案",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: RFC_WRITER_ID,
      deliverable_path: "rfcs/0001-jxw-agent.md",
    }),
  ],

  // ── Phase 4: 实施阶段 ────────────────────────────────────────────
  [Phase.IMPLEMENTATION]: [
    makeTask({
      task_id: "REQ-001", title: "需求确认与澄清",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: LEADER_ID,
      result_summary: "PostgreSQL 数据库, 4 类分析需求已确认",
    }),
    makeTask({
      task_id: "RFC-001", title: "经信委智能体设计文档",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: RFC_WRITER_ID,
      result_summary: "RFC-0001 设计方案已审批通过",
      deliverable_path: "rfcs/0001-jxw-agent.md",
    }),
    makeTask({
      task_id: "SKILL-001", title: "企业信息查询 Skill",
      description: "实现按名称/行业/地区多维度企业检索",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: BUILDER_DB_ID,
      deliverable_path: "skills/enterprise_query/",
    }),
    makeTask({
      task_id: "SKILL-002", title: "经营分析 Skill",
      description: "实现营收、利润、增长趋势分析",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: BUILDER_SKILL_ID,
      deliverable_path: "skills/business_analysis/",
    }),
    makeTask({
      task_id: "SKILL-003", title: "政策匹配 Skill",
      description: "根据企业特征智能匹配适用政策",
      status: "pending", created_by: LEADER_ID, assignee_agent_id: BUILDER_SKILL_ID,
      dependencies: ["SKILL-002"],
      deliverable_path: "skills/policy_match/",
    }),
    makeTask({
      task_id: "CONFIG-001", title: "智能体配置与启动脚本",
      description: "编写 nexau.yaml 配置和 run.py 启动入口",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: BUILDER_CFG_ID,
      deliverable_path: "nexau.yaml",
    }),
    makeTask({
      task_id: "BUILD-001", title: "启动脚本与入口",
      description: "编写 run.py 和 Dockerfile",
      status: "pending", created_by: LEADER_ID, assignee_agent_id: BUILDER_CFG_ID,
      dependencies: ["CONFIG-001"],
      deliverable_path: "run.py",
    }),
  ],

  // ── Phase 5: 全部完成 ────────────────────────────────────────────
  [Phase.COMPLETE]: [
    makeTask({
      task_id: "REQ-001", title: "需求确认与澄清",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: LEADER_ID,
      result_summary: "PostgreSQL 数据库, 4 类分析需求已确认",
    }),
    makeTask({
      task_id: "RFC-001", title: "经信委智能体设计文档",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: RFC_WRITER_ID,
      result_summary: "RFC-0001 设计方案已审批通过",
      deliverable_path: "rfcs/0001-jxw-agent.md",
    }),
    makeTask({
      task_id: "SKILL-001", title: "企业信息查询 Skill",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: BUILDER_DB_ID,
      result_summary: "enterprise_query skill 已完成，含 3 个 tool",
      deliverable_path: "skills/enterprise_query/",
    }),
    makeTask({
      task_id: "SKILL-002", title: "经营分析 Skill",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: BUILDER_SKILL_ID,
      result_summary: "business_analysis skill 已完成，含 2 个 tool",
      deliverable_path: "skills/business_analysis/",
    }),
    makeTask({
      task_id: "SKILL-003", title: "政策匹配 Skill",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: BUILDER_SKILL_ID,
      result_summary: "policy_match skill 已完成，含 2 个 tool",
      deliverable_path: "skills/policy_match/",
    }),
    makeTask({
      task_id: "CONFIG-001", title: "智能体配置与启动脚本",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: BUILDER_CFG_ID,
      result_summary: "nexau.yaml 配置已完成",
      deliverable_path: "nexau.yaml",
    }),
    makeTask({
      task_id: "BUILD-001", title: "启动脚本与入口",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: BUILDER_CFG_ID,
      result_summary: "run.py 已完成并通过集成测试",
      deliverable_path: "run.py",
    }),
    makeTask({
      task_id: "TEST-001", title: "集成测试与验收",
      description: "运行端到端测试验证所有 Skills",
      status: "completed", created_by: LEADER_ID, assignee_agent_id: BUILDER_DB_ID,
      result_summary: "全部 7 项测试通过 ✓",
    }),
  ],
};

// ---------------------------------------------------------------------------
// IMPLEMENTATION 阶段的中间 Task 快照（嵌入到事件中做渐进更新）
// ---------------------------------------------------------------------------
export const IMPL_TASK_SNAPSHOTS: TaskInfo[][] = [
  // 快照 1: SKILL-001 完成, SKILL-003 开始
  [
    ...MOCK_TASKS[Phase.IMPLEMENTATION].map((t) => {
      if (t.task_id === "SKILL-001") return { ...t, status: "completed", result_summary: "enterprise_query skill 已完成" };
      if (t.task_id === "SKILL-003") return { ...t, status: "in_progress", is_blocked: false };
      return t;
    }),
  ],
  // 快照 2: SKILL-002 完成, CONFIG-001 完成, BUILD-001 开始
  [
    ...MOCK_TASKS[Phase.IMPLEMENTATION].map((t) => {
      if (t.task_id === "SKILL-001") return { ...t, status: "completed", result_summary: "enterprise_query skill 已完成" };
      if (t.task_id === "SKILL-002") return { ...t, status: "completed", result_summary: "business_analysis skill 已完成" };
      if (t.task_id === "SKILL-003") return { ...t, status: "in_progress", is_blocked: false };
      if (t.task_id === "CONFIG-001") return { ...t, status: "completed", result_summary: "nexau.yaml 配置已完成" };
      if (t.task_id === "BUILD-001") return { ...t, status: "in_progress", is_blocked: false };
      return t;
    }),
  ],
  // 快照 3: 全部 SKILL/CONFIG/BUILD 完成, TEST-001 开始
  [
    ...MOCK_TASKS[Phase.IMPLEMENTATION].map((t) => {
      if (t.task_id === "SKILL-001") return { ...t, status: "completed", result_summary: "enterprise_query skill 已完成" };
      if (t.task_id === "SKILL-002") return { ...t, status: "completed", result_summary: "business_analysis skill 已完成" };
      if (t.task_id === "SKILL-003") return { ...t, status: "completed", result_summary: "policy_match skill 已完成" };
      if (t.task_id === "CONFIG-001") return { ...t, status: "completed", result_summary: "nexau.yaml 配置已完成" };
      if (t.task_id === "BUILD-001") return { ...t, status: "completed", result_summary: "run.py 已完成" };
      return t;
    }),
    makeTask({
      task_id: "TEST-001", title: "集成测试与验收",
      description: "运行端到端测试验证所有 Skills",
      status: "in_progress", created_by: LEADER_ID, assignee_agent_id: BUILDER_DB_ID,
    }),
  ],
];
