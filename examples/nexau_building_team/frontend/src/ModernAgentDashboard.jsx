import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  User, Cpu, PenTool, Wrench, ShieldCheck,
  CheckCircle2, Circle, Loader2, Play, RotateCcw,
  MessageSquare, Box, Layers, Zap, AlertTriangle, Users,
  FileJson, Terminal, Puzzle, Rocket
} from 'lucide-react';

// --- 全局动画与样式 (Palantir 轻量科技风) ---
const styles = `
  .wireframe-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
  }

  .line-flow {
    stroke-dasharray: 6;
    animation: flow 1s linear infinite;
  }
  @keyframes flow {
    to { stroke-dashoffset: -12; }
  }

  .node-spawn {
    animation: spawn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  @keyframes spawn {
    0% { transform: translate(-50%, -50%) scale(0.8); opacity: 0; }
    100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
  }

  .pulse-ring {
    position: absolute;
    inset: -5px;
    border-radius: inherit;
    border: 1px solid currentColor;
    opacity: 0;
    animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;
  }
  @keyframes ping {
    75%, 100% { transform: scale(1.4); opacity: 0; }
    0% { transform: scale(1); opacity: 0.4; }
  }

  .artifact-part-reveal {
    animation: revealPart 0.6s ease-out forwards;
  }
  @keyframes revealPart {
    0% { clip-path: inset(100% 0 0 0); }
    100% { clip-path: inset(0 0 0 0); }
  }

  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 0px; }
  ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
`;

export default function ModernAgentDashboard() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [interaction, setInteraction] = useState(null);

  // 动态控制 Builder 数量
  const [builderCount, setBuilderCount] = useState(3);

  // 1. 动态节点状态 (适配亮色主题的高对比度色彩)
  const [agents, setAgents] = useState({
    user: { id: 'user', role: 'User', name: 'You', status: 'idle', color: '#059669', icon: User },
    leader: { id: 'leader', role: 'Leader', name: 'Nexus-Prime', status: 'idle', color: '#6366f1', icon: Cpu },
  });

  const [tasks, setTasks] = useState([]);
  const [messages, setMessages] = useState([]);
  const messagesEndRef = useRef(null);

  // 3. 动态 Artifact 构建状态 (更新为 NexAU Agent 结构 + 启动脚本)
  const [artifactState, setArtifactState] = useState({
    systemPrompt: false,
    agentConfig: false,
    toolConfig: false,
    startupScript: false,
    skills: {}
  });

  // 4. 端到端测试 (E2E) 状态
  const [e2eStats, setE2eStats] = useState({ total: 142, passed: 0, status: 'idle' });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- 核心调度工具函数 ---
  const spawnAgent = (id, role, name, color, icon) => {
    setAgents(prev => ({ ...prev, [id]: { id, role, name, status: 'idle', color, icon } }));
  };

  const updateAgent = (id, status) => {
    setAgents(prev => prev[id] ? { ...prev, [id]: { ...prev[id], status } } : prev);
  };

  const addMessage = (fromId, toId, text, type = 'normal') => {
    setMessages(prev => [...prev, { id: Date.now() + Math.random(), fromId, toId, text, type }]);
  };

  const addTask = (id, title, assigneeId) => {
    setTasks(prev => [...prev, { id, title, assigneeId, status: 'pending' }]);
  };

  const updateTaskStatus = (id, status) => {
    setTasks(prev => prev.map(t => t.id === id ? { ...t, status } : t));
  };

  const completeArtifactPart = (type, id = null) => {
    setArtifactState(prev => {
      if (type === 'skill') return { ...prev, skills: { ...prev.skills, [id]: true } };
      return { ...prev, [type]: true };
    });
  };

  // --- 动态剧本生成器 ---
  const getScript = () => {
    const s = [];

    // Step 0: User 触发
    s.push({
      delay: 500, action: () => {
        addMessage('user', 'leader', `初始化 NexAU Agent 项目。我们需要 ${builderCount} 个 SKILL 并行开发。`);
        updateAgent('user', 'working'); updateAgent('leader', 'working');
      }
    });

    // Step 1: Leader 分配任务
    s.push({
      delay: 1500, action: () => {
        updateAgent('user', 'idle');
        addMessage('leader', 'user', '收到指令。正在拆解 NexAU Agent 组件并组建 Swarm 团队...', 'system');
        addTask('t_rfc', 'System Prompt & Config', 'rfc');
        for(let i=1; i<=builderCount; i++) {
          addTask(`t_b${i}`, `开发 SKILL: 功能模块 0${i}`, `b${i}`);
        }
        addTask('t_qa', 'Tool Config 集成验证', 'tester');
        addTask('t_startup', '生成 Agent 启动脚本', 'leader');
      }
    });

    // Step 2: Spawn RFC
    s.push({
      delay: 1500, action: () => {
        spawnAgent('rfc', 'RFC Writer', 'Doc-Alpha', '#0284c7', PenTool);
        addMessage('leader', 'rfc', 'Doc-Alpha, 请起草 Agent 的 System Prompt 并生成 Config Yaml。');
        updateAgent('leader', 'idle'); updateAgent('rfc', 'working');
        updateTaskStatus('t_rfc', 'in_progress');
      }
    });

    // Step 3: RFC 完成
    s.push({
      delay: 2000, action: () => {
        addMessage('rfc', 'leader', 'Prompt 与 Config Yaml 生成完毕。');
        updateAgent('rfc', 'success'); updateTaskStatus('t_rfc', 'done');
        completeArtifactPart('systemPrompt');
        completeArtifactPart('agentConfig');
      }
    });

    // Step 4: Spawn Builders 并行工作
    s.push({
      delay: 1500, action: () => {
        for(let i=1; i<=builderCount; i++) {
          spawnAgent(`b${i}`, 'Builder', `Eng-0${i}`, '#2563eb', Wrench);
          updateAgent(`b${i}`, 'working');
          updateTaskStatus(`t_b${i}`, 'in_progress');
        }
        addMessage('leader', 'b1', '开始并行开发各项业务 SKILL。', 'system');
      }
    });

    // Step 5: ASK USER (只针对 b1)
    s.push({
      delay: 2500, action: () => {
        updateAgent('b1', 'waiting');
        addMessage('b1', 'user', '⚠️ 遇到依赖冲突：应该使用哪种 SKILL 解析标准？', 'warning');

        setInteraction({
          agentName: "Eng-01 (Builder)",
          message: "在构建 SKILL 01 时，我们需要确定底层数据源解析标准：",
          options: [
            { id: 'cheerio', label: "Cheerio (快速，轻量，适合静态)" },
            { id: 'puppeteer', label: "Puppeteer (支持JS渲染，开销大)" }
          ],
          onResolve: (choice) => {
            addMessage('user', 'b1', `我选择: ${choice.label}`);
            setInteraction(null);
            updateAgent('b1', 'working');
            setStep(current => current + 1); // 继续下一步
          }
        });
        return "PAUSE";
      }
    });

    // Step 6: Builders 陆续完成
    s.push({
      delay: 2000, action: () => {
        for(let i=1; i<=builderCount; i++) {
          updateAgent(`b${i}`, 'success');
          updateTaskStatus(`t_b${i}`, 'done');
          completeArtifactPart('skill', `b${i}`);
        }
        addMessage('b1', 'leader', '所有 SKILL 组件开发并注册完毕。');
      }
    });

    // Step 7: Spawn Tester
    s.push({
      delay: 1500, action: () => {
        spawnAgent('tester', 'Tester', 'QA-Omega', '#ea580c', ShieldCheck);
        addMessage('leader', 'tester', '模块已集结。执行全链路沙盒测试，生成 Tool Config。', 'system');
        updateAgent('leader', 'idle'); updateAgent('tester', 'working');
        updateTaskStatus('t_qa', 'in_progress');
        setE2eStats({ total: 142, passed: 0, status: 'running' });
      }
    });

    // Step 8: 测试打回 b1
    s.push({
      delay: 2000, action: () => {
        updateAgent('tester', 'error');
        addMessage('tester', 'b1', '❌ 失败: SKILL 01 返回体不符合要求，Tool Config 生成失败！', 'error');
        updateAgent('b1', 'working');
        updateTaskStatus(`t_b1`, 'in_progress');
        setE2eStats(prev => ({ ...prev, passed: 89, status: 'failed' }));
        setArtifactState(prev => {
          const newSkills = {...prev.skills};
          delete newSkills['b1'];
          return {...prev, skills: newSkills};
        });
      }
    });

    // Step 9: b1 修复完毕
    s.push({
      delay: 2500, action: () => {
        updateAgent('b1', 'success');
        addMessage('b1', 'tester', '已修复返回格式问题，重新提交。');
        updateTaskStatus(`t_b1`, 'done');
        completeArtifactPart('skill', 'b1');
        updateAgent('tester', 'working');
        setE2eStats(prev => ({ ...prev, status: 'running' }));
      }
    });

    // Step 10: 测试通过
    s.push({
      delay: 2000, action: () => {
        updateAgent('tester', 'success');
        addMessage('tester', 'leader', '✅ 所有 SKILL 连通性测试通过，Tool Config 已锁定。');
        updateTaskStatus('t_qa', 'done');
        completeArtifactPart('toolConfig');
        setE2eStats(prev => ({ ...prev, passed: 142, status: 'passed' }));
      }
    });

    // Step 11: 启动脚本
    s.push({
      delay: 1500, action: () => {
        updateAgent('leader', 'working');
        updateTaskStatus('t_startup', 'in_progress');
        addMessage('leader', 'user', '测试完成，正在封装 NexAU 启动脚本 (Launch Script)...', 'system');
      }
    });

    // Step 12: 完成启动脚本
    s.push({
      delay: 1500, action: () => {
        updateAgent('leader', 'idle');
        updateTaskStatus('t_startup', 'done');
        completeArtifactPart('startupScript');
        addMessage('leader', 'user', '启动脚本生成完毕。', 'system');
      }
    });

    // Step 13: 结束
    s.push({
      delay: 1500, action: () => {
        updateAgent('leader', 'success');
        updateAgent('user', 'success');
        addMessage('leader', 'user', '🎉 NexAU Agent 构建完成，完整 Artifact 已准备就绪！', 'success');
        setIsPlaying(false);
      }
    });

    return s;
  };

  const script = useMemo(() => getScript(), [builderCount]);

  useEffect(() => {
    let timer;
    if (isPlaying && step < script.length && !interaction) {
      const currentAction = script[step];
      timer = setTimeout(() => {
        const result = currentAction.action();
        if (result !== "PAUSE") setStep(s => s + 1);
      }, currentAction.delay);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, step, interaction, script]);

  const handleStart = () => {
    setAgents({
      user: { id: 'user', role: 'User', name: 'You', status: 'idle', color: '#059669', icon: User },
      leader: { id: 'leader', role: 'Leader', name: 'Nexus-Prime', status: 'idle', color: '#6366f1', icon: Cpu },
    });
    setTasks([]);
    setArtifactState({ systemPrompt: false, agentConfig: false, toolConfig: false, startupScript: false, skills: {} });
    setE2eStats({ total: 142, passed: 0, status: 'idle' });
    setMessages([{ id: 'init', fromId: 'system', text: 'System ready. Awaiting initialization...', type: 'system' }]);
    setStep(0);
    setIsPlaying(true);
    setInteraction(null);
  };

  // --- 核心：自适应布局计算 (Adaptive Layout) ---
  const layoutedAgents = useMemo(() => {
    const list = Object.values(agents);
    const workers = list.filter(a => a.id !== 'user' && a.id !== 'leader');

    // 排序保证位置稳定: rfc -> b1..bN -> tester
    workers.sort((a, b) => {
      const weight = id => id === 'rfc' ? 0 : id === 'tester' ? 100 : parseInt(id.replace('b',''));
      return weight(a.id) - weight(b.id);
    });

    return list.map(agent => {
      let x = 50, y = 50;
      if (agent.id === 'user') { x = 50; y = 12; }
      else if (agent.id === 'leader') { x = 50; y = 35; }
      else {
        y = 75; // 工作节点在下方统一层级
        const index = workers.findIndex(w => w.id === agent.id);
        if (workers.length === 1) x = 50;
        else {
          x = 15 + (70 / (workers.length - 1)) * index;
        }
      }
      return { ...agent, x, y };
    });
  }, [agents]);

  // 计算 Artifact 进度条
  const totalArtifactParts = 4 + builderCount;
  const completedArtifactParts = (artifactState.systemPrompt ? 1 : 0) + (artifactState.agentConfig ? 1 : 0) + (artifactState.toolConfig ? 1 : 0) + (artifactState.startupScript ? 1 : 0) + Object.keys(artifactState.skills).length;
  const artifactProgress = (completedArtifactParts / totalArtifactParts) * 100;

  return (
    <div className="flex flex-col h-screen bg-[#f8fafc] text-slate-800 font-sans overflow-hidden selection:bg-slate-200">
      <style>{styles}</style>

      {/* 顶部 Header */}
      <header className="h-14 wireframe-panel border-b border-slate-200 flex items-center justify-between px-6 z-20 shrink-0 shadow-sm relative">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 bg-slate-800 flex items-center justify-center rounded-sm">
            <Zap size={16} className="text-white" />
          </div>
          <h1 className="font-semibold text-slate-900 tracking-tight text-sm uppercase">NexAU <span className="text-slate-400 font-normal">| Swarm Control</span></h1>
        </div>

        <div className="flex items-center gap-4">
          {/* Builder 数量控制 */}
          <div className="flex items-center gap-3 bg-white px-3 py-1.5 border border-slate-200 shadow-sm">
            <Users size={14} className="text-slate-500" />
            <span className="text-xs text-slate-500 font-mono uppercase tracking-wider">Builders:</span>
            <div className="flex items-center gap-1 bg-slate-50 border border-slate-200 px-1">
              <button onClick={() => setBuilderCount(Math.max(1, builderCount-1))} disabled={isPlaying} className="px-1.5 text-slate-500 hover:text-slate-900 disabled:opacity-30">-</button>
              <span className="text-sm font-mono w-4 text-center text-blue-600 font-bold">{builderCount}</span>
              <button onClick={() => setBuilderCount(Math.min(5, builderCount+1))} disabled={isPlaying} className="px-1.5 text-slate-500 hover:text-slate-900 disabled:opacity-30">+</button>
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={isPlaying && step < script.length}
            className={`flex items-center gap-2 px-4 py-1.5 font-medium text-xs uppercase tracking-wider transition-all border shadow-sm ${
              isPlaying && step < script.length
                ? 'bg-slate-100 border-slate-200 text-slate-400 cursor-not-allowed'
                : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-slate-400'
            }`}
          >
            {step >= script.length ? <RotateCcw size={14} /> : <Play size={14} fill="currentColor" />}
            {step >= script.length ? 'Reset Environment' : 'Initialize Swarm'}
          </button>
        </div>
      </header>

      {/* 主体三栏布局 */}
      <div className="flex flex-1 overflow-hidden">

        {/* 左侧：TaskBoard */}
        <div className="w-72 border-r border-slate-200 bg-white flex flex-col z-10 shadow-sm">
          <div className="p-3 border-b border-slate-200 flex items-center gap-2 bg-slate-50">
            <Layers size={14} className="text-slate-500" />
            <h2 className="font-semibold text-xs tracking-wider uppercase text-slate-700">Task Board</h2>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-[#fdfdfd]">
            {tasks.length === 0 && <p className="text-xs text-slate-400 italic text-center mt-4">No active tasks</p>}
            {tasks.map(task => {
              const assignee = agents[task.assigneeId];
              return (
                <div key={task.id} className="bg-white rounded-sm p-3 border border-slate-200 shadow-sm animate-in fade-in slide-in-from-left-2 transition-all hover:border-slate-300">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-semibold text-slate-700">{task.title}</span>
                    {task.status === 'pending' && <Circle size={14} className="text-slate-300" />}
                    {task.status === 'in_progress' && <Loader2 size={14} className="text-blue-500 animate-spin" />}
                    {task.status === 'done' && <CheckCircle2 size={14} className="text-emerald-500" />}
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    {assignee ? (
                      <div className="flex items-center gap-1 bg-slate-50 px-2 py-0.5 border border-slate-200 font-mono text-[10px]" style={{ color: assignee.color }}>
                        <assignee.icon size={10} />
                        {assignee.name}
                      </div>
                    ) : (
                      <div className="text-[10px] font-mono text-slate-400 bg-slate-50 px-2 py-0.5 border border-slate-200">Unassigned</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 中间：Swarm 拓扑图 (动态生成 + 自适应布局) */}
        <div className="flex-1 relative bg-[#f8fafc]">
          {/* 精致浅色网格背景 */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0.04)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.04)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

          {/* 动态连线 */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
            {layoutedAgents.map(target => {
              if (target.id === 'leader' || target.id === 'user') return null;
              const leader = layoutedAgents.find(a => a.id === 'leader');
              if (!leader) return null;

              const isActive = leader.status === 'working' || target.status === 'working';
              const color = isActive ? target.color : '#e2e8f0';

              return (
                <path
                  key={`link-leader-${target.id}`}
                  d={`M ${leader.x} ${leader.y} C ${leader.x} ${(leader.y+target.y)/2}, ${target.x} ${(leader.y+target.y)/2}, ${target.x} ${target.y}`}
                  fill="none" stroke={color} strokeWidth={isActive ? "1.5" : "1"}
                  className={isActive ? 'line-flow' : ''}
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
            {layoutedAgents.find(a => a.id === 'user') && layoutedAgents.find(a => a.id === 'leader') && (
              <path
                d={`M 50 12 L 50 35`}
                fill="none" stroke={agents.user.status === 'working' ? agents.user.color : '#e2e8f0'}
                strokeWidth={agents.user.status === 'working' ? "1.5" : "1"}
                className={agents.user.status === 'working' ? 'line-flow' : ''}
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>

          {/* 渲染节点与头顶浮动任务 */}
          {layoutedAgents.map(agent => {
            const isWorking = agent.status === 'working';
            const isWaiting = agent.status === 'waiting';
            const isError = agent.status === 'error';
            const isSuccess = agent.status === 'success';

            let statusColor = agent.color;
            if (isWaiting) statusColor = '#d97706'; // amber-600
            if (isError) statusColor = '#dc2626'; // red-600

            // 查找该节点正在执行的任务 (视觉任务绑定)
            const activeTask = tasks.find(t => t.assigneeId === agent.id && t.status === 'in_progress');

            return (
              <div
                key={agent.id}
                className="absolute flex flex-col items-center justify-center node-spawn z-10 transition-all duration-700 ease-in-out"
                style={{ left: `${agent.x}%`, top: `${agent.y}%`, transform: 'translate(-50%, -50%)' }}
              >
                {/* 浮动任务标签 (Hovering Task Badge) */}
                {activeTask && (
                  <div className="absolute -top-10 whitespace-nowrap bg-white text-slate-700 text-[10px] px-3 py-1.5 border shadow-sm animate-in slide-in-from-bottom-2 fade-in z-20 flex items-center gap-1.5" style={{ borderColor: `${statusColor}60` }}>
                    <Loader2 size={12} className="animate-spin" style={{ color: statusColor }}/>
                    <span className="font-semibold tracking-wide">{activeTask.title}</span>
                  </div>
                )}

                {/* 节点本体 (Wireframe 风格) */}
                <div
                  className="w-12 h-12 flex items-center justify-center relative bg-white transition-colors duration-300 shadow-sm"
                  style={{
                    border: `1.5px solid ${isWorking || isWaiting || isError ? statusColor : '#cbd5e1'}`,
                    color: isWorking || isWaiting || isError ? statusColor : '#64748b'
                  }}
                >
                  {isWorking && <div className="pulse-ring" style={{ color: statusColor }} />}
                  <agent.icon size={20} strokeWidth={1.5} />

                  {isSuccess && <div className="absolute -bottom-1 -right-1 bg-emerald-500 text-white p-0.5 border-2 border-white"><CheckCircle2 size={10} strokeWidth={3}/></div>}
                  {isError && <div className="absolute -bottom-1 -right-1 bg-red-500 text-white p-0.5 border-2 border-white"><AlertTriangle size={10} strokeWidth={3}/></div>}
                </div>

                {/* 节点信息牌 */}
                <div className="mt-2 text-center bg-white px-2 py-1 border border-slate-200 shadow-sm">
                  <p className="text-[9px] font-bold uppercase tracking-widest" style={{ color: agent.color }}>{agent.role}</p>
                  <p className="text-[10px] font-mono text-slate-500">{agent.name}</p>
                </div>
              </div>
            );
          })}

          {/* ASK USER 模态框 */}
          {interaction && (
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 bg-white border border-amber-300 p-6 shadow-2xl w-[400px] animate-in zoom-in-95 duration-200">
              <div className="flex items-center gap-2 text-amber-600 mb-4 border-b border-amber-100 pb-2">
                <AlertTriangle size={16} className="animate-pulse" />
                <h3 className="text-xs font-bold tracking-widest uppercase">Human Intervention Required</h3>
              </div>
              <div className="bg-amber-50/50 p-3 mb-5 border border-amber-100">
                <p className="text-[10px] font-mono text-slate-500 mb-1 uppercase tracking-wider">Source: <span className="text-blue-600 font-bold">{interaction.agentName}</span></p>
                <p className="text-sm text-slate-800 font-medium leading-relaxed">{interaction.message}</p>
              </div>
              <div className="space-y-2">
                {interaction.options.map(opt => (
                  <button
                    key={opt.id}
                    onClick={() => interaction.onResolve(opt)}
                    className="w-full text-left px-4 py-3 bg-white hover:bg-slate-50 border border-slate-200 hover:border-amber-400 text-sm transition-all group flex justify-between items-center shadow-sm"
                  >
                    <span className="text-slate-700 font-medium group-hover:text-amber-700">{opt.label}</span>
                    <span className="text-amber-600 text-[10px] uppercase font-bold tracking-wider opacity-0 group-hover:opacity-100 transition-opacity">Select</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 右侧：动态 Artifact Preview & E2E Tests */}
        <div className="w-[340px] border-l border-slate-200 bg-white flex flex-col z-10 shadow-sm">
          <div className="p-3 border-b border-slate-200 flex items-center gap-2 bg-slate-50 shrink-0">
            <Box size={14} className="text-slate-500" />
            <h2 className="font-semibold text-xs tracking-wider uppercase text-slate-700">NexAU Artifact</h2>
          </div>
          <div className="flex-1 p-6 flex flex-col items-center justify-start gap-4 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-opacity-5 overflow-y-auto pt-6">

            {/* 1. Agent Config Yaml */}
            <div className={`w-full max-w-[260px] h-12 shrink-0 flex items-center px-4 transition-all duration-700 relative ${artifactState.agentConfig ? 'border border-blue-500 bg-blue-50 artifact-part-reveal shadow-sm' : 'border border-dashed border-slate-300 bg-transparent'}`}>
              <FileJson size={16} className={artifactState.agentConfig ? 'text-blue-500 mr-3' : 'text-slate-400 mr-3'} />
              <span className={`text-[10px] uppercase tracking-widest font-bold ${artifactState.agentConfig ? 'text-blue-600' : 'text-slate-400'}`}>
                {artifactState.agentConfig ? 'Agent Config Yaml' : '[ Config Pending ]'}
              </span>
            </div>

            {/* 2. System Prompt */}
            <div className={`w-full max-w-[260px] h-12 shrink-0 flex items-center px-4 transition-all duration-700 relative ${artifactState.systemPrompt ? 'border border-purple-500 bg-purple-50 artifact-part-reveal shadow-sm' : 'border border-dashed border-slate-300 bg-transparent'}`}>
              <Terminal size={16} className={artifactState.systemPrompt ? 'text-purple-500 mr-3' : 'text-slate-400 mr-3'} />
              <span className={`text-[10px] uppercase tracking-widest font-bold ${artifactState.systemPrompt ? 'text-purple-600' : 'text-slate-400'}`}>
                {artifactState.systemPrompt ? 'System Prompt' : '[ Prompt Pending ]'}
              </span>
            </div>

            {/* 3. SKILLs */}
            <div className="w-full max-w-[260px] my-1">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5 border-b border-slate-200 pb-1">
                <Puzzle size={12}/> Registered SKILLs
              </div>
              <div className="flex gap-2 flex-wrap justify-start">
                {Array.from({length: builderCount}).map((_, i) => {
                  const id = `b${i+1}`;
                  const isDone = artifactState.skills[id];
                  return (
                    <div key={id} className={`flex-1 min-w-[70px] h-14 flex flex-col items-center justify-center transition-all duration-700 ${isDone ? 'border border-amber-500 bg-amber-50 artifact-part-reveal shadow-sm' : 'border border-dashed border-slate-300 bg-transparent'}`}>
                      <Zap size={14} className={isDone ? 'text-amber-500 mb-1.5' : 'text-slate-300 mb-1.5'} />
                      <span className={`text-[9px] font-bold text-center uppercase tracking-wider px-1 ${isDone ? 'text-amber-600' : 'text-slate-400'}`}>Skill 0${i+1}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 4. Tool Config */}
            <div className={`w-full max-w-[260px] h-12 shrink-0 flex items-center px-4 transition-all duration-700 relative ${artifactState.toolConfig ? 'border border-emerald-500 bg-emerald-50 artifact-part-reveal shadow-sm' : 'border border-dashed border-slate-300 bg-transparent'}`}>
              <Wrench size={16} className={artifactState.toolConfig ? 'text-emerald-500 mr-3' : 'text-slate-400 mr-3'} />
              <span className={`text-[10px] uppercase tracking-widest font-bold ${artifactState.toolConfig ? 'text-emerald-600' : 'text-slate-400'}`}>
                {artifactState.toolConfig ? 'Tool Config' : '[ Tools Pending ]'}
              </span>
            </div>

            {/* 5. 启动脚本 (Startup Script) */}
            <div className={`w-full max-w-[260px] h-12 shrink-0 flex items-center px-4 transition-all duration-700 relative ${artifactState.startupScript ? 'border border-indigo-500 bg-indigo-50 artifact-part-reveal shadow-sm' : 'border border-dashed border-slate-300 bg-transparent'}`}>
              <Rocket size={16} className={artifactState.startupScript ? 'text-indigo-500 mr-3' : 'text-slate-400 mr-3'} />
              <span className={`text-[10px] uppercase tracking-widest font-bold ${artifactState.startupScript ? 'text-indigo-600' : 'text-slate-400'}`}>
                {artifactState.startupScript ? 'Launch Script.sh' : '[ Script Pending ]'}
              </span>
            </div>

            <div className="mt-4 text-center w-full max-w-[260px]">
              <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-2">
                <span>Build Progress</span>
                <span>{Math.round(artifactProgress)}%</span>
              </div>
              <div className="h-1 w-full bg-slate-200 overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-1000"
                  style={{ width: `${artifactProgress}%` }}
                />
              </div>
            </div>

          </div>

          {/* 端到端测试 (E2E Test) 仪表盘 */}
          <div className="h-36 border-t border-slate-200 bg-white flex flex-col shrink-0">
            <div className="p-2.5 border-b border-slate-100 flex items-center gap-2 bg-slate-50">
              <ShieldCheck size={14} className={e2eStats.status === 'running' ? 'text-blue-500 animate-pulse' : e2eStats.status === 'failed' ? 'text-red-500' : e2eStats.status === 'passed' ? 'text-emerald-500' : 'text-slate-400'} />
              <h2 className="font-semibold text-[11px] tracking-wider uppercase text-slate-700">E2E Verification</h2>
              {e2eStats.status === 'running' && <span className="ml-auto text-[9px] text-blue-500 font-bold animate-pulse uppercase">Running...</span>}
              {e2eStats.status === 'failed' && <span className="ml-auto text-[9px] text-red-500 font-bold uppercase">Failed</span>}
              {e2eStats.status === 'passed' && <span className="ml-auto text-[9px] text-emerald-500 font-bold uppercase">Passed</span>}
            </div>
            <div className="flex-1 p-4 flex flex-col justify-center">
              <div className="flex justify-between items-end mb-2">
                <span className="text-xs font-semibold text-slate-600">Test Cases Passed</span>
                <div className="font-mono font-bold text-lg">
                  <span className={`transition-colors duration-300 ${e2eStats.status === 'failed' ? 'text-red-500' : e2eStats.status === 'passed' ? 'text-emerald-500' : 'text-blue-600'}`}>
                    {e2eStats.passed}
                  </span>
                  <span className="text-slate-400 text-sm ml-1">/ {e2eStats.total}</span>
                </div>
              </div>
              <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden shadow-inner">
                <div
                  className={`h-full transition-all duration-500 ease-out ${e2eStats.status === 'failed' ? 'bg-red-500' : e2eStats.status === 'passed' ? 'bg-emerald-500' : 'bg-blue-500'}`}
                  style={{ width: `${(e2eStats.passed / e2eStats.total) * 100}%` }}
                />
              </div>
              <div className="mt-2 text-[9px] font-mono text-slate-400 uppercase tracking-widest overflow-hidden text-ellipsis whitespace-nowrap">
                {e2eStats.status === 'idle' ? '> Waiting for deployment...' :
                 e2eStats.status === 'running' ? '> Executing suite [auth, api, core]...' :
                 e2eStats.status === 'failed' ? '> Error at node_modules. Halting test.' :
                 '> All systems go. Ready for production.'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 底部：消息流 / Chat Stream */}
      <div className="h-48 border-t border-slate-200 bg-white shrink-0 flex flex-col z-20">
        <div className="h-8 border-b border-slate-200 flex items-center px-4 gap-2 bg-slate-50">
          <MessageSquare size={14} className="text-slate-500" />
          <span className="text-[10px] uppercase tracking-widest text-slate-600 font-bold">Tracking Stream</span>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-[#fdfdfd]">
          {messages.map(msg => {
            const sender = agents[msg.fromId] || { name: 'System', color: '#64748b' };
            const isSystem = msg.type === 'system';
            const isError = msg.type === 'error';
            const isWarning = msg.type === 'warning';

            let bgClass = "bg-white border border-slate-200";
            if (isError) bgClass = "bg-red-50 border-red-200 text-red-800";
            if (isWarning) bgClass = "bg-amber-50 border-amber-200 text-amber-800";
            if (msg.fromId === 'user') bgClass = "bg-emerald-50 border-emerald-200 text-emerald-800";

            return (
              <div key={msg.id} className={`p-2.5 px-3 shadow-sm text-sm border transition-colors animate-in fade-in slide-in-from-bottom-2 ${bgClass}`}>
                <div className="flex items-baseline gap-2 font-mono">
                  <span className="font-bold text-[11px] uppercase tracking-wider" style={{ color: sender.color }}>
                    {msg.fromId === 'user' ? 'You' : sender.name}
                  </span>
                  {!isSystem && msg.toId && agents[msg.toId] && (
                    <span className="text-[10px] text-slate-400">
                      ▶ <span style={{ color: agents[msg.toId].color }}>@{agents[msg.toId].name}</span>
                    </span>
                  )}
                  <span className="text-slate-700 ml-2 font-sans text-[13px]">{msg.text}</span>
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} className="h-2" />
        </div>
      </div>

    </div>
  );
}
