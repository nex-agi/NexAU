/*
 * Copyright (c) Nex-AGI. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React, {useState, useEffect, useMemo, useRef} from 'react';
import {Box, Text, useInput, useApp} from 'ink';
import TextInput from 'ink-text-input';
import Spinner from 'ink-spinner';
import {spawn} from 'child_process';
import path from 'path';
import {fileURLToPath} from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const Message = ({role, content, steps}) => {
	const color = role === 'user' ? 'cyan' : 'green';
	const prefix = role === 'user' ? '❯' : '⚡';

	return (
		<Box flexDirection="column" marginBottom={1}>
			<Box>
				<Text bold color={color}>
					{prefix} {role === 'user' ? 'You' : 'Agent'}:
				</Text>
			</Box>
			{/* Show steps for agent messages */}
			{steps && steps.length > 0 && (
				<Box flexDirection="column" paddingLeft={2} marginBottom={1}>
					{steps.map((step, index) => (
						<StepMessage
							key={index}
							content={step.content}
							metadata={step.metadata}
						/>
					))}
					<Box marginTop={0}>
						<Text dimColor>─────</Text>
					</Box>
				</Box>
			)}
			<Box paddingLeft={2}>
				<Text>{content}</Text>
			</Box>
		</Box>
	);
};

const StepMessage = ({content, metadata}) => {
	let icon = '•';
	let color = 'gray';
	let isBold = false;

	if (metadata) {
		switch (metadata.type) {
			case 'thinking':
				icon = '🧠';
				color = 'magenta';
				break;
			case 'agent_thinking':
				icon = '💭';
				color = 'magenta';
				break;
			case 'tool_plan':
				icon = '🔧';
				color = 'blue';
				break;
			case 'tool_plan_header':
				icon = '🔧';
				color = 'blue';
				isBold = true;
				break;
			case 'tool_detail':
				icon = '';
				color = 'cyan';
				break;
			case 'tool_executed':
				icon = '✓';
				color = 'green';
				break;
			case 'subagent_plan':
				icon = '🤖';
				color = 'cyan';
				break;
			case 'start':
				icon = '▶';
				color = 'yellow';
				break;
			case 'complete':
				icon = '✓';
				color = 'green';
				break;
			default:
				icon = '•';
				color = 'gray';
		}
	}

	return (
		<Box paddingLeft={2} marginBottom={0}>
			<Text color={color} bold={isBold}>
				{icon && `${icon} `}
				{content}
			</Text>
		</Box>
	);
};

const StatusMessage = ({message, isThinking}) => {
	if (!message) return null;

	return (
		<Box marginBottom={1}>
			<Text color="yellow">
				{isThinking && (
					<>
						<Spinner type="dots" />{' '}
					</>
				)}
				{message}
			</Text>
		</Box>
	);
};

const parseTimestamp = value => {
	const parsed = Date.parse(value || '');
	return Number.isNaN(parsed) ? 0 : parsed;
};

const formatRelativeTime = value => {
	if (!value) return '-';
	const timestamp = parseTimestamp(value);
	if (!timestamp) return '-';

	const diffMs = Date.now() - timestamp;
	const absMs = Math.abs(diffMs);
	const units = [
		{label: 'day', ms: 24 * 60 * 60 * 1000},
		{label: 'hour', ms: 60 * 60 * 1000},
		{label: 'minute', ms: 60 * 1000},
		{label: 'second', ms: 1000},
	];

	for (const unit of units) {
		if (absMs >= unit.ms || unit.label === 'second') {
			const amount = Math.max(1, Math.floor(absMs / unit.ms));
			const suffix = amount === 1 ? '' : 's';
			return diffMs >= 0
				? `${amount} ${unit.label}${suffix} ago`
				: `in ${amount} ${unit.label}${suffix}`;
		}
	}

	return '-';
};

const truncateText = (value, maxLength) => {
	if (!value) return '';
	if (value.length <= maxLength) return value;
	return `${value.slice(0, Math.max(0, maxLength - 3))}...`;
};

const toSingleLine = value =>
	String(value || '')
		.replace(/\s+/g, ' ')
		.trim();

export default function App({yamlPath}) {
	const [messages, setMessages] = useState([]);
	const [currentSteps, setCurrentSteps] = useState([]);
	const [input, setInput] = useState('');
	const [isProcessing, setIsProcessing] = useState(false);
	const [isInterrupting, setIsInterrupting] = useState(false);
	const [statusMessage, setStatusMessage] = useState('Initializing...');
	const [isReady, setIsReady] = useState(false);
	const [error, setError] = useState(null);
	const [sessionInfo, setSessionInfo] = useState(null);
	const [subAgentTraces, setSubAgentTraces] = useState(() => new Map());
	const [activeSubAgentId, setActiveSubAgentId] = useState(null);
	const [resumePickerOpen, setResumePickerOpen] = useState(false);
	const [resumePickerLoading, setResumePickerLoading] = useState(false);
	const [resumePickerError, setResumePickerError] = useState('');
	const [resumeSessions, setResumeSessions] = useState([]);
	const [resumeSearch, setResumeSearch] = useState('');
	const [resumeSortBy, setResumeSortBy] = useState('updated_at');
	const [resumeSelectedIndex, setResumeSelectedIndex] = useState(0);
	const [modelPickerOpen, setModelPickerOpen] = useState(false);
	const [modelPickerLoading, setModelPickerLoading] = useState(false);
	const [modelPickerError, setModelPickerError] = useState('');
	const [modelOptions, setModelOptions] = useState([]);
	const [modelSearch, setModelSearch] = useState('');
	const [modelSelectedIndex, setModelSelectedIndex] = useState(0);
	const agentProcess = useRef(null);
	const shutdownRequested = useRef(false);
	const currentStepsRef = useRef([]);
	const {exit} = useApp();
	const resetConversationView = () => {
		setMessages([]);
		currentStepsRef.current = [];
		setCurrentSteps([]);
		setSubAgentTraces(new Map());
		setActiveSubAgentId(null);
		setError(null);
	};

	const ensureSubAgentEntry = (metadata = {}, content = '') => ({
		agentId: metadata.agent_id || 'unknown',
		agentName: metadata.agent_name || 'Sub-Agent',
		displayName: metadata.display_name || metadata.agent_name || 'Sub-Agent',
		parentAgentName: metadata.parent_agent_name || '',
		parentAgentId: metadata.parent_agent_id || '',
		message: content || metadata.message || '',
		steps: [],
		status: 'running',
		result: '',
		error: '',
		lastUpdated: Date.now(),
	});

	useEffect(() => {
		if (subAgentTraces.size === 0) {
			setActiveSubAgentId(null);
			return;
		}

		if (activeSubAgentId && !subAgentTraces.has(activeSubAgentId)) {
			const firstId = subAgentTraces.keys().next().value;
			setActiveSubAgentId(firstId || null);
		}
	}, [subAgentTraces, activeSubAgentId]);

	useEffect(() => {
		// Display welcome message
		setMessages([
			{
				role: 'assistant',
				content:
					"Welcome to NexAU Agent CLI! I'm ready to help you with your tasks.\n\nAvailable commands:\n• Type your message to start a task\n• /clear - Start a fresh session\n• /interrupt - Stop the current run and keep the session\n• /resume - Open session picker\n• /model - Open model picker\n\nPress Esc to exit. Press Ctrl+C while running to interrupt.",
				steps: [],
			},
		]);

		// Start the Python agent process from the packaged module
		const repoRoot = path.join(__dirname, '..', '..');
		const pythonModule = 'nexau.cli.agent_runner';
		const python = spawn(
			'uv',
			['run', 'python', '-m', pythonModule, yamlPath],
			{
				cwd: repoRoot,
			},
		);

		agentProcess.current = python;

		python.stdout.on('data', data => {
			const lines = data
				.toString()
				.split('\n')
				.filter(line => line.trim());

			for (const line of lines) {
				try {
					const message = JSON.parse(line);
					const metadata = message.metadata || {};
					if (metadata.session_id || metadata.user_id || metadata.agent_id) {
						setSessionInfo({
							sessionId: metadata.session_id || '',
							userId: metadata.user_id || '',
							agentId: metadata.agent_id || '',
							model: metadata.model || '',
							storagePath: metadata.storage_path || '',
							storageRawPath: metadata.storage_raw_path || '',
							indexPath: metadata.index_path || '',
							indexRawPath: metadata.index_raw_path || '',
							restored: Boolean(metadata.restored),
						});
					}

					switch (message.type) {
						case 'status':
							setStatusMessage(message.content);
							break;
						case 'session':
							if (metadata.reset_ui) {
								resetConversationView();
							}
							setSessionInfo({
								sessionId: metadata.session_id || '',
								userId: metadata.user_id || '',
								agentId: metadata.agent_id || '',
								model: metadata.model || '',
								storagePath: metadata.storage_path || '',
								storageRawPath: metadata.storage_raw_path || '',
								indexPath: metadata.index_path || '',
								indexRawPath: metadata.index_raw_path || '',
								restored: Boolean(metadata.restored),
							});
							break;
						case 'ready':
							setIsReady(true);
							setIsProcessing(false);
							setIsInterrupting(false);
							setStatusMessage('');
							// Don't clear steps - they're now part of history
							break;
						case 'resume_list': {
							const sessions = Array.isArray(metadata.sessions)
								? metadata.sessions
								: [];
							setResumeSessions(sessions);
							setResumePickerOpen(true);
							setResumePickerLoading(false);
							setResumePickerError('');
							setResumeSearch('');
							setResumeSortBy('updated_at');
							setResumeSelectedIndex(0);
							break;
						}
						case 'model_list': {
							const models = Array.isArray(metadata.models)
								? metadata.models
								: [];
							setModelOptions(models);
							setModelPickerOpen(true);
							setModelPickerLoading(false);
							setModelPickerError('');
							setModelSearch('');
							setModelSelectedIndex(0);
							break;
						}
						case 'subagent_start': {
							const agentId = metadata.agent_id;
							if (!agentId) break;

							setSubAgentTraces(prev => {
								const next = new Map(prev);
								next.set(
									agentId,
									ensureSubAgentEntry(metadata, message.content),
								);
								return next;
							});
							setActiveSubAgentId(prev => prev ?? agentId);
							break;
						}
						case 'subagent_step':
						case 'subagent_text': {
							const agentId = metadata.agent_id;
							if (!agentId) break;

							setSubAgentTraces(prev => {
								const next = new Map(prev);
								const existing =
									next.get(agentId) || ensureSubAgentEntry(metadata);
								const updated = {
									...existing,
									agentName: metadata.agent_name || existing.agentName,
									displayName: metadata.display_name || existing.displayName,
									parentAgentName:
										metadata.parent_agent_name ?? existing.parentAgentName,
									parentAgentId:
										metadata.parent_agent_id ?? existing.parentAgentId,
									steps: [
										...existing.steps,
										{content: message.content, metadata},
									],
									lastUpdated: Date.now(),
								};
								next.set(agentId, updated);
								return next;
							});
							setActiveSubAgentId(prev => prev ?? agentId);
							break;
						}
						case 'subagent_complete': {
							const agentId = metadata.agent_id;
							if (!agentId) break;

							setSubAgentTraces(prev => {
								if (!prev.has(agentId)) return prev;
								const next = new Map(prev);
								next.delete(agentId);
								return next;
							});
							break;
						}
						case 'subagent_error': {
							const agentId = metadata.agent_id;
							if (!agentId) break;

							setSubAgentTraces(prev => {
								if (!prev.has(agentId)) return prev;
								const next = new Map(prev);
								next.delete(agentId);
								return next;
							});
							break;
						}
						case 'step':
							// Add intermediate step
							const newStep = {
								content: message.content,
								metadata: message.metadata,
							};
							currentStepsRef.current = [...currentStepsRef.current, newStep];
							setCurrentSteps(currentStepsRef.current);
							break;
						case 'agent_text':
							// Agent's text response (non-tool part)
							const textStep = {
								content: message.content,
								metadata: {type: 'agent_thinking', isText: true},
							};
							currentStepsRef.current = [...currentStepsRef.current, textStep];
							setCurrentSteps(currentStepsRef.current);
							break;
						case 'thinking':
							setStatusMessage(message.content);
							setIsProcessing(true);
							break;
						case 'interrupted':
							setMessages(prev => [
								...prev,
								{
									role: 'assistant',
									content: message.content,
									steps: [...currentStepsRef.current],
								},
							]);
							currentStepsRef.current = [];
							setCurrentSteps([]);
							setIsProcessing(false);
							setIsInterrupting(false);
							break;
						case 'response':
							// Add final response and preserve the steps with it
							setMessages(prev => [
								...prev,
								{
									role: 'assistant',
									content: message.content,
									steps: [...currentStepsRef.current], // Use ref for current steps
								},
							]);
							// Clear current steps for next interaction
							currentStepsRef.current = [];
							setCurrentSteps([]);
							break;
						case 'error':
							setError(message.content);
							setResumePickerLoading(false);
							setModelPickerLoading(false);
							setStatusMessage('');
							setIsProcessing(false);
							setIsInterrupting(false);
							currentStepsRef.current = [];
							setCurrentSteps([]);
							break;
					}
				} catch (e) {
					// Ignore non-JSON output
				}
			}
		});

		python.stderr.on('data', data => {
			// Silent stderr or log to file if needed
			console.error('Agent stderr:', data.toString());
		});

		python.on('close', code => {
			if (shutdownRequested.current) {
				return;
			}
			if (code !== 0) {
				setError(`Agent process exited with code ${code}`);
			}
		});

		return () => {
			if (!agentProcess.current) {
				return;
			}

			shutdownRequested.current = true;
			const proc = agentProcess.current;
			agentProcess.current = null;

			proc.stdin.write(JSON.stringify({type: 'exit'}) + '\n');
			proc.stdin.end();

			setTimeout(() => {
				if (!proc.killed) {
					proc.kill();
				}
			}, 2000);
		};
	}, [yamlPath]);

	const filteredResumeSessions = useMemo(() => {
		const normalizedQuery = resumeSearch.trim().toLowerCase();
		let filtered = resumeSessions;
		if (normalizedQuery) {
			filtered = resumeSessions.filter(session => {
				const id = String(session.id || '').toLowerCase();
				const conversation = String(session.conversation || '').toLowerCase();
				const createdAt = String(session.created_at || '').toLowerCase();
				const updatedAt = String(session.updated_at || '').toLowerCase();
				return (
					id.includes(normalizedQuery) ||
					conversation.includes(normalizedQuery) ||
					createdAt.includes(normalizedQuery) ||
					updatedAt.includes(normalizedQuery)
				);
			});
		}

		const sorted = [...filtered];
		sorted.sort((left, right) => {
			const leftTs = parseTimestamp(left?.[resumeSortBy]);
			const rightTs = parseTimestamp(right?.[resumeSortBy]);
			if (leftTs !== rightTs) {
				return rightTs - leftTs;
			}
			return String(left?.id || '').localeCompare(String(right?.id || ''));
		});
		return sorted;
	}, [resumeSessions, resumeSearch, resumeSortBy]);

	useEffect(() => {
		setResumeSelectedIndex(prev => {
			if (filteredResumeSessions.length === 0) return 0;
			if (prev < 0) return 0;
			if (prev >= filteredResumeSessions.length)
				return filteredResumeSessions.length - 1;
			return prev;
		});
	}, [filteredResumeSessions]);

	const filteredModelOptions = useMemo(() => {
		const normalizedQuery = modelSearch.trim().toLowerCase();
		if (!normalizedQuery) {
			return modelOptions;
		}

		return modelOptions.filter(option => {
			const name = String(option.name || '').toLowerCase();
			const alias = String(option.profile_alias || '').toLowerCase();
			return name.includes(normalizedQuery) || alias.includes(normalizedQuery);
		});
	}, [modelOptions, modelSearch]);

	useEffect(() => {
		setModelSelectedIndex(prev => {
			if (filteredModelOptions.length === 0) return 0;
			if (prev < 0) return 0;
			if (prev >= filteredModelOptions.length)
				return filteredModelOptions.length - 1;
			return prev;
		});
	}, [filteredModelOptions]);

	const requestModelPicker = () => {
		if (!agentProcess.current) {
			setModelPickerError('Agent process is not available.');
			return;
		}

		setResumePickerOpen(false);
		setModelPickerOpen(true);
		setModelPickerLoading(true);
		setModelPickerError('');
		setModelSearch('');
		setModelSelectedIndex(0);
		setModelOptions([]);
		agentProcess.current.stdin.write(
			JSON.stringify({type: 'model_list'}) + '\n',
		);
	};

	const runModelUse = modelName => {
		if (!agentProcess.current) return;
		const target = String(modelName || '').trim();
		if (!target) return;

		setModelPickerOpen(false);
		setModelPickerLoading(false);
		setModelPickerError('');
		setIsReady(false);
		setIsProcessing(true);
		setError(null);
		setStatusMessage(`Switching model to ${target}...`);
		agentProcess.current.stdin.write(
			JSON.stringify({type: 'model_use', model_name: target}) + '\n',
		);
	};

	const requestResumePicker = () => {
		if (!agentProcess.current) {
			setResumePickerError('Agent process is not available.');
			return;
		}

		setModelPickerOpen(false);
		setResumePickerOpen(true);
		setResumePickerLoading(true);
		setResumePickerError('');
		setResumeSearch('');
		setResumeSortBy('updated_at');
		setResumeSelectedIndex(0);
		setResumeSessions([]);
		agentProcess.current.stdin.write(
			JSON.stringify({type: 'resume_list'}) + '\n',
		);
	};

	const runResumeUse = sessionId => {
		if (!agentProcess.current) return;
		const target = String(sessionId || '').trim();
		if (!target) return;

		setResumePickerOpen(false);
		setResumePickerLoading(false);
		setResumePickerError('');
		setIsReady(false);
		setIsProcessing(true);
		setError(null);
		setStatusMessage('Resuming selected session...');
		agentProcess.current.stdin.write(
			JSON.stringify({type: 'resume_use', session_id: target}) + '\n',
		);
	};

	const runResumeNew = () => {
		if (!agentProcess.current) return;
		setResumePickerOpen(false);
		setResumePickerLoading(false);
		setResumePickerError('');
		setIsReady(false);
		setIsProcessing(true);
		setError(null);
		setStatusMessage('Starting a fresh session...');
		agentProcess.current.stdin.write(
			JSON.stringify({type: 'resume_new'}) + '\n',
		);
	};

	const handleSubmit = value => {
		if (!value.trim()) return;

		const userMessage = value.trim();

		if (isProcessing) {
			if (userMessage === '/interrupt') {
				if (!agentProcess.current || isInterrupting) return;

				setInput('');
				setError(null);
				setIsInterrupting(true);
				setStatusMessage('Interrupting current run...');
				agentProcess.current.stdin.write(
					JSON.stringify({type: 'interrupt'}) + '\n',
				);
				return;
			}

			setError(
				'Agent is running. Press Ctrl+C or type /interrupt to stop the current run.',
			);
			setInput('');
			return;
		}

		if (!isReady) return;

		// Check for /clear command
		if (userMessage === '/clear' || userMessage.startsWith('/clear ')) {
			// Clear all conversation history and reset to fresh empty state
			resetConversationView();
			setInput('');

			// Send message to Python agent
			if (agentProcess.current) {
				agentProcess.current.stdin.write(
					JSON.stringify({type: 'clear'}) + '\n',
				);
			}
			return;
		}

		if (userMessage === '/interrupt') {
			setError('No active run to interrupt.');
			setInput('');
			return;
		}

		if (userMessage === '/resume') {
			setInput('');
			setError(null);
			requestResumePicker();
			return;
		}

		if (userMessage === '/model') {
			setInput('');
			setError(null);
			requestModelPicker();
			return;
		}

		if (userMessage.startsWith('/model ')) {
			setError('Usage: /model');
			setInput('');
			return;
		}

		setMessages(prev => [...prev, {role: 'user', content: userMessage}]);
		setInput('');
		setIsReady(false);
		setIsProcessing(true);
		setError(null);
		currentStepsRef.current = [];
		setCurrentSteps([]);

		// Send message to Python agent
		if (agentProcess.current) {
			agentProcess.current.stdin.write(
				JSON.stringify({type: 'message', content: userMessage}) + '\n',
			);
		}
	};

	const requestExit = () => {
		if (!agentProcess.current) {
			exit();
			return;
		}

		shutdownRequested.current = true;
		const proc = agentProcess.current;
		agentProcess.current = null;
		proc.stdin.write(JSON.stringify({type: 'exit'}) + '\n');
		proc.stdin.end();
		setTimeout(() => {
			if (!proc.killed) {
				proc.kill();
			}
		}, 2000);
		exit();
	};

	useInput((input, key) => {
		if (modelPickerOpen) {
			if (input === 'c' && key.ctrl) {
				if (isProcessing) {
					if (agentProcess.current && !isInterrupting) {
						setError(null);
						setIsInterrupting(true);
						setStatusMessage('Interrupting current run...');
						agentProcess.current.stdin.write(
							JSON.stringify({type: 'interrupt'}) + '\n',
						);
					}
					return;
				}
				requestExit();
				return;
			}

			if (key.escape) {
				setModelPickerOpen(false);
				setModelPickerLoading(false);
				setModelPickerError('');
				return;
			}

			if (key.return) {
				const selected = filteredModelOptions[modelSelectedIndex];
				if (selected?.name) {
					runModelUse(selected.name);
				}
				return;
			}

			if (key.upArrow) {
				setModelSelectedIndex(prev => Math.max(0, prev - 1));
				return;
			}

			if (key.downArrow) {
				setModelSelectedIndex(prev =>
					Math.min(Math.max(0, filteredModelOptions.length - 1), prev + 1),
				);
				return;
			}

			if (key.backspace || key.delete) {
				setModelSearch(prev => prev.slice(0, -1));
				return;
			}

			if (!key.ctrl && !key.meta && input) {
				setModelSearch(prev => prev + input);
			}
			return;
		}

		if (resumePickerOpen) {
			if (input === 'c' && key.ctrl) {
				if (isProcessing) {
					if (agentProcess.current && !isInterrupting) {
						setError(null);
						setIsInterrupting(true);
						setStatusMessage('Interrupting current run...');
						agentProcess.current.stdin.write(
							JSON.stringify({type: 'interrupt'}) + '\n',
						);
					}
					return;
				}
				requestExit();
				return;
			}

			if (key.escape) {
				runResumeNew();
				return;
			}

			if (key.return) {
				const selected = filteredResumeSessions[resumeSelectedIndex];
				if (selected?.id) {
					runResumeUse(selected.id);
				} else if (!resumePickerLoading) {
					runResumeNew();
				}
				return;
			}

			if (key.tab) {
				setResumeSortBy(prev =>
					prev === 'updated_at' ? 'created_at' : 'updated_at',
				);
				return;
			}

			if (key.upArrow) {
				setResumeSelectedIndex(prev => Math.max(0, prev - 1));
				return;
			}

			if (key.downArrow) {
				setResumeSelectedIndex(prev =>
					Math.min(Math.max(0, filteredResumeSessions.length - 1), prev + 1),
				);
				return;
			}

			if (key.backspace || key.delete) {
				setResumeSearch(prev => prev.slice(0, -1));
				return;
			}

			if (!key.ctrl && !key.meta && input) {
				setResumeSearch(prev => prev + input);
			}
			return;
		}

		if (key.escape) {
			requestExit();
			return;
		}

		if (input === 'c' && key.ctrl) {
			if (isProcessing) {
				if (agentProcess.current && !isInterrupting) {
					setError(null);
					setIsInterrupting(true);
					setStatusMessage('Interrupting current run...');
					agentProcess.current.stdin.write(
						JSON.stringify({type: 'interrupt'}) + '\n',
					);
				}
				return;
			}

			requestExit();
			return;
		}

		if (key.ctrl && input === 't' && subAgentTraces.size > 0) {
			setActiveSubAgentId(prev => {
				const ids = Array.from(subAgentTraces.keys());
				if (ids.length === 0) return prev;
				const currentIndex = prev ? ids.indexOf(prev) : -1;
				const nextIndex = (currentIndex + 1) % ids.length;
				return ids[nextIndex];
			});
		}
	});

	const subAgentEntries = Array.from(subAgentTraces.values());
	const activeSubAgent =
		activeSubAgentId && subAgentTraces.has(activeSubAgentId)
			? subAgentTraces.get(activeSubAgentId)
			: subAgentEntries[0] || null;
	const activeSubAgentIndex = activeSubAgent
		? subAgentEntries.findIndex(
				entry => entry.agentId === activeSubAgent.agentId,
		  )
		: -1;
	const totalSubAgents = subAgentEntries.length;
	const resumeSortLabel =
		resumeSortBy === 'updated_at' ? 'Updated at' : 'Created at';
	const resumeCreatedHeader = 'Created at';
	const resumeUpdatedHeader = 'Updated at';
	const resumeCreatedWidth = 14;
	const resumeUpdatedWidth = 14;
	const resumeHeaderRow = `${resumeCreatedHeader.padEnd(
		resumeCreatedWidth,
	)} ${resumeUpdatedHeader.padEnd(resumeUpdatedWidth)} Conversation`;
	const modelHeaderRow = 'Model';

	if (modelPickerOpen) {
		return (
			<Box flexDirection="column" height="100%" paddingX={1}>
				<Box marginBottom={1}>
					<Text bold>Select a model</Text>
				</Box>
				<Box marginBottom={1}>
					<Text dimColor>Type to search</Text>
					<Text> {modelSearch || ''}</Text>
				</Box>
				<Box marginBottom={1}>
					<Text bold>{modelHeaderRow}</Text>
				</Box>
				{modelPickerLoading && (
					<Box marginBottom={1}>
						<Text color="yellow">
							<Spinner type="dots" /> Loading models...
						</Text>
					</Box>
				)}
				{!modelPickerLoading && filteredModelOptions.length === 0 && (
					<Box marginBottom={1}>
						<Text color={modelOptions.length === 0 ? 'yellow' : 'red'}>
							{modelOptions.length === 0
								? 'No models available.'
								: 'No models match your current search.'}
						</Text>
					</Box>
				)}
				{modelPickerError && (
					<Box marginBottom={1}>
						<Text color="red">{modelPickerError}</Text>
					</Box>
				)}
				{!modelPickerLoading && filteredModelOptions.length > 0 && (
					<Box flexDirection="column" marginBottom={1}>
						{filteredModelOptions.map((option, index) => {
							const selected = index === modelSelectedIndex;
							const currentMarker = option.current ? ' [current]' : '';
							const alias = option.profile_alias
								? ` (profile: ${option.profile_alias})`
								: '';
							const row = truncateText(
								toSingleLine(`${option.name}${currentMarker}${alias}`),
								100,
							);
							return (
								<Text
									key={`${option.name}-${index}`}
									color={selected ? 'cyan' : undefined}
									wrap="truncate-end"
								>
									{selected ? '>' : ' '} {row}
								</Text>
							);
						})}
					</Box>
				)}
				<Box marginTop={1}>
					<Text dimColor>
						enter to switch esc to cancel ctrl + c to quit ↑/↓ to browse
					</Text>
				</Box>
			</Box>
		);
	}

	if (resumePickerOpen) {
		return (
			<Box flexDirection="column" height="100%" paddingX={1}>
				<Box marginBottom={1}>
					<Text bold>Resume a previous session</Text>
					<Text dimColor> Sort: {resumeSortLabel}</Text>
				</Box>
				<Box marginBottom={1}>
					<Text dimColor>Type to search</Text>
					<Text> {resumeSearch || ''}</Text>
				</Box>
				<Box marginBottom={1}>
					<Text bold>{resumeHeaderRow}</Text>
				</Box>
				{resumePickerLoading && (
					<Box marginBottom={1}>
						<Text color="yellow">
							<Spinner type="dots" /> Loading sessions...
						</Text>
					</Box>
				)}
				{!resumePickerLoading && filteredResumeSessions.length === 0 && (
					<Box marginBottom={1}>
						<Text color={resumeSessions.length === 0 ? 'yellow' : 'red'}>
							{resumeSessions.length === 0
								? 'No saved sessions found. Press Esc to start new.'
								: 'No sessions match your current search.'}
						</Text>
					</Box>
				)}
				{resumePickerError && (
					<Box marginBottom={1}>
						<Text color="red">{resumePickerError}</Text>
					</Box>
				)}
				{!resumePickerLoading && filteredResumeSessions.length > 0 && (
					<Box flexDirection="column" marginBottom={1}>
						{filteredResumeSessions.map((session, index) => {
							const selected = index === resumeSelectedIndex;
							const createdAt = formatRelativeTime(session.created_at).padEnd(
								resumeCreatedWidth,
							);
							const updatedAt = formatRelativeTime(session.updated_at).padEnd(
								resumeUpdatedWidth,
							);
							const conversation = truncateText(
								toSingleLine(session.conversation),
								80,
							);
							const row = `${createdAt} ${updatedAt} ${conversation}`;
							return (
								<Text
									key={session.id || `${session.index}-${index}`}
									color={selected ? 'cyan' : undefined}
									wrap="truncate-end"
								>
									{selected ? '>' : ' '} {row}
								</Text>
							);
						})}
					</Box>
				)}
				<Box marginTop={1}>
					<Text dimColor>
						enter to resume esc to start new ctrl + c to quit tab to toggle sort
						↑/↓ to browse
					</Text>
				</Box>
			</Box>
		);
	}

	return (
		<Box flexDirection="column" height="100%">
			{/* Header */}
			<Box borderStyle="round" borderColor="blue" paddingX={1} marginBottom={1}>
				<Text bold color="blue">
					🤖 NexAU Agent CLI
				</Text>
				<Text dimColor> (Esc exits, Ctrl+C interrupts active runs)</Text>
			</Box>

			{/* Main Content */}
			<Box flexDirection="row" flexGrow={1} marginBottom={1} paddingX={1}>
				<Box
					flexDirection="column"
					flexGrow={2}
					marginRight={activeSubAgent ? 1 : 0}
				>
					{sessionInfo?.sessionId && (
						<Box marginBottom={1} flexDirection="column">
							<Text dimColor>
								Session: {sessionInfo.sessionId}
								{sessionInfo.restored ? ' (restored)' : ''}
							</Text>
							{sessionInfo.model && (
								<Text dimColor>Model: {sessionInfo.model}</Text>
							)}
							{sessionInfo.storagePath && (
								<Text dimColor>Store: {sessionInfo.storagePath}</Text>
							)}
							{sessionInfo.storageRawPath && (
								<Text dimColor>Raw: {sessionInfo.storageRawPath}</Text>
							)}
							{sessionInfo.indexPath && (
								<Text dimColor>Index: {sessionInfo.indexPath}</Text>
							)}
							{sessionInfo.indexRawPath && (
								<Text dimColor>Index Raw: {sessionInfo.indexRawPath}</Text>
							)}
						</Box>
					)}

					{messages.map((msg, index) => (
						<Message
							key={index}
							role={msg.role}
							content={msg.content}
							steps={msg.steps}
						/>
					))}

					{error && (
						<Box marginBottom={1}>
							<Text color="red">✗ Error: {error}</Text>
						</Box>
					)}

					{/* Show current steps (work in progress) */}
					{currentSteps.length > 0 && (
						<Box flexDirection="column" marginBottom={1}>
							<Box paddingLeft={2} marginBottom={0}>
								<Text color="yellow" bold>
									<Spinner type="dots" /> Working...
								</Text>
							</Box>
							{currentSteps.map((step, index) => (
								<StepMessage
									key={index}
									content={step.content}
									metadata={step.metadata}
								/>
							))}
						</Box>
					)}

					{statusMessage && !currentSteps.length && (
						<StatusMessage message={statusMessage} isThinking={isProcessing} />
					)}
				</Box>

				{activeSubAgent && (
					<Box
						flexDirection="column"
						flexGrow={1}
						borderStyle="round"
						borderColor="cyan"
						paddingX={1}
					>
						<Box marginBottom={1} flexDirection="column">
							<Text bold color="cyan">
								Sub-Agent Trace
							</Text>
							<Text dimColor>
								{activeSubAgent.displayName}
								{totalSubAgents > 1 &&
									activeSubAgentIndex >= 0 &&
									` (${
										activeSubAgentIndex + 1
									}/${totalSubAgents}, press "ctrl+t" to switch)`}
							</Text>
							{activeSubAgent.parentAgentName && (
								<Text dimColor>Parent: {activeSubAgent.parentAgentName}</Text>
							)}
						</Box>

						{activeSubAgent.message && (
							<Box marginBottom={1}>
								<Text dimColor>Message: {activeSubAgent.message}</Text>
							</Box>
						)}

						{activeSubAgent.status === 'running' && (
							<Box marginBottom={1}>
								<Text color="yellow">
									<Spinner type="dots" /> Running...
								</Text>
							</Box>
						)}

						{activeSubAgent.steps.map((step, index) => (
							<StepMessage
								key={index}
								content={step.content}
								metadata={step.metadata}
							/>
						))}

						{activeSubAgent.status === 'complete' && activeSubAgent.result && (
							<Box marginTop={1}>
								<Text color="green">Result: {activeSubAgent.result}</Text>
							</Box>
						)}

						{activeSubAgent.status === 'error' && activeSubAgent.error && (
							<Box marginTop={1}>
								<Text color="red">Error: {activeSubAgent.error}</Text>
							</Box>
						)}
					</Box>
				)}
			</Box>

			{/* Input Bar */}
			<Box
				borderStyle="round"
				borderColor={isReady ? 'green' : isProcessing ? 'yellow' : 'gray'}
				paddingX={1}
			>
				<Text color={isReady ? 'green' : isProcessing ? 'yellow' : 'gray'}>
					{isReady ? '▶' : isProcessing ? '■' : '⏸'}{' '}
				</Text>
				<TextInput
					value={input}
					onChange={setInput}
					onSubmit={handleSubmit}
					placeholder={
						isReady
							? 'Type your message and press Enter...'
							: isProcessing && !isInterrupting
							? 'Agent is running. Type /interrupt or press Ctrl+C to stop...'
							: isInterrupting
							? 'Interrupting current run...'
							: 'Waiting for agent...'
					}
					isDisabled={(!isReady && !isProcessing) || isInterrupting}
				/>
			</Box>
		</Box>
	);
}
