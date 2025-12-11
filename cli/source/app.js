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

import React, {useState, useEffect, useRef} from 'react';
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

export default function App({yamlPath}) {
	const [messages, setMessages] = useState([]);
	const [currentSteps, setCurrentSteps] = useState([]);
	const [input, setInput] = useState('');
	const [isProcessing, setIsProcessing] = useState(false);
	const [statusMessage, setStatusMessage] = useState('Initializing...');
	const [isReady, setIsReady] = useState(false);
	const [error, setError] = useState(null);
	const [subAgentTraces, setSubAgentTraces] = useState(() => new Map());
	const [activeSubAgentId, setActiveSubAgentId] = useState(null);
	const agentProcess = useRef(null);
	const currentStepsRef = useRef([]);
	const {exit} = useApp();

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
		setMessages([{
			role: 'assistant',
			content: 'Welcome to NexAU Agent CLI! I\'m ready to help you with your tasks.\n\nAvailable commands:\n• Type your message to start a task\n• /clear - Clear the conversation history\n\nPress Esc or Ctrl+C to exit.',
			steps: []
		}]);

		// Start the Python agent process from the packaged module
		const repoRoot = path.join(__dirname, '..', '..');
		const pythonModule = 'nexau.cli.agent_runner';
		const python = spawn('uv', ['run', 'python', '-m', pythonModule, yamlPath], {
			cwd: repoRoot,
		});

		agentProcess.current = python;

		python.stdout.on('data', data => {
			const lines = data.toString().split('\n').filter(line => line.trim());

			for (const line of lines) {
				try {
					const message = JSON.parse(line);
					const metadata = message.metadata || {};

					switch (message.type) {
						case 'status':
							setStatusMessage(message.content);
							break;
						case 'ready':
							setIsReady(true);
							setIsProcessing(false);
							setStatusMessage('');
							// Don't clear steps - they're now part of history
							break;
						case 'subagent_start': {
							const agentId = metadata.agent_id;
							if (!agentId) break;

							setSubAgentTraces(prev => {
								const next = new Map(prev);
								next.set(agentId, ensureSubAgentEntry(metadata, message.content));
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
								const existing = next.get(agentId) || ensureSubAgentEntry(metadata);
								const updated = {
									...existing,
									agentName: metadata.agent_name || existing.agentName,
									displayName: metadata.display_name || existing.displayName,
									parentAgentName: metadata.parent_agent_name ?? existing.parentAgentName,
									parentAgentId: metadata.parent_agent_id ?? existing.parentAgentId,
									steps: [...existing.steps, {content: message.content, metadata}],
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
							setStatusMessage('');
							setIsProcessing(false);
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
			if (code !== 0) {
				setError(`Agent process exited with code ${code}`);
			}
		});

		return () => {
			if (agentProcess.current) {
				agentProcess.current.stdin.write(
					JSON.stringify({type: 'exit'}) + '\n',
				);
				agentProcess.current.kill();
			}
		};
	}, [yamlPath]);

	const handleSubmit = value => {
		if (!value.trim() || !isReady || isProcessing) return;

		const userMessage = value.trim();

		// Check for /clear command
		if (userMessage === "/clear" || userMessage.startsWith("/clear ")) {
			// Clear all conversation history and reset to fresh empty state
			setMessages([]);
			setInput('');
			setError(null);
			currentStepsRef.current = [];
			setCurrentSteps([]);
			setSubAgentTraces(new Map());
			setActiveSubAgentId(null);

			// Send message to Python agent
			if (agentProcess.current) {
				agentProcess.current.stdin.write(
					JSON.stringify({type: 'message', content: userMessage}) + '\n',
				);
			}
			return;
		}

		setMessages(prev => [...prev, {role: 'user', content: userMessage}]);
		setInput('');
		setIsReady(false);
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

	useInput((input, key) => {
		if (key.escape || (input === 'c' && key.ctrl)) {
			if (agentProcess.current) {
				agentProcess.current.stdin.write(
					JSON.stringify({type: 'exit'}) + '\n',
				);
				agentProcess.current.kill();
			}

			exit();
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
		? subAgentEntries.findIndex(entry => entry.agentId === activeSubAgent.agentId)
		: -1;
	const totalSubAgents = subAgentEntries.length;

	return (
		<Box flexDirection="column" height="100%">
			{/* Header */}
			<Box
				borderStyle="round"
				borderColor="blue"
				paddingX={1}
				marginBottom={1}
			>
				<Text bold color="blue">
					🤖 NexAU Agent CLI
				</Text>
				<Text dimColor> (Press Esc or Ctrl+C to exit)</Text>
			</Box>

			{/* Main Content */}
			<Box flexDirection="row" flexGrow={1} marginBottom={1} paddingX={1}>
				<Box flexDirection="column" flexGrow={2} marginRight={activeSubAgent ? 1 : 0}>
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
								{totalSubAgents > 1 && activeSubAgentIndex >= 0 && ` (${activeSubAgentIndex + 1}/${totalSubAgents}, press "ctrl+t" to switch)`}
							</Text>
							{activeSubAgent.parentAgentName && (
								<Text dimColor>
									Parent: {activeSubAgent.parentAgentName}
								</Text>
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
				borderColor={isReady ? 'green' : 'gray'}
				paddingX={1}
			>
				<Text color={isReady ? 'green' : 'gray'}>{isReady ? '▶' : '⏸'} </Text>
				<TextInput
					value={input}
					onChange={setInput}
					onSubmit={handleSubmit}
					placeholder={
						isReady
							? 'Type your message and press Enter...'
							: 'Waiting for agent...'
					}
					isDisabled={!isReady || isProcessing}
				/>
			</Box>
		</Box>
	);
}
