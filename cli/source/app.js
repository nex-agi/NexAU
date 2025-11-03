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
	const agentProcess = useRef(null);
	const currentStepsRef = useRef([]);
	const {exit} = useApp();

	useEffect(() => {
		// Display welcome message
		setMessages([{
			role: 'assistant',
			content: 'Welcome to NorthAU Agent CLI! I\'m ready to help you with your tasks.\n\nAvailable commands:\n• Type your message to start a task\n• /clear - Clear the conversation history\n\nPress Esc or Ctrl+C to exit.',
			steps: []
		}]);

		// Start the Python agent process
		const pythonScript = path.join(__dirname, '..', 'agent_runner.py');
		const python = spawn('uv', ['run', 'python', pythonScript, yamlPath], {
			cwd: path.join(__dirname, '..', '..'),
		});

		agentProcess.current = python;

		python.stdout.on('data', data => {
			const lines = data.toString().split('\n').filter(line => line.trim());

			for (const line of lines) {
				try {
					const message = JSON.parse(line);

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
	});

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
					🤖 NorthAU Agent CLI
				</Text>
				<Text dimColor> (Press Esc or Ctrl+C to exit)</Text>
			</Box>

			{/* Conversation History */}
			<Box flexDirection="column" flexGrow={1} marginBottom={1} paddingX={1}>
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
