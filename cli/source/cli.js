#!/usr/bin/env node
import React from 'react';
import {render} from 'ink';
import meow from 'meow';
import App from './app.js';
import path from 'path';
import fs from 'fs';

const cli = meow(
	`
		Usage
		  $ northau-cli <yaml-file>

		Arguments
		  <yaml-file>  Path to the NorthAU agent configuration YAML file

		Options
		  --help       Show this help message

		Examples
		  $ northau-cli ./my_agent.yaml
		  $ northau-cli ../examples/fake_claude_code/cc_agent.yaml
	`,
	{
		importMeta: import.meta,
		flags: {},
	},
);

if (cli.input.length === 0) {
	console.error('Error: Please provide a path to a YAML configuration file');
	console.log('');
	cli.showHelp();
	process.exit(1);
}

const yamlPath = path.resolve(cli.input[0]);

// Check if the file exists
if (!fs.existsSync(yamlPath)) {
	console.error(`Error: File not found: ${yamlPath}`);
	process.exit(1);
}

// Check if it's a YAML file
if (!['.yaml', '.yml'].includes(path.extname(yamlPath).toLowerCase())) {
	console.error('Error: File must have .yaml or .yml extension');
	process.exit(1);
}

render(<App yamlPath={yamlPath} />);
