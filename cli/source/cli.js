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
		  $ nexau-cli <yaml-file>

		Arguments
		  <yaml-file>  Path to the NexAU agent configuration YAML file

		Options
		  --help       Show this help message

		Examples
		  $ nexau-cli ./my_agent.yaml
		  $ nexau-cli ../examples/code_agent/cc_agent.yaml
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
