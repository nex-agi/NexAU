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

import React from 'react';
import chalk from 'chalk';
import test from 'ava';
import {render} from 'ink-testing-library';
import App from './source/app.js';

test('greet unknown user', t => {
	const {lastFrame} = render(<App />);

	t.is(lastFrame(), `Hello, ${chalk.green('Stranger')}`);
});

test('greet user with a name', t => {
	const {lastFrame} = render(<App name="Jane" />);

	t.is(lastFrame(), `Hello, ${chalk.green('Jane')}`);
});
