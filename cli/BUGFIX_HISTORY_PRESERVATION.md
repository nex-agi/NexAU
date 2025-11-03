# Bug Fix: Action History Preservation

## Issue

After implementing action history preservation, the steps were not actually being preserved. They disappeared after task completion, showing only the agent's final response.

## Root Cause

**React Closure Problem**: When the `response` message handler executed, it captured `currentSteps` from the closure scope. Due to React's asynchronous state updates and closure behavior, this value was stale/empty when the response came in.

```javascript
// BEFORE (Buggy Code)
case 'response':
  setMessages(prev => [...prev, {
    role: 'assistant',
    content: message.content,
    steps: currentSteps,  // ❌ Stale closure value - often empty!
  }]);
  setCurrentSteps([]);
  break;
```

**Problem Flow**:
1. Steps come in → `setCurrentSteps()` called
2. Response arrives quickly
3. Response handler reads `currentSteps` from closure
4. But the state hasn't updated yet in that closure scope
5. Result: empty array captured

## Solution

**Use React Ref for Immediate Access**: Add a `useRef` to track steps synchronously, bypassing React's async state updates.

```javascript
// AFTER (Fixed Code)
const currentStepsRef = useRef([]);

case 'step':
  const newStep = {content: message.content, metadata: message.metadata};
  currentStepsRef.current = [...currentStepsRef.current, newStep];  // ✅ Immediate update
  setCurrentSteps(currentStepsRef.current);  // Also update state for rendering
  break;

case 'response':
  setMessages(prev => [...prev, {
    role: 'assistant',
    content: message.content,
    steps: [...currentStepsRef.current],  // ✅ Always current value!
  }]);
  currentStepsRef.current = [];  // Clear ref
  setCurrentSteps([]);            // Clear state
  break;
```

## Key Changes

### 1. Added `currentStepsRef`
```javascript
const currentStepsRef = useRef([]);
```

### 2. Update Both Ref and State
Every time a step comes in:
```javascript
currentStepsRef.current = [...currentStepsRef.current, newStep];
setCurrentSteps(currentStepsRef.current);
```

### 3. Read from Ref in Response Handler
```javascript
steps: [...currentStepsRef.current]  // Current value, not closure value
```

### 4. Clear Both on Submit
```javascript
currentStepsRef.current = [];
setCurrentSteps([]);
```

## Visual Enhancement

Added a separator line between steps and final response for better clarity:

```javascript
<Box marginTop={0}>
  <Text dimColor>─────</Text>
</Box>
```

## Expected Behavior Now

```
❯ You:
  Write hello.py

⚡ Agent:
  💭 I'll create a Python hello world program...
  🔧 Planning to execute 1 tool(s): file_write
  ✓ Tool 'file_write' completed
  ─────
  I've created hello.py successfully!

❯ You:
  Run it

⚡ Agent:
  💭 I'll execute the hello.py file...
  🔧 Planning to execute 1 tool(s): bash
  ✓ Tool 'bash' completed
  ─────
  Output: Hello, World!
```

**Note**: All steps remain visible and can be scrolled back to view!

## Technical Details

### Why useRef Works

- **Synchronous**: Ref updates are immediate, no batching
- **Persistent**: Value persists across renders
- **No re-render**: Changing `.current` doesn't trigger re-render
- **Closure-safe**: Always reads current value, not captured value

### Why State Alone Failed

- **Asynchronous**: State updates are batched/delayed
- **Closure capture**: Functions capture state at closure creation time
- **Stale values**: Fast message sequences can read old values

## Testing

To verify the fix:

```bash
cd cli
npm run build
./dist/cli.js ../examples/fake_claude_code/cc_agent.yaml

# Test with a multi-step task
> Write a hello world program in Python

# You should see:
# 1. Agent reasoning (💭)
# 2. Tool planning (🔧)
# 3. Tool execution (✓)
# 4. Separator line (─────)
# 5. Final response

# Try another task
> Run the program

# Verify: 
# - Previous task steps are still visible above
# - New task steps are shown
# - Full history preserved
```

## Files Modified

- `cli/source/app.js`:
  - Added `currentStepsRef` 
  - Updated step handlers to use ref
  - Updated response handler to read from ref
  - Added visual separator

## Lesson Learned

**When dealing with fast message streams in React:**
- Use `useRef` for synchronous tracking
- Use state for rendering
- Update both to maintain consistency
- Read from ref when timing matters

---

**Status**: ✅ Fixed  
**Date**: November 2, 2025  
**Build**: Required after fix

