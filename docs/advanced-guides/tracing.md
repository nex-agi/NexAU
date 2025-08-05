
### Dump trace to file

In addition to langfuse trace, you can also dump trace to files by setting `dump_trace_path` in `agent.run`, the input and output of each round of main agent is dump to `dump_trace_path` and the trace of sub-agent will be saved in to a subfolder with the same name of dump_trace_path (without .json extention).

```python
response = agent.run(input_message, dump_trace_path="test_log.json")
```

```
|- test_log.json
|- test_log
    |- sub_agent_1.json
    |- sub_agent_2.json
```

