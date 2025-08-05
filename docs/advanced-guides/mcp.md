
### MCP

Define the mcp servers and create agent by setting `mcp_servers` as follows:

```python
llm_config = LLMConfig(
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

mcp_servers = [
    {
        "name": "amap-maps-streamableHTTP",
        "type": "http",
        "url": "https://mcp.amap.com/mcp?key=xxx",
        "headers": {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        },
        "timeout": 10
    }
]

agent = create_agent(
    name="amap_agent",
    system_prompt="""You are an AI agent with access to Amap Maps services through MCP.

You can use Amap Maps tools to:
- Search for locations and points of interest
- Get directions and navigation information
- Calculate distances and travel times
- Find nearby businesses and services
- Access real-time traffic information
- And other location-based services

When using map tools, always provide clear and helpful information to users.
Explain what you're doing and provide context for the results.""",
    mcp_servers=mcp_servers,
    llm_config=llm_config,
)

response = agent.run("现在从漕河泾现代服务园A6到上南路 4265弄要多久？")
print(response)
```

If you use YAML-based agent config, you can add:
```yaml
mcp_servers:
  - name: github
    type: stdio
    command: npx
    args: ['-y', '@modelcontextprotocol/server-github']
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "xxxx"
    timeout: 30
  
  - name: amap-maps
    type: http
    url: "https://mcp.amap.com/mcp?key=your_amap_key_here"
    headers:
      Content-Type: "application/json"
      Accept: "application/json, text/event-stream"
    timeout: 30
```
