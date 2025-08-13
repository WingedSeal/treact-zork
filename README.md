# Zork-Tool

## Requirements

- Download frotz cli
- `pip install -r requirements.txt`
- Download zork games from `sources.txt`

## Example Usage

### Server

```
fastapi dev ./src/server.py
```

### Windows

```
curl -X POST "http://127.0.0.1:8000/zork/zork285" -H "Content-Type: application/json" -d "{\"commands\": [\"look\",\"inventory\",\"go east\",\"go north\"]}"
```

### Linux

```
curl -X POST "http://127.0.0.1:8000/zork/zork285" \
  -H "Content-Type: application/json" \
  -d '{"commands": ["look", "inventory", "go east", "go north"]}'
```

### UV

```
uv init
uv .venv
uv add requirements.txt
```

### MCP Server

```
uv run mcp dev mcp_server.py
```

### Server (Zork)

```
uv run server.pu
```

### LLM (Gemini 2.5 Flash)

```
require access_key.json
```
