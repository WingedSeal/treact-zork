# Zork-Tool

## Requirements

- Download frotz cli
- `pip install -r requirements.txt`
- Download zork games from `sources.txt`

## Example Usage

### Server

```
fastapi dev ./src/main.py
python ./src/main.py
```

### Windows

```
curl -X POST "http://127.0.0.1:8000/zork/zork285" -H "Content-Type: application/json" -d "{\"commands\": [\"look\",\"inventory\",\"go east\",\"go north\"]}"

curl "http://127.0.0.1:8000/gen_key/zork285"

curl -X POST "http://127.0.0.1:8000/use_key/zork285" -H "Content-Type: application/json" -d "{\"command\":\"look\",\"key\":\"49e6\"}"
```

### Linux

```
curl -X POST "http://127.0.0.1:8000/zork/zork285" \
  -H "Content-Type: application/json" \
  -d '{"commands": ["look", "inventory", "go east", "go north"]}'
```

### How to setup

1. cd mcp-server
2. pip install uv
3. uv venv. venv
4. uv add -r requirements.txt
5. uv sync

### How to run

1. Open the first terminal

- uv run zork-tool\src\main.py

2. Open the second terminal

- uv run mcp-server\src\server.py

3. Open the third terminal

- uv run mcp-client\src\client.py
