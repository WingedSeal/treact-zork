# Zork-Tool

## Requirements

- Download frotz cli
  - If you are on Windows, download from `sources.txt` and put it in `./zork-tool/` and rename it to `frotz.exe`
- `pip install -r requirements.txt`
- Download zork games from `sources.txt` and put them in `./zork-tool/games/`

## Example Usage

### Server

```
cd zork-tool
fastapi dev ./src/main.py
# OR
python ./src/main.py
```

### Windows

```

curl "http://127.0.0.1:8000/gen_key/zork285"

curl "http://127.0.0.1:8000/dict/zork285"
curl "http://127.0.0.1:8000/dict_with_types/zork285"

curl -X POST "http://127.0.0.1:8000/use_key/zork285" -H "Content-Type: application/json" -d "{\"command\":\"look\",\"key\":\"49e6\"}"
```

### Linux

```
curl "http://127.0.0.1:8000/gen_key/zork285"

curl "http://127.0.0.1:8000/dict/zork285"
curl "http://127.0.0.1:8000/dict_with_types/zork285"

curl -X POST "http://127.0.0.1:8000/use_key/zork285" -H "Content-Type: application/json" -d '{"command": "look", "key":"49e6"}'
```

### How to setup

1. cd mcp-server
2. pip install uv
3. uv init
4. uv venv .venv
5. uv add -r requirements.txt
6. uv sync

Don't forget to delete README and main.py in mcp-server folder

### How to run

1. Open the first terminal

- uv run zork-tool\src\main.py

2. Open the second terminal

- uv run mcp-server\src\server.py

3. Open the third terminal

- uv run mcp-client\src\client.py
