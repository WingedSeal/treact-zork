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
