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
curl "http://127.0.0.1:8000/dict/zork285?types=true"

curl -X POST "http://127.0.0.1:8000/use_key/zork285" -H "Content-Type: application/json" -d "{\"command\":\"look\",\"key\":\"49e6\"}"
```

### Linux

```
curl "http://127.0.0.1:8000/gen_key/zork285"

curl "http://127.0.0.1:8000/dict/zork285"
curl "http://127.0.0.1:8000/dict/zork285?types=true"

curl -X POST "http://127.0.0.1:8000/zork/zork285" \
  -H "Content-Type: application/json" \
  -d '{"commands": "look", "key":"49e6"}'
```
