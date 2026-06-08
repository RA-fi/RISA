.PHONY: venv docker-up docker-build mcp

venv:
	./scripts/setup_venv.sh

docker-build:
	docker compose build --no-cache

docker-up:
	docker compose up -d --build

mcp:
	RISA_API_BASE=http://127.0.0.1:8000 python3 -m risa_mcp.server
