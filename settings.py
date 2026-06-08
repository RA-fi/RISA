import os
from dotenv import load_dotenv, find_dotenv

# Only load .env file if it exists (for local development)
# Railway and other cloud platforms provide environment variables directly
try:
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
except Exception:
    # If dotenv loading fails, continue with system environment variables
    pass

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

_origins = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = [o.strip() for o in _origins.split(",") if o.strip()] or ["*"]

