#!/usr/bin/env python3
"""Generate a secure admin token for LEARNED_GUIDANCE_ADMIN_TOKEN.

Usage:
  python3 scripts/generate_token.py [--length 32]

The script prints a single token to stdout. Do NOT commit tokens to source control.
"""
import argparse
import secrets


def make_token(length: int) -> str:
    return secrets.token_urlsafe(length)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a secure admin token for RISA")
    p.add_argument("--length", type=int, default=32, help="entropy bytes for the token (default: 32)")
    args = p.parse_args()
    token = make_token(args.length)
    print(token)


if __name__ == "__main__":
    main()
