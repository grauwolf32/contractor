#!/usr/bin/env python3
"""Exchange a Caido PAT for an access token and print it.

Usage:
    python scripts/caido_auth.py <PAT> [--url http://127.0.0.1:8080]

The access token is printed to stdout. Pipe it into your .env:
    echo "CAIDO_AUTH_TOKEN=$(python scripts/caido_auth.py caido_xxx)" >> cli/.env
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import httpx

CAIDO_CLOUD_API = "https://api.caido.io"
AUTH_SCOPES = "profile:read,offline"

START_AUTH_FLOW = """
mutation {
  startAuthenticationFlow {
    error {
      ... on AuthenticationUserError { code reason }
      ... on CloudUserError { code }
      ... on OtherUserError { code }
    }
    request { id userCode verificationUrl expiresAt }
  }
}
"""

CREATED_AUTH_TOKEN_SUB = """
subscription CreatedAuthenticationToken($requestId: ID!) {
  createdAuthenticationToken(requestId: $requestId) {
    token {
      accessToken
      expiresAt
      refreshToken
      scopes
    }
  }
}
"""


async def exchange(instance_url: str, pat: str, timeout: float = 30.0) -> str:
    import websockets

    graphql_url = f"{instance_url.rstrip('/')}/graphql"
    ws_url = (
        instance_url.rstrip("/")
        .replace("http://", "ws://")
        .replace("https://", "wss://")
        + "/ws/graphql"
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), verify=False) as client:
        # Step 1: start auth flow on local instance
        resp = await client.post(graphql_url, json={"query": START_AUTH_FLOW})
        resp.raise_for_status()
        body = resp.json()
        flow = body.get("data", {}).get("startAuthenticationFlow", {})
        if flow.get("error"):
            print(f"error: failed to start auth flow: {flow['error']}", file=sys.stderr)
            sys.exit(1)
        request = flow.get("request")
        if not request:
            print("error: no auth request returned", file=sys.stderr)
            sys.exit(1)
        request_id = request["id"]
        user_code = request["userCode"]
        print(f"auth flow started (userCode={user_code})", file=sys.stderr)

        # Step 2: approve via cloud API with PAT
        approve_url = (
            f"{CAIDO_CLOUD_API}/oauth2/device/approve"
            f"?user_code={user_code}&scope={AUTH_SCOPES}"
        )
        approve_resp = await client.post(
            approve_url,
            headers={"Authorization": f"Bearer {pat}"},
        )
        if approve_resp.status_code >= 400:
            print(
                f"error: cloud API rejected PAT ({approve_resp.status_code}): "
                f"{approve_resp.text[:200]}",
                file=sys.stderr,
            )
            sys.exit(1)
        print("device code approved", file=sys.stderr)

    # Step 3: subscribe via WebSocket for the access token
    sub_payload = {
        "id": str(uuid.uuid4()),
        "type": "subscribe",
        "payload": {
            "query": CREATED_AUTH_TOKEN_SUB,
            "variables": {"requestId": request_id},
        },
    }

    async with websockets.connect(
        ws_url,
        subprotocols=["graphql-transport-ws"],
        additional_headers={},
    ) as ws:
        await ws.send(json.dumps({"type": "connection_init"}))
        init_ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if init_ack.get("type") != "connection_ack":
            print(f"error: WebSocket init failed: {init_ack}", file=sys.stderr)
            sys.exit(1)

        await ws.send(json.dumps(sub_payload))

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if msg.get("type") == "next":
                token_data = (
                    msg.get("payload", {})
                    .get("data", {})
                    .get("createdAuthenticationToken", {})
                    .get("token")
                )
                if token_data and token_data.get("accessToken"):
                    print(
                        f"token obtained (expires {token_data.get('expiresAt', '?')})",
                        file=sys.stderr,
                    )
                    return token_data["accessToken"]
            elif msg.get("type") == "error":
                print(f"error: subscription error: {msg.get('payload')}", file=sys.stderr)
                sys.exit(1)

    print("error: timed out waiting for token", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exchange a Caido PAT for an access token")
    parser.add_argument("pat", help="Caido PAT (starts with caido_)")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Caido instance URL")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if not args.pat.startswith("caido_"):
        print("warning: PAT doesn't start with 'caido_', proceeding anyway", file=sys.stderr)

    token = asyncio.run(exchange(args.url, args.pat, args.timeout))
    print(token)


if __name__ == "__main__":
    main()
