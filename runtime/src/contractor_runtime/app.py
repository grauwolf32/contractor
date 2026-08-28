"""ASGI application for the Runtime Agent process."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def create_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/readyz", health, methods=["GET"]),
        ]
    )
