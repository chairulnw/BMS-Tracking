import asyncpg
from fastapi import Request


async def get_pool(request: Request) -> asyncpg.Pool:
    return request.app.state.pool
