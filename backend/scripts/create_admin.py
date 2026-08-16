"""One-time script to create/reset the single admin account.

Usage: python scripts/create_admin.py [username]
Prompts for password via getpass (not stored in shell history).
Safe to re-run: upserts by username, resetting the password if the account
already exists.
"""
import asyncio
import getpass
import os
import sys

import asyncpg
import bcrypt
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    username = sys.argv[1] if len(sys.argv) > 1 else input("Username: ")
    password = getpass.getpass("Password: ")
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        await conn.execute(
            """
            INSERT INTO users (username, password_hash) VALUES ($1, $2)
            ON CONFLICT (username) DO UPDATE SET password_hash = EXCLUDED.password_hash
            """,
            username,
            password_hash,
        )
        print(f"Admin user '{username}' created/updated.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
