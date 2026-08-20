"""Telegram notification untuk event critical (plan2/spesifikasi.md Fase 1).
Gratis, tanpa verifikasi bisnis — buat bot lewat @BotFather, isi TELEGRAM_BOT_TOKEN
+ TELEGRAM_CHAT_ID di .env. Kalau salah satu kosong, notify jadi no-op."""

import os

import httpx

_BOT_TOKEN = lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID   = lambda: os.getenv("TELEGRAM_CHAT_ID", "")


async def notify_telegram(text: str) -> None:
    """Best-effort — gagal kirim tidak boleh menggagalkan request yang memicunya."""
    token, chat_id = _BOT_TOKEN(), _CHAT_ID()
    if not token or not chat_id:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
            )
    except Exception as exc:
        print(f"[notify] gagal kirim Telegram: {exc}")
