"""
MARKET SUMMARY IHSG - dikirim ke Telegram jam 18:00 WIB (setelah market tutup)
============================================================================
Berisi: nilai IHSG penutupan, perubahan poin & persen, volume & value transaksi.

Catatan: breakdown net foreign/domestic buy-sell yang detail (seperti contoh
RTI/sekuritas) TIDAK disertakan di sini karena datanya tidak tersedia via
API publik gratis yang reliable - itu perlu data feed berbayar/scraping
yang rapuh. Untuk info itu, tetap cek app sekuritas kamu.

Setup jadwal terpisah di cron-job.org untuk jam 18:00 WIB, memicu workflow
GitHub Actions yang menjalankan file ini (lihat market_summary.yml).
"""

import yfinance as yf
import requests
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "ISI_TOKEN_BOT_KAMU_DISINI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "ISI_CHAT_ID_KAMU_DISINI")


def waktu_wib():
    return datetime.now(timezone.utc) + timedelta(hours=7)


def kirim_telegram(pesan):
    if TELEGRAM_TOKEN == "ISI_TOKEN_BOT_KAMU_DISINI":
        print("[INFO] Telegram belum dikonfigurasi:\n")
        print(pesan)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": pesan, "parse_mode": "HTML"})
    except Exception as e:
        print(f"Gagal kirim Telegram: {e}")


def ambil_ringkasan_ihsg():
    """Ambil data IHSG (^JKSE) untuk hari terakhir yang tersedia"""
    df = yf.download("^JKSE", period="5d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest["Close"]
    prev_close = prev["Close"]
    perubahan_poin = close - prev_close
    perubahan_pct = (perubahan_poin / prev_close) * 100
    volume = latest["Volume"]

    return {
        "tanggal": df.index[-1].strftime("%d %b %Y"),
        "close": close,
        "perubahan_poin": perubahan_poin,
        "perubahan_pct": perubahan_pct,
        "volume": volume,
        "high": latest["High"],
        "low": latest["Low"],
    }


def run_market_summary():
    print(f"Mengambil ringkasan IHSG - {waktu_wib().strftime('%Y-%m-%d %H:%M')}")

    try:
        data = ambil_ringkasan_ihsg()
    except Exception as e:
        print(f"Gagal ambil data IHSG: {e}")
        kirim_telegram(f"⚠️ Gagal mengambil data market summary hari ini.\nError: {e}")
        return

    arah = "🟢" if data["perubahan_poin"] >= 0 else "🔴"
    tanda = "+" if data["perubahan_poin"] >= 0 else ""

    pesan = f"""<b>📊 MARKET SUMMARY IHSG - {data['tanggal']}</b>

{arah} IHSG: <b>{data['close']:,.2f}</b> ({tanda}{data['perubahan_poin']:,.2f} / {tanda}{data['perubahan_pct']:.2f}%)

Tertinggi: {data['high']:,.2f}
Terendah: {data['low']:,.2f}
Volume: {data['volume']/1e9:.2f} Miliar lembar

<i>Catatan: breakdown net foreign/domestic tidak tersedia otomatis - cek app sekuritas untuk detail itu.</i>
"""

    print(pesan)
    kirim_telegram(pesan)


if __name__ == "__main__":
    run_market_summary()
