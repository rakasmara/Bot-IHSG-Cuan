"""
MARKET SUMMARY IHSG - dikirim ke Telegram jam 18:00 WIB (setelah market tutup)
============================================================================
Berisi: nilai IHSG penutupan, perubahan poin & persen, volume & value transaksi,
dan indikator valuasi PROXY (Murah/Normal/Kemahalan).

PENTING soal indikator valuasi di bawah - baca sebelum dipakai:
Ini BUKAN valuasi fundamental asli (bukan PBV/PER). Valuasi asli butuh data
agregat laba & ekuitas seluruh emiten IHSG yang TIDAK tersedia gratis via
API publik yang reliable (biasanya cuma ada di Bloomberg/riset sekuritas,
di-update bulanan). Index level mentah (misal "9.000 = mahal") JUGA BUKAN
patokan valuasi yang valid, karena nilai wajar pasar naik seiring waktu
selama laba/ekuitas emiten tumbuh - level yang "mahal" tahun ini bisa jadi
"murah" tahun depan kalau laba emiten naik lebih cepat dari harga.

Yang dipakai di sini: PERSENTIL HARGA PENUTUPAN IHSG dibanding 5 tahun
riwayatnya sendiri. Ini proxy TEKNIKAL murni (posisi harga relatif
terhadap rentang historisnya), bukan valuasi fundamental:
- Persentil rendah (harga dekat titik terendah 5 tahun) -> "Murah" (proxy)
- Persentil tengah -> "Normal" (proxy)
- Persentil tinggi (harga dekat titik tertinggi 5 tahun) -> "Kemahalan" (proxy)

Keterbatasan yang perlu disadari:
- Proxy ini TIDAK memperhitungkan pertumbuhan laba/ekuitas emiten sama
  sekali - murni posisi harga, sama seperti membandingkan harga saham
  individual terhadap rentang 52 minggunya.
- Kalau IHSG sedang dalam tren struktural naik jangka panjang (rebasing
  akibat pertumbuhan ekonomi), proxy ini bisa terus-terusan bilang
  "Kemahalan" padahal secara PBV/PER sebenarnya masih wajar/murah -
  seperti kasus 2026 di mana IHSG pernah dibilang "murah" oleh analis
  (PBV/PER rendah) walau secara harga nominal sudah lebih tinggi dari
  rata-rata historisnya.
- Untuk keputusan investasi yang serius, tetap cek data PBV/PER asli dari
  riset sekuritas (BNI Sekuritas, Bloomberg, dsb) - proxy ini cuma
  pelengkap cepat, bukan pengganti.
============================================================================
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

# --- KONFIGURASI PROXY VALUASI ---
VALUASI_LOOKBACK_PERIOD = "5y"   # rentang historis untuk hitung persentil
PERSENTIL_MURAH = 30             # di bawah ini dianggap "Murah" (proxy)
PERSENTIL_KEMAHALAN = 70         # di atas ini dianggap "Kemahalan" (proxy)


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


def hitung_proxy_valuasi():
    """Hitung persentil harga closing IHSG hari ini dibanding riwayat 5 tahun.
    Return dict berisi persentil dan label, atau None kalau data gagal diambil."""
    try:
        df_panjang = yf.download("^JKSE", period=VALUASI_LOOKBACK_PERIOD, progress=False)
        if isinstance(df_panjang.columns, pd.MultiIndex):
            df_panjang.columns = df_panjang.columns.get_level_values(0)

        if df_panjang.empty or len(df_panjang) < 250:  # minimal ~1 tahun data
            return None

        close_sekarang = df_panjang["Close"].iloc[-1]
        # Persentil: berapa % dari hari-hari historis yang closingnya LEBIH RENDAH
        # dari harga sekarang. 90 berarti harga sekarang lebih tinggi dari 90%
        # hari-hari dalam 5 tahun terakhir.
        persentil = (df_panjang["Close"] < close_sekarang).mean() * 100

        if persentil <= PERSENTIL_MURAH:
            label = "MURAH (proxy)"
            emoji = "🟢"
        elif persentil >= PERSENTIL_KEMAHALAN:
            label = "KEMAHALAN (proxy)"
            emoji = "🔴"
        else:
            label = "NORMAL (proxy)"
            emoji = "🟡"

        return {
            "persentil": round(persentil, 1),
            "label": label,
            "emoji": emoji,
            "titik_terendah_5th": round(df_panjang["Close"].min(), 0),
            "titik_tertinggi_5th": round(df_panjang["Close"].max(), 0),
        }
    except Exception as e:
        print(f"Gagal hitung proxy valuasi: {e}")
        return None


def run_market_summary():
    print(f"Mengambil ringkasan IHSG - {waktu_wib().strftime('%Y-%m-%d %H:%M')}")
    try:
        data = ambil_ringkasan_ihsg()
    except Exception as e:
        print(f"Gagal ambil data IHSG: {e}")
        kirim_telegram(f"⚠️ Gagal mengambil data market summary hari ini.\nError: {e}")
        return

    valuasi = hitung_proxy_valuasi()  # boleh None kalau gagal, pesan tetap dikirim tanpa bagian ini

    arah = "🟢" if data["perubahan_poin"] >= 0 else "🔴"
    tanda = "+" if data["perubahan_poin"] >= 0 else ""

    pesan = f"""<b>📊 MARKET SUMMARY IHSG - {data['tanggal']}</b>

{arah} IHSG: <b>{data['close']:,.2f}</b> ({tanda}{data['perubahan_poin']:,.2f} / {tanda}{data['perubahan_pct']:.2f}%)

Tertinggi: {data['high']:,.2f}
Terendah: {data['low']:,.2f}
Volume: {data['volume']/1e9:.2f} Miliar lembar
"""

    if valuasi:
        pesan += f"""
{valuasi['emoji']} Valuasi harga (persentil 5 tahun): <b>{valuasi['label']}</b>
Persentil: {valuasi['persentil']}% (0%=terendah 5th, 100%=tertinggi 5th)
Rentang 5th: {valuasi['titik_terendah_5th']:,.0f} - {valuasi['titik_tertinggi_5th']:,.0f}
<i>Proxy posisi harga saja, BUKAN valuasi PBV/PER asli - lihat catatan di kode.</i>
"""

    pesan += "\n<i>Catatan: breakdown net foreign/domestic tidak tersedia otomatis - cek app sekuritas untuk detail itu.</i>\n"

    print(pesan)
    kirim_telegram(pesan)


if __name__ == "__main__":
    run_market_summary()
