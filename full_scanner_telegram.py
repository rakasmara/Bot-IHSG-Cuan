"""
FULL MARKET SCANNER IHSG + TELEGRAM ALERT (v2 - 20 Sept 2026)
============================================================================
PERUBAHAN BESAR dari versi sebelumnya, berdasarkan hasil ~20 backtest kita:

DIHAPUS:
- ARA Detector (kena Stop-Loss 68-74% di backtest, memicu FOMO)
- Akumulasi OBV standalone (diganti CMF+konfirmasi harga naik di bawah)

DITAMBAHKAN (JUJUR: TIDAK ada yang terbukti punya edge kuat di backtest -
paling bagus MACD Histogram+Rezim di +0,16% sebelum fee, sisanya di bawah
breakeven. Ini dipasang atas permintaan eksplisit untuk uji coba psikologis/
money-management, BUKAN karena terbukti profitable):
- Stochastic(6,3,3) + EMA(6,18) Golden Cross BERSAMAAN
- Ichimoku Kinko Hyo versi RINGKAS (3 syarat, tanpa Chikou Span)
- MACD Histogram Divergence + Filter Rezim Bullish IHSG (hasil backtest
  TERBAIK kita, tapi tetap breakeven-ish setelah fee)
- CMF Akumulasi Terkonfirmasi DIREVISI: sekarang wajib ada KONFIRMASI HARGA
  NAIK + VOLUME NAIK (bukan cuma "akumulasi diam-diam" seperti sebelumnya)
  - CATATAN: "Frequency Analyzer" (jumlah transaksi/hari) yang diminta TIDAK
    BISA dibuat - yfinance tidak punya data itu. Ini pakai proxy Volume+Harga
    saja. Kalau mau data Frequency asli, perlu cek manual di ihsgscreener/
    Stockbit dan silangkan sendiri dengan hasil bot ini.

SEMUA kategori baru ini TETAP PAKAI filter likuiditas & nilai transaksi yang
sama seperti sebelumnya (menghindari kasus AMAG-style saham tipis).

JADWAL SCAN: diatur di file GitHub Actions (.yml), BUKAN di script ini. Untuk
swing trading, tidak perlu tiap 30 menit - lihat rekomendasi di README/pesan
terakhir dari Claude soal ini (misal: 2x/hari jam 10:30 & 14:30 WIB).
============================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")


def waktu_wib():
    return datetime.now(timezone.utc) + timedelta(hours=7)


# ============================================================
# 1. KONFIGURASI
# ============================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "ISI_TOKEN_BOT_KAMU_DISINI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "ISI_CHAT_ID_KAMU_DISINI")

MIN_HARGA = 50
MIN_VOLUME_HARIAN = 100000
MIN_NILAI_TRANSAKSI_HARIAN = 2_000_000_000

DAFTAR_SAHAM_FILE = "Daftar_Saham_Idx.csv"
DAFTAR_SAHAM_KOLOM = "Kode"
GUNAKAN_DAFTAR_MANUAL_DULU = False
DAFTAR_MANUAL = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP", "INDF",
    "ANTM", "MDKA", "INKP", "TPIA", "BRPT", "CUAN", "AMMN", "ADRO", "PTBA",
    "GEMS", "MBMA", "NCKL", "PANI", "BREN", "BRMS", "TINS", "MEDC", "PGAS", "AKRA",
]

# --- Confluence lama (Stoch + Supertrend + BB) ---
STOCH_K, STOCH_SMOOTH, STOCH_D = 6, 3, 3
ST_ATR_PERIOD, ST_MULTIPLIER = 10, 1
BB_PERIOD, BB_STD = 20, 2
BB_LOWER_THRESHOLD_PCT = 1.0
MIN_SKOR_ALERT = 3

VOL_SPIKE_MULTIPLIER = 3.0
VOL_LOOKBACK = 20
EARLY_VOL_MULTIPLIER = 2.5
EARLY_MAX_KENAIKAN_PCT = 8
EARLY_MIN_KENAIKAN_PCT = -5

# --- CMF Akumulasi (DIREVISI - wajib harga & volume naik, bukan flat) ---
CMF_PERIOD = 20
CMF_THRESHOLD = 0.1
KONFIRMASI_LOOKBACK = 5          # cek kenaikan harga N hari terakhir
KONFIRMASI_MIN_KENAIKAN_PCT = 2  # harga wajib naik minimal segini % (bukan flat)
KONFIRMASI_MIN_VOL_RATIO = 1.5   # volume wajib di atas rata-rata segini x

# --- Stochastic + MA Golden Cross bersamaan (BARU) ---
MA_FAST_PERIOD = 6    # selaras dengan periode Stochastic (6,3,3)
MA_SLOW_PERIOD = 18   # 3x MA_FAST - rasio umum untuk golden cross jangka pendek
GUNAKAN_EMA = True    # True = Exponential MA, False = Simple MA

# --- Ichimoku Ringkas (BARU - 3 syarat, Chikou Span dibuang karena paling lambat) ---
TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, DISPLACEMENT = 9, 26, 52, 26

# --- MACD Histogram Divergence + Filter Rezim (BARU - hasil backtest terbaik) ---
DIVERGENCE_LOOKBACK = 10
REZIM_MA_IHSG = 100

# --- Persistensi sinyal ---
FOLDER_HASIL_SCAN = "."
LOOKBACK_DAYS = "6mo"
JEDA_ANTAR_REQUEST = 0.3


# ============================================================
# 2. FUNGSI INDIKATOR
# ============================================================

def calculate_stochastic(df, k_period=6, smooth_k=3, d_period=3):
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    raw_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    k = raw_k.rolling(window=smooth_k).mean()
    d = k.rolling(window=d_period).mean()
    return k, d


def calculate_supertrend(df, atr_period=10, multiplier=1):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
            lower_band.iloc[i] = lower_band.iloc[i - 1]
        if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
            upper_band.iloc[i] = upper_band.iloc[i - 1]

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend, direction


def calculate_bollinger(df, period=20, std_mult=2):
    mid = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def calculate_money_flow_multiplier(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    range_hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / range_hl
    return mfm.fillna(0)


def calculate_cmf(df, period=20):
    mfm = calculate_money_flow_multiplier(df)
    mfv = mfm * df["Volume"]
    return mfv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()


def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_ichimoku(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tenkan = (high.rolling(TENKAN_PERIOD).max() + low.rolling(TENKAN_PERIOD).min()) / 2
    kijun = (high.rolling(KIJUN_PERIOD).max() + low.rolling(KIJUN_PERIOD).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (high.rolling(SENKOU_B_PERIOD).max() + low.rolling(SENKOU_B_PERIOD).min()) / 2
    return {
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a_raw": senkou_a_raw, "senkou_b_raw": senkou_b_raw,
        "senkou_a_plotted": senkou_a_raw.shift(DISPLACEMENT),
        "senkou_b_plotted": senkou_b_raw.shift(DISPLACEMENT),
    }


# ============================================================
# 3. REZIM IHSG (dihitung SEKALI per scan, bukan per saham)
# ============================================================

def hitung_rezim_ihsg():
    try:
        df_ihsg = yf.download("^JKSE", period=LOOKBACK_DAYS, progress=False)
        if isinstance(df_ihsg.columns, pd.MultiIndex):
            df_ihsg.columns = df_ihsg.columns.get_level_values(0)
        if df_ihsg.empty or len(df_ihsg) < REZIM_MA_IHSG:
            # Kalau data IHSG tidak cukup panjang (misal LOOKBACK_DAYS terlalu pendek
            # untuk MA100), ambil histori lebih panjang khusus untuk IHSG
            df_ihsg = yf.download("^JKSE", period="1y", progress=False)
            if isinstance(df_ihsg.columns, pd.MultiIndex):
                df_ihsg.columns = df_ihsg.columns.get_level_values(0)
        ma = df_ihsg["Close"].rolling(REZIM_MA_IHSG).mean()
        uptrend = df_ihsg["Close"] > ma
        rezim_sekarang = bool(uptrend.iloc[-1]) if len(uptrend) > 0 and pd.notna(uptrend.iloc[-1]) else False
        return uptrend, rezim_sekarang
    except Exception as e:
        print(f"Gagal hitung rezim IHSG: {e} - anggap TIDAK uptrend (aman/konservatif)")
        return pd.Series(dtype=bool), False


# ============================================================
# 4. AMBIL DAFTAR SAHAM
# ============================================================

def get_watchlist():
    if GUNAKAN_DAFTAR_MANUAL_DULU:
        print(f"Menggunakan daftar manual: {len(DAFTAR_MANUAL)} saham")
        return [f"{kode}.JK" for kode in DAFTAR_MANUAL]
    try:
        df_saham = pd.read_csv(DAFTAR_SAHAM_FILE)
        kode_list = df_saham[DAFTAR_SAHAM_KOLOM].astype(str).str.strip().tolist()
        kode_list = [k for k in kode_list if k and k.lower() != "nan"]
        print(f"Berhasil load {len(kode_list)} saham dari {DAFTAR_SAHAM_FILE}")
        return [f"{kode}.JK" for kode in kode_list]
    except Exception as e:
        print(f"Gagal load file ({e}), fallback ke daftar manual")
        return [f"{kode}.JK" for kode in DAFTAR_MANUAL]


# ============================================================
# 5. ANALISIS PER SAHAM
# ============================================================

def analyze_ticker(ticker, ihsg_rezim_sekarang):
    try:
        df = yf.download(ticker, period=LOOKBACK_DAYS, progress=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Stoch_K"], df["Stoch_D"] = calculate_stochastic(df, STOCH_K, STOCH_SMOOTH, STOCH_D)
        df["Supertrend"], df["ST_Direction"] = calculate_supertrend(df, ST_ATR_PERIOD, ST_MULTIPLIER)
        df["BB_upper"], df["BB_mid"], df["BB_lower"] = calculate_bollinger(df, BB_PERIOD, BB_STD)
        df["CMF"] = calculate_cmf(df, CMF_PERIOD)
        df["Vol_avg"] = df["Volume"].rolling(window=VOL_LOOKBACK).mean()

        if GUNAKAN_EMA:
            df["MA_Fast"] = df["Close"].ewm(span=MA_FAST_PERIOD, adjust=False).mean()
            df["MA_Slow"] = df["Close"].ewm(span=MA_SLOW_PERIOD, adjust=False).mean()
        else:
            df["MA_Fast"] = df["Close"].rolling(MA_FAST_PERIOD).mean()
            df["MA_Slow"] = df["Close"].rolling(MA_SLOW_PERIOD).mean()

        macd_line, macd_signal, macd_hist = calculate_macd(df)
        df["MACD_Hist"] = macd_hist

        ich = calculate_ichimoku(df)

        latest, prev = df.iloc[-1], df.iloc[-2]

        # --- Filter likuiditas UNIVERSAL (semua kategori) ---
        avg_vol_20 = df["Volume"].tail(20).mean()
        if latest["Close"] < MIN_HARGA or avg_vol_20 < MIN_VOLUME_HARIAN:
            return None
        nilai_transaksi_20 = (df["Close"].tail(20) * df["Volume"].tail(20)).mean()
        if nilai_transaksi_20 < MIN_NILAI_TRANSAKSI_HARIAN:
            return None

        # ---------- KATEGORI 1: Confluence Kuat (lama) ----------
        stoch_golden_cross = (prev["Stoch_K"] <= prev["Stoch_D"]) and (latest["Stoch_K"] > latest["Stoch_D"])
        supertrend_bullish = latest["ST_Direction"] == 1
        supertrend_baru_hijau = (prev["ST_Direction"] == -1) and (latest["ST_Direction"] == 1)
        jarak_lower_pct = ((latest["Close"] - latest["BB_lower"]) / latest["BB_lower"]) * 100
        di_lower_band = jarak_lower_pct <= BB_LOWER_THRESHOLD_PCT
        skor_confluence = sum([stoch_golden_cross, supertrend_bullish, di_lower_band])
        confluence_kuat = skor_confluence >= MIN_SKOR_ALERT

        # ---------- KATEGORI 2: Volume/Momentum & Deteksi Dini (lama) ----------
        vol_ratio = latest["Volume"] / latest["Vol_avg"] if latest["Vol_avg"] > 0 else 0
        volume_alert = vol_ratio >= VOL_SPIKE_MULTIPLIER
        harga_5hari_lalu = df["Close"].iloc[-6] if len(df) > 6 else df["Close"].iloc[0]
        kenaikan_5hari_pct = ((latest["Close"] - harga_5hari_lalu) / harga_5hari_lalu) * 100
        chart_naik_signifikan = kenaikan_5hari_pct >= 15
        deteksi_dini = (vol_ratio >= EARLY_VOL_MULTIPLIER) and (EARLY_MIN_KENAIKAN_PCT <= kenaikan_5hari_pct <= EARLY_MAX_KENAIKAN_PCT)

        # ---------- KATEGORI 3: CMF Akumulasi Terkonfirmasi (DIREVISI) ----------
        cmf_bullish = latest["CMF"] > CMF_THRESHOLD
        harga_naik_konfirmasi = False
        if len(df) > KONFIRMASI_LOOKBACK:
            harga_dulu_k = df["Close"].iloc[-(KONFIRMASI_LOOKBACK + 1)]
            kenaikan_konfirmasi_pct = ((latest["Close"] - harga_dulu_k) / harga_dulu_k) * 100
            harga_naik_konfirmasi = kenaikan_konfirmasi_pct >= KONFIRMASI_MIN_KENAIKAN_PCT
        else:
            kenaikan_konfirmasi_pct = 0
        volume_naik_konfirmasi = vol_ratio >= KONFIRMASI_MIN_VOL_RATIO
        cmf_akumulasi_terkonfirmasi = cmf_bullish and harga_naik_konfirmasi and volume_naik_konfirmasi

        # ---------- KATEGORI 4 (BARU): Stochastic + MA Golden Cross BERSAMAAN ----------
        ma_golden_cross = (prev["MA_Fast"] <= prev["MA_Slow"]) and (latest["MA_Fast"] > latest["MA_Slow"])
        double_golden_cross = stoch_golden_cross and ma_golden_cross

        # ---------- KATEGORI 5 (DIPERBAIKI): Ichimoku berbasis CROSSOVER, bukan status ----------
        # Temuan penting dari cross-check manual ke chart: versi lama pakai STATUS
        # ("apakah sekarang di atas") yang bikin telat masuk - sinyal baru muncul
        # setelah reli sudah jalan beberapa hari. Diperbaiki jadi CROSSOVER (titik
        # potong BARU hari ini), sesuai 3 syarat dari chart TradingView/Stockbit:
        # 1. Leading Span A memotong ke ATAS Leading Span B (cikal bakal Kumo bullish)
        # 2. Conversion Line (Tenkan) memotong ke ATAS Base Line (Kijun)
        # 3. Lagging Span (Chikou = harga close sekarang) memotong ke ATAS harga
        #    di posisi 26 hari lalu (mulai naik di atas chart)
        senkou_a_prev = ich["senkou_a_raw"].iloc[-2]
        senkou_b_prev = ich["senkou_b_raw"].iloc[-2]
        senkou_a_now = ich["senkou_a_raw"].iloc[-1]
        senkou_b_now = ich["senkou_b_raw"].iloc[-1]
        kumo_cross_up = (senkou_a_prev <= senkou_b_prev) and (senkou_a_now > senkou_b_now)

        tenkan_prev = ich["tenkan"].iloc[-2]
        kijun_prev = ich["kijun"].iloc[-2]
        tk_cross_up = (tenkan_prev <= kijun_prev) and (ich["tenkan"].iloc[-1] > ich["kijun"].iloc[-1])

        if len(df) > DISPLACEMENT + 1:
            harga_26_lalu_now = df["Close"].iloc[-(DISPLACEMENT + 1)]
            harga_27_lalu_prev = df["Close"].iloc[-(DISPLACEMENT + 2)]
            chikou_cross_up = (prev["Close"] <= harga_27_lalu_prev) and (latest["Close"] > harga_26_lalu_now)
        else:
            chikou_cross_up = False

        ichimoku_ringkas = kumo_cross_up and tk_cross_up and chikou_cross_up
        # Ketiga syarat harus persis bersamaan itu sangat ketat (jarang terjadi
        # bareng di hari yang sama) - kalau ternyata TERLALU sedikit sinyal yang
        # lolos, longgarkan jadi "ketiganya terjadi dalam rentang 3-5 hari
        # terakhir" alih-alih harus PERSIS di hari yang sama.

        # ---------- Cek tambahan: WARNING kalau sudah mentok upper Bollinger Band ----------
        # Temuan dari BAJA: 3 sinyal bullish sekaligus tapi ternyata harga sudah
        # mentok upper BB (overbought) - perlu ditandai eksplisit, bukan disembunyikan.
        jarak_upper_pct = ((latest["Close"] - latest["BB_upper"]) / latest["BB_upper"]) * 100
        sudah_mentok_upper_bb = jarak_upper_pct >= -1.0  # dalam 1% dari upper band atau sudah tembus

        # ---------- KATEGORI 6 (BARU): MACD Histogram Divergence + Filter Rezim ----------
        macd_divergence = False
        if len(df) > DIVERGENCE_LOOKBACK:
            harga_turun_div = latest["Close"] < df["Close"].iloc[-(DIVERGENCE_LOOKBACK + 1)]
            hist_naik_div = latest["MACD_Hist"] > df["MACD_Hist"].iloc[-(DIVERGENCE_LOOKBACK + 1)]
            hist_negatif = latest["MACD_Hist"] < 0
            macd_divergence = harga_turun_div and hist_naik_div and hist_negatif
        macd_divergence_bull_regime = macd_divergence and ihsg_rezim_sekarang

        # ---------- Susun keterangan ----------
        keterangan = []
        if stoch_golden_cross:
            keterangan.append("Stoch golden cross")
        if supertrend_baru_hijau:
            keterangan.append("Supertrend BARU hijau")
        elif supertrend_bullish:
            keterangan.append("Supertrend hijau")
        if di_lower_band:
            keterangan.append(f"Di lower BB ({jarak_lower_pct:.1f}%)")
        if volume_alert:
            keterangan.append(f"VOLUME SPIKE {vol_ratio:.1f}x")
        if chart_naik_signifikan:
            keterangan.append(f"Harga naik {kenaikan_5hari_pct:.0f}% (5 hari)")
        if deteksi_dini:
            keterangan.append(f"🔍 DETEKSI DINI: vol {vol_ratio:.1f}x, harga baru {kenaikan_5hari_pct:+.1f}%")
        if cmf_akumulasi_terkonfirmasi:
            keterangan.append(f"CMF {latest['CMF']:.2f} + harga naik {kenaikan_konfirmasi_pct:.1f}% + vol {vol_ratio:.1f}x")
        if double_golden_cross:
            tipe_ma = "EMA" if GUNAKAN_EMA else "SMA"
            keterangan.append(f"⚡ Stoch+{tipe_ma}({MA_FAST_PERIOD}/{MA_SLOW_PERIOD}) Golden Cross bersamaan")
        if ichimoku_ringkas:
            keterangan.append("☁️ Ichimoku CROSSOVER (Kumo+TK+Chikou bersamaan)")
        if macd_divergence_bull_regime:
            keterangan.append(f"📈 MACD Div+Rezim (harga {DIVERGENCE_LOOKBACK}hr lalu: Rp{df['Close'].iloc[-(DIVERGENCE_LOOKBACK+1)]:.0f} -> sekarang Rp{latest['Close']:.0f})")
        elif macd_divergence:
            keterangan.append("MACD Histogram Divergence (rezim IHSG belum bullish)")
        if sudah_mentok_upper_bb:
            keterangan.append("⚠️ SUDAH MENTOK UPPER BB (overbought, hati-hati kejar)")

        return {
            "Ticker": ticker.replace(".JK", ""),
            "Harga": round(latest["Close"], 0),
            "Skor_Confluence": skor_confluence,
            "Confluence_Kuat": confluence_kuat,
            "Vol_ratio": round(vol_ratio, 2),
            "Kenaikan_5hari_%": round(kenaikan_5hari_pct, 1),
            "Volume_Alert": volume_alert,
            "Chart_Naik_Signifikan": chart_naik_signifikan,
            "Deteksi_Dini": deteksi_dini,
            "CMF": round(latest["CMF"], 3) if pd.notna(latest["CMF"]) else 0,
            "CMF_Akumulasi_Terkonfirmasi": cmf_akumulasi_terkonfirmasi,
            "Double_Golden_Cross": double_golden_cross,
            "Ichimoku_Ringkas": ichimoku_ringkas,
            "MACD_Divergence_Bull_Regime": macd_divergence_bull_regime,
            "Sudah_Mentok_Upper_BB": sudah_mentok_upper_bb,
            "Keterangan": " | ".join(keterangan) if keterangan else "-",
        }

    except Exception:
        return None


# ============================================================
# 6. TELEGRAM & PERSISTENSI
# ============================================================

def kirim_telegram(pesan):
    if TELEGRAM_TOKEN == "ISI_TOKEN_BOT_KAMU_DISINI":
        print("[INFO] Telegram belum dikonfigurasi:\n")
        print(pesan)
        return
    BATAS_KARAKTER = 4000
    potongan = [pesan[i:i + BATAS_KARAKTER] for i in range(0, len(pesan), BATAS_KARAKTER)] or [pesan]
    for i, bagian in enumerate(potongan):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": bagian, "parse_mode": "HTML"}, timeout=15)
            if resp.status_code != 200:
                print(f"[ERROR] Gagal kirim Telegram (bagian {i+1}/{len(potongan)}): {resp.status_code} - {resp.text}")
            else:
                print(f"[OK] Pesan Telegram bagian {i+1}/{len(potongan)} terkirim.")
        except Exception as e:
            print(f"[ERROR] Exception saat kirim Telegram (bagian {i+1}/{len(potongan)}): {e}")


def get_ticker_persisten(kategori_hari_ini, nama_kolom):
    import glob
    file_lama = sorted(glob.glob(os.path.join(FOLDER_HASIL_SCAN, "full_scan_*.csv")))
    if not file_lama:
        return set()
    try:
        df_kemarin = pd.read_csv(file_lama[-1])
        ticker_kemarin = set(df_kemarin[df_kemarin[nama_kolom] == True]["Ticker"]) if nama_kolom in df_kemarin.columns else set()
        return set(kategori_hari_ini) & ticker_kemarin
    except Exception:
        return set()


# ============================================================
# 7. JALANKAN FULL SCAN
# ============================================================

def run_full_scan():
    tickers = get_watchlist()
    print(f"\nMemulai scan {len(tickers)} saham...\n")

    print("Menghitung rezim IHSG dulu (untuk filter MACD Divergence)...")
    _, ihsg_rezim_sekarang = hitung_rezim_ihsg()
    print(f"Rezim IHSG saat ini: {'BULLISH (di atas MA100)' if ihsg_rezim_sekarang else 'belum bullish'}\n")

    hasil = []
    for i, ticker in enumerate(tickers):
        r = analyze_ticker(ticker, ihsg_rezim_sekarang)
        if r:
            hasil.append(r)
        time.sleep(JEDA_ANTAR_REQUEST)
        if (i + 1) % 10 == 0:
            print(f"   ...progress {i+1}/{len(tickers)}")

    if not hasil:
        print("Tidak ada data berhasil diambil.")
        return

    df_hasil = pd.DataFrame(hasil)

    confluence_kuat = df_hasil[df_hasil["Confluence_Kuat"]].sort_values("Skor_Confluence", ascending=False)
    momentum_alert = df_hasil[df_hasil["Volume_Alert"] | df_hasil["Chart_Naik_Signifikan"]]
    deteksi_dini_alert = df_hasil[df_hasil["Deteksi_Dini"]].sort_values("Vol_ratio", ascending=False)
    cmf_alert = df_hasil[df_hasil["CMF_Akumulasi_Terkonfirmasi"]].sort_values("CMF", ascending=False)
    double_golden_alert = df_hasil[df_hasil["Double_Golden_Cross"]]
    ichimoku_alert = df_hasil[df_hasil["Ichimoku_Ringkas"]]
    macd_div_alert = df_hasil[df_hasil["MACD_Divergence_Bull_Regime"]]

    persisten_confluence = get_ticker_persisten(confluence_kuat["Ticker"].tolist(), "Confluence_Kuat")
    persisten_cmf = get_ticker_persisten(cmf_alert["Ticker"].tolist(), "CMF_Akumulasi_Terkonfirmasi")

    print(f"\n{'='*80}")
    print(f"HASIL FULL SCAN - {waktu_wib().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"Total saham dianalisis: {len(df_hasil)}")
    print(f"Confluence Kuat: {len(confluence_kuat)} | Momentum: {len(momentum_alert)} | Deteksi Dini: {len(deteksi_dini_alert)}")
    print(f"CMF Akumulasi Terkonfirmasi: {len(cmf_alert)} | Stoch+MA Golden Cross: {len(double_golden_alert)}")
    print(f"Ichimoku Ringkas: {len(ichimoku_alert)} | MACD Div+Rezim: {len(macd_div_alert)}\n")

    pesan = f"<b>SCAN IHSG v2 - {waktu_wib().strftime('%d %b %Y %H:%M')}</b>\n"
    pesan += f"<i>Rezim IHSG: {'🟢 Bullish' if ihsg_rezim_sekarang else '🔴 Belum bullish'}</i>\n\n"

    if not confluence_kuat.empty:
        pesan += "<b>Confluence Kuat (Stoch+Supertrend+BB):</b>\n"
        for _, row in confluence_kuat.head(10).iterrows():
            tanda = " 🔥2hr" if row["Ticker"] in persisten_confluence else ""
            tanda_ov = " ⚠️OVERBOUGHT" if row["Sudah_Mentok_Upper_BB"] else ""
            pesan += f"• {row['Ticker']}{tanda}{tanda_ov} (Rp{row['Harga']:.0f}) - {row['Keterangan']}\n"
        pesan += "\n"

    if not double_golden_alert.empty:
        tipe_ma = "EMA" if GUNAKAN_EMA else "SMA"
        pesan += f"<b>⚡ Stoch({STOCH_K},{STOCH_SMOOTH},{STOCH_D}) + {tipe_ma}({MA_FAST_PERIOD}/{MA_SLOW_PERIOD}) Golden Cross Bersamaan:</b>\n"
        for _, row in double_golden_alert.head(10).iterrows():
            tanda_ov = " ⚠️OVERBOUGHT" if row["Sudah_Mentok_Upper_BB"] else ""
            pesan += f"• {row['Ticker']}{tanda_ov} (Rp{row['Harga']:.0f})\n"
        pesan += "\n"

    if not ichimoku_alert.empty:
        pesan += "<b>☁️ Ichimoku Crossover (Kumo+Tenkan/Kijun+Chikou memotong BERSAMAAN hari ini):</b>\n"
        for _, row in ichimoku_alert.head(10).iterrows():
            tanda_ov = " ⚠️OVERBOUGHT" if row["Sudah_Mentok_Upper_BB"] else ""
            pesan += f"• {row['Ticker']}{tanda_ov} (Rp{row['Harga']:.0f})\n"
        pesan += "\n"

    if not macd_div_alert.empty:
        pesan += "<b>📈 MACD Histogram Divergence + Rezim Bullish (hasil backtest TERBAIK kita):</b>\n"
        for _, row in macd_div_alert.head(10).iterrows():
            tanda_ov = " ⚠️OVERBOUGHT" if row["Sudah_Mentok_Upper_BB"] else ""
            pesan += f"• {row['Ticker']}{tanda_ov} (Rp{row['Harga']:.0f})\n"
        pesan += "\n"
    elif not ihsg_rezim_sekarang:
        pesan += "<i>📈 MACD Divergence: tidak ada sinyal - rezim IHSG belum bullish (syarat wajib).</i>\n\n"

    if not cmf_alert.empty:
        pesan += "<b>💰 CMF Akumulasi Terkonfirmasi (harga naik + volume naik, bukan cuma teori):</b>\n"
        for _, row in cmf_alert.head(10).iterrows():
            tanda = " 🔥2hr" if row["Ticker"] in persisten_cmf else ""
            tanda_ov = " ⚠️OVERBOUGHT" if row["Sudah_Mentok_Upper_BB"] else ""
            pesan += f"• {row['Ticker']}{tanda}{tanda_ov} (Rp{row['Harga']:.0f}) - CMF {row['CMF']:.2f}\n"
        pesan += "\n"

    if not momentum_alert.empty:
        pesan += "<b>⚠ Volume/Momentum Alert:</b>\n"
        for _, row in momentum_alert.head(10).iterrows():
            pesan += f"• {row['Ticker']} (Rp{row['Harga']:.0f}) - Vol {row['Vol_ratio']:.1f}x, naik {row['Kenaikan_5hari_%']:.0f}% (5hr)\n"
        pesan += "\n"

    if not deteksi_dini_alert.empty:
        pesan += "<b>🔍 Deteksi Dini:</b>\n"
        for _, row in deteksi_dini_alert.head(10).iterrows():
            pesan += f"• {row['Ticker']} (Rp{row['Harga']:.0f}) - Vol {row['Vol_ratio']:.1f}x, baru {row['Kenaikan_5hari_%']:+.1f}% (5hr)\n"
        pesan += "\n"

    pesan += ("<i>PENGINGAT: dari ~20 backtest kita, TIDAK ADA kategori di atas yang terbukti "
              "punya edge kuat setelah fee - yang terbaik (MACD Div+Rezim) cuma breakeven-ish. "
              "Gunakan size kecil, WAJIB stop-loss disiplin, dan anggap ini latihan psikologi/"
              "money-management, bukan sinyal pasti profit.</i>")

    kirim_telegram(pesan)

    filename = f"full_scan_{waktu_wib().strftime('%Y%m%d_%H%M')}.csv"
    df_hasil.to_csv(filename, index=False)
    print(f"\nHasil lengkap disimpan ke: {filename}")


if __name__ == "__main__":
    run_full_scan()
