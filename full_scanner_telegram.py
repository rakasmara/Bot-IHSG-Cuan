"""
FULL MARKET SCANNER IHSG + TELEGRAM ALERT
============================================================================
Scan SEMUA saham IDX (bukan cuma watchlist manual) untuk kombinasi:
- Stochastic(6,3,3) golden cross
- Supertrend(10,1) hijau
- Bollinger Bands - harga di lower band
- Volume spike (indikasi mau breakout)
- CMF & A/D Line (tekanan beli/jual berbobot posisi closing - lebih akurat dari OBV)
- ARA Detector (saham yang closing di/dekat batas Auto Reject Atas)

Lalu kirim hasil confluence tinggi ke Telegram otomatis.

SETUP AWAL (WAJIB, lakukan sekali):
============================================================================
1. Dapatkan daftar lengkap kode saham IDX:
   - Buka https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham/
   - Download file Excel/CSV daftar saham
   - ATAU pakai sumber lain seperti sahamidx.com yang punya daftar kode saham
   - Simpan sebagai "daftar_saham_idx.csv" dengan minimal 1 kolom bernama "Kode"
   - Upload file itu ke Colab (klik ikon folder di sidebar kiri > upload)

2. Setup Telegram Bot (gratis, 5 menit):
   a. Di Telegram, chat ke @BotFather
   b. Ketik /newbot, ikuti instruksi, kamu akan dapat TOKEN (contoh: 123456:ABC-DEF...)
   c. Chat bot kamu sekali (ketik apa saja) supaya bot bisa balas ke kamu
   d. Buka di browser: https://api.telegram.org/bot<TOKEN>/getUpdates
      (ganti <TOKEN> dengan token kamu)
   e. Cari angka "chat":{"id": XXXXXXX  <- ini CHAT_ID kamu
   f. Isi TOKEN dan CHAT_ID di bagian konfigurasi di bawah
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

warnings.filterwarnings("ignore")  # supaya FutureWarning yfinance tidak memenuhi layar


def waktu_wib():
    """Waktu WIB (UTC+7), dihitung manual dari UTC - tidak bergantung pada
    database timezone sistem (lebih aman untuk server seperti GitHub Actions)"""
    return datetime.now(timezone.utc) + timedelta(hours=7)


# ============================================================
# 1. KONFIGURASI - WAJIB DIISI
# ============================================================

# Token & chat_id diambil dari environment variable (GitHub Secrets) kalau ada,
# kalau tidak ada (misal saat test manual di Colab), pakai nilai default di bawah.
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "ISI_TOKEN_BOT_KAMU_DISINI")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "ISI_CHAT_ID_KAMU_DISINI")

# --- FILTER LIKUIDITAS ---
# Menyaring saham "gocap"/tidak likuid yang bisa memicu sinyal palsu
MIN_HARGA = 50               # abaikan saham di bawah harga ini
MIN_VOLUME_HARIAN = 100000   # abaikan saham dengan rata-rata volume < ini (lembar/hari)

# Filter tambahan berbasis NILAI transaksi (Rupiah), bukan cuma jumlah lembar.
# Ini penting karena saham murah dengan volume lembar besar tapi nilai Rupiah
# kecil paling rawan "dipoles" candle-nya (closing dipertahankan tinggi oleh
# pemain besar sambil pelan-pelan distribusi) - persis kasus false positive
# akumulasi yang perlu disaring dari kategori CMF/Akumulasi.
MIN_NILAI_TRANSAKSI_HARIAN = 2_000_000_000  # rata-rata 20 hari, dalam Rupiah

DAFTAR_SAHAM_FILE = "Daftar_Saham_Idx.csv"  # file CSV yang kamu upload
DAFTAR_SAHAM_KOLOM = "Kode"                 # nama kolom yang berisi kode saham

# Kalau belum punya file lengkap, bisa pakai daftar manual dulu (contoh saham likuid + grup Haji Isam)
GUNAKAN_DAFTAR_MANUAL_DULU = False  # set False supaya scan pakai daftar dari file CSV
DAFTAR_MANUAL = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP", "INDF",
    "ANTM", "MDKA", "INKP", "TPIA", "BRPT", "CUAN", "AMMN", "ADRO", "PTBA",
    "TEBE", "JARR", "PGUN", "DEWA", "ELPI", "BNBR", "RAJA", "SHIP", "GEMS",
    "MBMA", "NCKL", "PANI", "BREN", "BRMS", "TINS", "MEDC", "PGAS", "AKRA",
]

STOCH_K, STOCH_SMOOTH, STOCH_D = 6, 3, 3
STOCH_OVERSOLD = 20

ST_ATR_PERIOD, ST_MULTIPLIER = 10, 1

BB_PERIOD, BB_STD = 20, 2
BB_LOWER_THRESHOLD_PCT = 1.0

VOL_SPIKE_MULTIPLIER = 3.0
VOL_LOOKBACK = 20

# --- DETEKSI DINI ---
# Menangkap saham yang volume-nya melonjak TAPI harga belum bergerak jauh -
# indikasi akumulasi awal, sebelum breakout harga terjadi (bukan setelah telat)
EARLY_VOL_MULTIPLIER = 2.5     # volume naik minimal 2.5x rata-rata
EARLY_MAX_KENAIKAN_PCT = 8     # tapi harga masih naik di bawah 8% (belum "telat")
EARLY_MIN_KENAIKAN_PCT = -5    # dan tidak sedang jatuh tajam (turun lebih dari 5%)

# --- AKUMULASI DIAM-DIAM via OBV ---
# Deteksi OBV naik signifikan sementara harga masih relatif flat -
# proxy gratis untuk "ada yang mengumpulkan barang" tanpa perlu data broker
OBV_LOOKBACK = 15             # bandingkan OBV hari ini vs N hari lalu
OBV_MAX_HARGA_FLAT_PCT = 5    # harga dianggap "masih flat" kalau perubahan < ini

# --- CMF & A/D LINE (baru) ---
# Versi lebih akurat dari OBV: mempertimbangkan POSISI closing di dalam
# range High-Low hari itu, bukan cuma arah close vs close kemarin.
CMF_PERIOD = 20               # jumlah hari untuk hitung rata-rata CMF
CMF_THRESHOLD = 0.1           # di atas ini dianggap tekanan beli bersih kuat
AD_LOOKBACK = 15              # sama seperti OBV_LOOKBACK, untuk divergence
AD_MAX_HARGA_FLAT_PCT = 5

# --- ARA DETECTOR (baru) ---
# Mendeteksi saham yang closing KEMARIN di/dekat batas Auto Reject Atas.
# Sifatnya retrospektif (data harian), bukan real-time intraday.
TOLERANSI_ARA_PCT = 1.5          # dianggap "kena ARA" kalau >= (batas - toleransi)
AMBANG_MENDEKATI_ARA_PCT = 5.0   # dianggap "mendekati ARA" kalau masih segini % di bawah batas

MIN_SKOR_ALERT = 3   # perketat: wajib SEMUA indikator (Stoch+Supertrend+BB) sejalan, bukan cuma 2 dari 3

# --- PERSISTENSI SINYAL ---
# Sinyal yang muncul 2 hari scan berturut-turut jauh lebih bisa dipercaya
# daripada yang muncul sekali lalu hilang. File CSV hasil scan kemarin
# dipakai sebagai pembanding otomatis.
FOLDER_HASIL_SCAN = "."  # folder tempat file full_scan_*.csv disimpan
LOOKBACK_DAYS = "6mo"
JEDA_ANTAR_REQUEST = 0.3   # detik, supaya tidak kena rate-limit yfinance


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


def calculate_obv(df):
    """On-Balance Volume - proxy gratis untuk deteksi akumulasi/distribusi
    tanpa perlu data broker. Naik terus = ada tekanan beli kumulatif."""
    arah = np.sign(df["Close"].diff()).fillna(0)
    obv = (arah * df["Volume"]).cumsum()
    return obv


def calculate_money_flow_multiplier(df):
    """Posisi closing price di dalam range High-Low.
    +1 = closing di High (tekanan beli maksimal)
    -1 = closing di Low (tekanan jual maksimal)
     0 = closing di tengah (netral)"""
    high, low, close = df["High"], df["Low"], df["Close"]
    range_hl = (high - low).replace(0, np.nan)  # hindari divide-by-zero saat High==Low
    mfm = ((close - low) - (high - close)) / range_hl
    return mfm.fillna(0)


def calculate_cmf(df, period=20):
    """Chaikin Money Flow - rata-rata tekanan beli/jual selama N hari,
    dibobotkan volume. Range -1 sampai +1. Lebih akurat dari OBV karena
    mempertimbangkan posisi closing di dalam range harian, bukan cuma
    arah close vs close kemarin."""
    mfm = calculate_money_flow_multiplier(df)
    mfv = mfm * df["Volume"]
    cmf = mfv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()
    return cmf


def calculate_ad_line(df):
    """Accumulation/Distribution Line - versi OBV yang dibobotkan posisi
    closing. Kumulatif seperti OBV, dibaca dari TRENnya (naik terus vs
    mendatar/turun), bukan angka absolutnya."""
    mfm = calculate_money_flow_multiplier(df)
    mfv = mfm * df["Volume"]
    return mfv.cumsum()


def get_batas_ara(harga_acuan):
    """Kembalikan persentase batas ARA sesuai tingkatan harga acuan
    (harga penutupan hari sebelumnya)."""
    if harga_acuan < 200:
        return 35.0
    elif harga_acuan <= 5000:
        return 25.0
    else:
        return 20.0


def cek_status_ara(harga_acuan, harga_close_hari_ini):
    """Bandingkan closing hari ini terhadap harga acuan (closing kemarin)
    dan tentukan status ARA-nya.

    Catatan keterbatasan: saham baru IPO beberapa hari pertama punya aturan
    ARA berbeda dan tidak ditangani khusus di sini - kalau bot menandai
    saham yang belum lama listing sebagai ARA/mendekati ARA secara keliru,
    itu sumbernya. Filter len(df) minimal di analyze_ticker membantu
    mengurangi ini."""
    if harga_acuan <= 0:
        return {"persen_kenaikan": 0, "batas_ara": 0, "kena_ara": False, "mendekati_ara": False}

    persen_kenaikan = ((harga_close_hari_ini - harga_acuan) / harga_acuan) * 100
    batas = get_batas_ara(harga_acuan)

    kena_ara = persen_kenaikan >= (batas - TOLERANSI_ARA_PCT)
    mendekati_ara = (not kena_ara) and (persen_kenaikan >= (batas - AMBANG_MENDEKATI_ARA_PCT))

    return {
        "persen_kenaikan": round(persen_kenaikan, 2),
        "batas_ara": batas,
        "kena_ara": kena_ara,
        "mendekati_ara": mendekati_ara,
    }


# ============================================================
# 3. AMBIL DAFTAR SAHAM
# ============================================================

def get_watchlist():
    if GUNAKAN_DAFTAR_MANUAL_DULU:
        print(f"Menggunakan daftar manual: {len(DAFTAR_MANUAL)} saham")
        return [f"{kode}.JK" for kode in DAFTAR_MANUAL]
    else:
        try:
            # File CSV yang diupload dibaca langsung
            df_saham = pd.read_csv(DAFTAR_SAHAM_FILE)
            kode_list = df_saham[DAFTAR_SAHAM_KOLOM].astype(str).str.strip().tolist()
            kode_list = [k for k in kode_list if k and k.lower() != "nan"]
            print(f"Berhasil load {len(kode_list)} saham dari {DAFTAR_SAHAM_FILE}")
            return [f"{kode}.JK" for kode in kode_list]
        except Exception as e:
            print(f"Gagal load file ({e}), fallback ke daftar manual")
            return [f"{kode}.JK" for kode in DAFTAR_MANUAL]


# ============================================================
# 4. ANALISIS PER SAHAM
# ============================================================

def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period=LOOKBACK_DAYS, progress=False)
        if df.empty or len(df) < 60:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Stoch_K"], df["Stoch_D"] = calculate_stochastic(df, STOCH_K, STOCH_SMOOTH, STOCH_D)
        df["Supertrend"], df["ST_Direction"] = calculate_supertrend(df, ST_ATR_PERIOD, ST_MULTIPLIER)
        df["BB_upper"], df["BB_mid"], df["BB_lower"] = calculate_bollinger(df, BB_PERIOD, BB_STD)
        df["Vol_avg"] = df["Volume"].rolling(window=VOL_LOOKBACK).mean()
        df["OBV"] = calculate_obv(df)
        df["CMF"] = calculate_cmf(df, CMF_PERIOD)
        df["AD"] = calculate_ad_line(df)

        latest, prev = df.iloc[-1], df.iloc[-2]

        # --- Filter likuiditas: skip saham gocap/tidak likuid ---
        avg_vol_20 = df["Volume"].tail(20).mean()
        if latest["Close"] < MIN_HARGA or avg_vol_20 < MIN_VOLUME_HARIAN:
            return None

        # --- Filter nilai transaksi (Rupiah) - saring saham "gampang dipoles" ---
        nilai_transaksi_20 = (df["Close"].tail(20) * df["Volume"].tail(20)).mean()
        likuid_untuk_akumulasi = nilai_transaksi_20 >= MIN_NILAI_TRANSAKSI_HARIAN

        stoch_golden_cross = (prev["Stoch_K"] <= prev["Stoch_D"]) and (latest["Stoch_K"] > latest["Stoch_D"])
        supertrend_bullish = latest["ST_Direction"] == 1
        supertrend_baru_hijau = (prev["ST_Direction"] == -1) and (latest["ST_Direction"] == 1)

        jarak_lower_pct = ((latest["Close"] - latest["BB_lower"]) / latest["BB_lower"]) * 100
        di_lower_band = jarak_lower_pct <= BB_LOWER_THRESHOLD_PCT

        vol_ratio = latest["Volume"] / latest["Vol_avg"] if latest["Vol_avg"] > 0 else 0
        volume_alert = vol_ratio >= VOL_SPIKE_MULTIPLIER

        # Deteksi "chart naik signifikan" - perubahan harga 5 hari terakhir
        harga_5hari_lalu = df["Close"].iloc[-6] if len(df) > 6 else df["Close"].iloc[0]
        kenaikan_5hari_pct = ((latest["Close"] - harga_5hari_lalu) / harga_5hari_lalu) * 100
        chart_naik_signifikan = kenaikan_5hari_pct >= 15  # naik >=15% dalam 5 hari dianggap signifikan

        # --- Deteksi Dini: volume melonjak TAPI harga belum bergerak jauh ---
        deteksi_dini = (
            vol_ratio >= EARLY_VOL_MULTIPLIER
            and EARLY_MIN_KENAIKAN_PCT <= kenaikan_5hari_pct <= EARLY_MAX_KENAIKAN_PCT
        )

        # --- Akumulasi Diam-diam: OBV naik signifikan, harga masih flat ---
        akumulasi_obv = False
        if len(df) > OBV_LOOKBACK:
            obv_now = latest["OBV"]
            obv_dulu = df["OBV"].iloc[-(OBV_LOOKBACK + 1)]
            harga_dulu = df["Close"].iloc[-(OBV_LOOKBACK + 1)]
            harga_flat_pct = ((latest["Close"] - harga_dulu) / harga_dulu) * 100
            obv_naik = obv_now > obv_dulu
            harga_masih_flat = abs(harga_flat_pct) <= OBV_MAX_HARGA_FLAT_PCT
            akumulasi_obv = obv_naik and harga_masih_flat
        else:
            harga_flat_pct = 0

        # --- CMF Bullish: tekanan beli bersih kuat dalam N hari terakhir ---
        cmf_bullish = latest["CMF"] > CMF_THRESHOLD

        # --- A/D Divergence: mirip akumulasi_obv, tapi pakai A/D Line yang
        # lebih akurat karena mempertimbangkan posisi closing, bukan cuma arah ---
        ad_divergence = False
        harga_flat_pct_ad = 0
        if len(df) > AD_LOOKBACK:
            ad_now = latest["AD"]
            ad_dulu = df["AD"].iloc[-(AD_LOOKBACK + 1)]
            harga_dulu_ad = df["Close"].iloc[-(AD_LOOKBACK + 1)]
            harga_flat_pct_ad = ((latest["Close"] - harga_dulu_ad) / harga_dulu_ad) * 100
            ad_naik = ad_now > ad_dulu
            harga_masih_flat_ad = abs(harga_flat_pct_ad) <= AD_MAX_HARGA_FLAT_PCT
            ad_divergence = ad_naik and harga_masih_flat_ad

        # Sinyal paling kuat: OBV DAN A/D Line dua-duanya konfirmasi akumulasi.
        akumulasi_terkonfirmasi = akumulasi_obv and ad_divergence

        # Gerbang likuiditas: sinyal akumulasi/CMF HANYA valid kalau nilai
        # transaksinya cukup besar. Saham tipis terlalu mudah "dipoles"
        # closing-nya oleh modal kecil, menghasilkan CMF/OBV tinggi palsu
        # yang sebenarnya distribusi (lihat kasus AMAG).
        if not likuid_untuk_akumulasi:
            cmf_bullish = False
            akumulasi_obv = False
            ad_divergence = False
            akumulasi_terkonfirmasi = False

        # --- ARA Detector: closing hari ini vs closing kemarin ---
        status_ara = cek_status_ara(prev["Close"], latest["Close"])

        skor = sum([stoch_golden_cross, supertrend_bullish, di_lower_band])

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
        if akumulasi_obv:
            keterangan.append(f"🤫 AKUMULASI OBV: naik {OBV_LOOKBACK}hr, harga flat {harga_flat_pct:+.1f}%")
        if cmf_bullish:
            keterangan.append(f"CMF {latest['CMF']:.2f} (tekanan beli kuat)")
        if akumulasi_terkonfirmasi:
            keterangan.append("✅ AKUMULASI TERKONFIRMASI (OBV + A/D Line sejalan)")
        elif ad_divergence:
            keterangan.append(f"A/D Divergence: naik {AD_LOOKBACK}hr, harga flat {harga_flat_pct_ad:+.1f}%")
        if status_ara["kena_ara"]:
            keterangan.append(f"🚀 ARA! naik {status_ara['persen_kenaikan']}% (batas {status_ara['batas_ara']}%)")
        elif status_ara["mendekati_ara"]:
            keterangan.append(f"Mendekati ARA: naik {status_ara['persen_kenaikan']}% (batas {status_ara['batas_ara']}%)")

        return {
            "Ticker": ticker.replace(".JK", ""),
            "Harga": round(latest["Close"], 0),
            "Skor": skor,
            "Nilai_Transaksi_20hr": round(nilai_transaksi_20, 0),
            "Likuid_Akumulasi": likuid_untuk_akumulasi,
            "Vol_ratio": round(vol_ratio, 2),
            "Kenaikan_5hari_%": round(kenaikan_5hari_pct, 1),
            "Volume_Alert": volume_alert,
            "Chart_Naik_Signifikan": chart_naik_signifikan,
            "Deteksi_Dini": deteksi_dini,
            "Akumulasi_OBV": akumulasi_obv,
            "CMF": round(latest["CMF"], 3) if pd.notna(latest["CMF"]) else 0,
            "CMF_Bullish": cmf_bullish,
            "AD_Divergence": ad_divergence,
            "Akumulasi_Terkonfirmasi": akumulasi_terkonfirmasi,
            "Persen_Kenaikan": status_ara["persen_kenaikan"],
            "Kena_ARA": status_ara["kena_ara"],
            "Mendekati_ARA": status_ara["mendekati_ara"],
            "Keterangan": " | ".join(keterangan) if keterangan else "-",
        }

    except Exception:
        return None


# ============================================================
# 5. KIRIM ALERT KE TELEGRAM
# ============================================================

def kirim_telegram(pesan):
    if TELEGRAM_TOKEN == "ISI_TOKEN_BOT_KAMU_DISINI":
        print("[INFO] Telegram belum dikonfigurasi, alert hanya ditampilkan di sini:\n")
        print(pesan)
        return

    # Telegram membatasi maksimal 4096 karakter per pesan - potong jadi
    # beberapa bagian kalau kepanjangan, supaya tidak gagal kirim total
    BATAS_KARAKTER = 4000  # kasih sedikit margin dari batas resmi 4096
    potongan = [pesan[i:i + BATAS_KARAKTER] for i in range(0, len(pesan), BATAS_KARAKTER)] or [pesan]

    for i, bagian in enumerate(potongan):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": bagian, "parse_mode": "HTML"}, timeout=15)
            if resp.status_code != 200:
                print(f"[ERROR] Gagal kirim Telegram (bagian {i+1}/{len(potongan)}): "
                      f"status {resp.status_code} - {resp.text}")
            else:
                print(f"[OK] Pesan Telegram bagian {i+1}/{len(potongan)} terkirim.")
        except Exception as e:
            print(f"[ERROR] Exception saat kirim Telegram (bagian {i+1}/{len(potongan)}): {e}")


def get_ticker_persisten(kategori_hari_ini, nama_kategori):
    """Bandingkan daftar ticker kategori tertentu hari ini dengan hasil
    scan TERAKHIR yang tersimpan (file full_scan_*.csv sebelumnya).
    Return set ticker yang muncul di KEDUA scan (dianggap lebih meyakinkan
    karena tidak cuma noise sehari)."""
    import glob

    file_lama = sorted(glob.glob(os.path.join(FOLDER_HASIL_SCAN, "full_scan_*.csv")))
    if not file_lama:
        return set()  # belum ada riwayat scan sebelumnya

    try:
        df_kemarin = pd.read_csv(file_lama[-1])  # scan terakhir yang tersimpan
        if nama_kategori == "confluence":
            ticker_kemarin = set(df_kemarin[df_kemarin["Skor"] >= MIN_SKOR_ALERT]["Ticker"])
        elif nama_kategori == "akumulasi_terkonfirmasi":
            ticker_kemarin = set(df_kemarin[df_kemarin["Akumulasi_Terkonfirmasi"]]["Ticker"])
        else:
            return set()
        return set(kategori_hari_ini) & ticker_kemarin
    except Exception:
        return set()


# ============================================================
# 6. JALANKAN FULL SCAN
# ============================================================

def run_full_scan():
    tickers = get_watchlist()
    print(f"\nMemulai scan {len(tickers)} saham... (bisa beberapa menit)\n")

    hasil = []
    for i, ticker in enumerate(tickers):
        r = analyze_ticker(ticker)
        if r:
            hasil.append(r)
        time.sleep(JEDA_ANTAR_REQUEST)
        if (i + 1) % 10 == 0:
            print(f"   ...progress {i+1}/{len(tickers)}")

    if not hasil:
        print("Tidak ada data berhasil diambil.")
        return

    df_hasil = pd.DataFrame(hasil)

    # Saham dengan confluence tinggi (2-3 indikator sejalan)
    confluence_kuat = df_hasil[df_hasil["Skor"] >= MIN_SKOR_ALERT].sort_values("Skor", ascending=False)

    # Saham dengan volume alert ATAU chart naik signifikan (terlepas dari skor confluence)
    momentum_alert = df_hasil[df_hasil["Volume_Alert"] | df_hasil["Chart_Naik_Signifikan"]]

    # Saham Deteksi Dini: volume melonjak, harga BELUM bergerak jauh (early stage)
    deteksi_dini_alert = df_hasil[df_hasil["Deteksi_Dini"]].sort_values("Vol_ratio", ascending=False)

    # Saham Akumulasi Terkonfirmasi: OBV + A/D Line sejalan (paling kuat)
    akumulasi_kuat_alert = df_hasil[df_hasil["Akumulasi_Terkonfirmasi"]]

    # Saham Akumulasi OBV saja (fallback, masih berguna kalau A/D belum sejalan)
    akumulasi_alert = df_hasil[df_hasil["Akumulasi_OBV"] & ~df_hasil["Akumulasi_Terkonfirmasi"]]

    # Saham ARA kemarin & yang mendekati ARA
    ara_alert = df_hasil[df_hasil["Kena_ARA"]].sort_values("Persen_Kenaikan", ascending=False)
    mendekati_ara_alert = df_hasil[df_hasil["Mendekati_ARA"]].sort_values("Persen_Kenaikan", ascending=False)

    # --- Cek persistensi vs scan sebelumnya (harus dipanggil SEBELUM file CSV baru disimpan) ---
    persisten_confluence = get_ticker_persisten(confluence_kuat["Ticker"].tolist(), "confluence")
    persisten_akumulasi = get_ticker_persisten(akumulasi_kuat_alert["Ticker"].tolist(), "akumulasi_terkonfirmasi")

    print(f"\n{'='*80}")
    print(f"HASIL FULL SCAN - {waktu_wib().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"Total saham dianalisis: {len(df_hasil)}")
    print(f"Saham confluence >= {MIN_SKOR_ALERT}: {len(confluence_kuat)}")
    print(f"Saham momentum/volume alert: {len(momentum_alert)}")
    print(f"Saham deteksi dini: {len(deteksi_dini_alert)}")
    print(f"Saham akumulasi terkonfirmasi (OBV+A/D): {len(akumulasi_kuat_alert)}")
    print(f"Saham akumulasi OBV saja: {len(akumulasi_alert)}")
    print(f"Saham ARA kemarin: {len(ara_alert)}")
    print(f"Saham mendekati ARA: {len(mendekati_ara_alert)}\n")

    if not confluence_kuat.empty:
        print(confluence_kuat[["Ticker", "Harga", "Skor", "Keterangan"]].to_string(index=False))

    # ---- Susun pesan Telegram ----
    pesan = f"<b>SCAN IHSG - {waktu_wib().strftime('%d %b %Y %H:%M')}</b>\n\n"

    if not confluence_kuat.empty:
        pesan += "<b>Confluence Kuat (Stoch+Supertrend+BB semua sejalan):</b>\n"
        for _, row in confluence_kuat.head(10).iterrows():
            tanda = " 🔥2hr" if row["Ticker"] in persisten_confluence else ""
            pesan += f"• {row['Ticker']}{tanda} (Rp{row['Harga']:.0f}) - {row['Keterangan']}\n"
        pesan += "\n"

    if not momentum_alert.empty:
        pesan += "<b>⚠ Volume/Momentum Alert (sudah bergerak):</b>\n"
        for _, row in momentum_alert.head(10).iterrows():
            pesan += f"• {row['Ticker']} (Rp{row['Harga']:.0f}) - Vol {row['Vol_ratio']:.1f}x, naik {row['Kenaikan_5hari_%']:.0f}% (5hr)\n"
        pesan += "\n"

    if not deteksi_dini_alert.empty:
        pesan += "<b>🔍 Deteksi Dini (volume naik, harga BELUM bergerak jauh):</b>\n"
        for _, row in deteksi_dini_alert.head(10).iterrows():
            pesan += f"• {row['Ticker']} (Rp{row['Harga']:.0f}) - Vol {row['Vol_ratio']:.1f}x, baru {row['Kenaikan_5hari_%']:+.1f}% (5hr)\n"
        pesan += "\n"

    if not akumulasi_kuat_alert.empty:
        pesan += "<b>✅ Akumulasi Terkonfirmasi (sudah difilter nilai transaksi ≥2M/hari):</b>\n"
        for _, row in akumulasi_kuat_alert.head(10).iterrows():
            tanda = " 🔥2hr" if row["Ticker"] in persisten_akumulasi else ""
            pesan += f"• {row['Ticker']}{tanda} (Rp{row['Harga']:.0f}) - CMF {row['CMF']:.2f}\n"
        pesan += "\n<i>🔥2hr = muncul juga di scan sebelumnya, lebih meyakinkan. Tetap bukan jaminan - CMF/OBV tidak bisa membedakan akumulasi asli vs distribusi absorptif (lihat catatan bawah).</i>\n\n"

    if not akumulasi_alert.empty:
        pesan += (f"<b>🤫 Akumulasi OBV (belum lolos filter nilai transaksi/A-D, harga flat {OBV_LOOKBACK} hari):</b>\n"
                  f"{len(akumulasi_alert)} saham - kategori paling spekulatif, cek manual dulu sebelum ikuti.\n\n")

    if not ara_alert.empty:
        pesan += "<b>🚀 ARA Kemarin (watchlist lanjutan):</b>\n"
        for _, row in ara_alert.head(10).iterrows():
            pesan += f"• {row['Ticker']}: +{row['Persen_Kenaikan']}% (Rp{row['Harga']:.0f})\n"
        pesan += "\n"

    if not mendekati_ara_alert.empty:
        pesan += "<b>Mendekati ARA:</b>\n"
        for _, row in mendekati_ara_alert.head(10).iterrows():
            pesan += f"• {row['Ticker']}: +{row['Persen_Kenaikan']}% (Rp{row['Harga']:.0f})\n"
        pesan += "\n"

    ada_sinyal = not (confluence_kuat.empty and momentum_alert.empty and deteksi_dini_alert.empty
                      and akumulasi_kuat_alert.empty and akumulasi_alert.empty
                      and ara_alert.empty and mendekati_ara_alert.empty)

    if ada_sinyal:
        pesan += ("<i>Ingat: volume spike/ARA bisa lanjut naik ATAU jadi ajang distribusi. "
                  "CMF/OBV membaca bentuk candle, BUKAN data broker asli - closing tinggi karena "
                  "serapan agresif ritel bisa terlihat identik dengan akumulasi asli padahal itu "
                  "distribusi (cek Trade Flow/Smart Money di sekuritas kamu untuk saham yang menarik "
                  "sebelum eksekusi). Selalu cek berita & pakai cut-loss.</i>")
    else:
        pesan += "Tidak ada sinyal signifikan hari ini."

    kirim_telegram(pesan)

    filename = f"full_scan_{waktu_wib().strftime('%Y%m%d_%H%M')}.csv"
    df_hasil.to_csv(filename, index=False)
    print(f"\nHasil lengkap disimpan ke: {filename}")


if __name__ == "__main__":
    run_full_scan()
