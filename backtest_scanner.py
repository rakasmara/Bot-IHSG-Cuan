"""
BACKTEST HISTORIS - Bot-IHSG-Cuan
============================================================================
Menguji 3 kategori sinyal (Confluence Kuat, Akumulasi Terkonfirmasi, ARA
Kemarin) terhadap data historis, supaya kamu punya gambaran statistik
SEBELUM forward-test pakai uang beneran di IPOT.

Cara kerja:
1. Untuk setiap saham, hitung SEMUA indikator di seluruh riwayat harga
   (bukan cuma hari terakhir seperti di full_scanner_telegram.py)
2. Tandai SETIAP hari di masa lalu di mana sinyal pernah muncul
3. Simulasikan: kalau kamu beli di harga OPEN keesokan harinya (karena
   sinyal baru kamu tahu setelah market tutup) dan jual di CLOSE setelah
   N hari, berapa return-nya?
4. Kumpulkan semua kejadian itu, hitung win rate & rata-rata return

KETERBATASAN yang perlu disadari (baca sebelum percaya hasilnya):
- Ini backtest SEDERHANA: tidak memperhitungkan biaya transaksi (fee
  beli/jual sekitar 0.15-0.3%), pajak, atau slippage (harga aktual saat
  eksekusi vs harga yang tercatat).
- Tidak ada position sizing/money management - setiap sinyal dianggap
  "all-in" satu posisi terpisah, padahal di real trading kamu punya modal
  terbatas dan harus pilih mana yang dieksekusi.
- Survivorship bias: kalau daftar saham yang dipakai adalah daftar SAAT
  INI, saham yang sudah delisting/suspend di masa lalu tidak ikut
  terhitung - bisa bikin hasil terlihat lebih bagus dari kenyataan.
- Rules ini juga tidak divalidasi ke periode SEBELUM window backtest -
  ada risiko "overfitting" kalau nanti thresholdnya diutak-atik supaya
  cocok dengan hasil backtest ini. Anggap ini pemeriksaan kewarasan awal,
  bukan jaminan performa ke depan.

CARA PAKAI:
1. Jalankan di Google Colab (sama seperti bot utama) atau di komputer lokal
2. Upload Daftar_Saham_Idx.csv (sama seperti untuk full_scanner_telegram.py)
   - kalau tidak ada, otomatis pakai daftar manual yang lebih pendek
3. Jalankan: python backtest_scanner.py
4. Tunggu (bisa 10-30 menit tergantung jumlah saham & koneksi)
5. Baca ringkasan di layar, dan buka file CSV yang dihasilkan untuk detail
   per transaksi
============================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


# ============================================================
# 1. KONFIGURASI
# ============================================================

DAFTAR_SAHAM_FILE = "Daftar_Saham_Idx.csv"
DAFTAR_SAHAM_KOLOM = "Kode"
GUNAKAN_DAFTAR_MANUAL_DULU = False

DAFTAR_MANUAL = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNVR", "ICBP", "INDF",
    "ANTM", "MDKA", "INKP", "TPIA", "BRPT", "CUAN", "AMMN", "ADRO", "PTBA",
    "GEMS", "MBMA", "NCKL", "PANI", "BREN", "BRMS", "TINS", "MEDC", "PGAS", "AKRA",
]

PERIODE_BACKTEST = "2y"     # rentang data historis yang diuji
HORIZON_HARI = [5, 10]      # simulasikan hold 5 hari dan 10 hari (sejalan konsep BSJP vs swing)

# --- Parameter indikator (SAMA PERSIS dengan full_scanner_telegram.py) ---
STOCH_K, STOCH_SMOOTH, STOCH_D = 6, 3, 3
ST_ATR_PERIOD, ST_MULTIPLIER = 10, 1
BB_PERIOD, BB_STD = 20, 2
BB_LOWER_THRESHOLD_PCT = 1.0
MIN_SKOR_ALERT = 3

OBV_LOOKBACK = 15
OBV_MAX_HARGA_FLAT_PCT = 5
CMF_PERIOD = 20
CMF_THRESHOLD = 0.1
AD_LOOKBACK = 15
AD_MAX_HARGA_FLAT_PCT = 5

TOLERANSI_ARA_PCT = 1.5
AMBANG_MENDEKATI_ARA_PCT = 5.0

MIN_HARGA = 50
MIN_VOLUME_HARIAN = 100000
MIN_NILAI_TRANSAKSI_HARIAN = 2_000_000_000

JEDA_ANTAR_REQUEST = 0.3


# ============================================================
# 2. FUNGSI INDIKATOR (sama persis dengan full_scanner_telegram.py)
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
    arah = np.sign(df["Close"].diff()).fillna(0)
    return (arah * df["Volume"]).cumsum()


def calculate_money_flow_multiplier(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    range_hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / range_hl
    return mfm.fillna(0)


def calculate_cmf(df, period=20):
    mfm = calculate_money_flow_multiplier(df)
    mfv = mfm * df["Volume"]
    return mfv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()


def calculate_ad_line(df):
    mfm = calculate_money_flow_multiplier(df)
    mfv = mfm * df["Volume"]
    return mfv.cumsum()


def get_batas_ara(harga_acuan):
    if harga_acuan < 200:
        return 35.0
    elif harga_acuan <= 5000:
        return 25.0
    else:
        return 20.0


# ============================================================
# 3. HITUNG SEMUA SINYAL SECARA VEKTOR (untuk seluruh riwayat sekaligus)
# ============================================================

def hitung_semua_sinyal(df):
    """Hitung boolean Series untuk setiap kategori sinyal, di SETIAP hari
    dalam riwayat df (bukan cuma hari terakhir)."""
    df = df.copy()

    df["Stoch_K"], df["Stoch_D"] = calculate_stochastic(df, STOCH_K, STOCH_SMOOTH, STOCH_D)
    df["Supertrend"], df["ST_Direction"] = calculate_supertrend(df, ST_ATR_PERIOD, ST_MULTIPLIER)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = calculate_bollinger(df, BB_PERIOD, BB_STD)
    df["OBV"] = calculate_obv(df)
    df["CMF"] = calculate_cmf(df, CMF_PERIOD)
    df["AD"] = calculate_ad_line(df)

    # --- Filter likuiditas (rolling, sama seperti versi live) ---
    nilai_transaksi_20 = (df["Close"] * df["Volume"]).rolling(20).mean()
    vol_avg_20 = df["Volume"].rolling(20).mean()
    likuid = (df["Close"] >= MIN_HARGA) & (vol_avg_20 >= MIN_VOLUME_HARIAN) & \
             (nilai_transaksi_20 >= MIN_NILAI_TRANSAKSI_HARIAN)

    # --- Confluence (Stoch + Supertrend + BB) ---
    stoch_golden_cross = (df["Stoch_K"].shift(1) <= df["Stoch_D"].shift(1)) & (df["Stoch_K"] > df["Stoch_D"])
    supertrend_bullish = df["ST_Direction"] == 1
    jarak_lower_pct = ((df["Close"] - df["BB_lower"]) / df["BB_lower"]) * 100
    di_lower_band = jarak_lower_pct <= BB_LOWER_THRESHOLD_PCT

    skor = stoch_golden_cross.astype(int) + supertrend_bullish.astype(int) + di_lower_band.astype(int)
    sinyal_confluence = (skor >= MIN_SKOR_ALERT) & likuid

    # --- CMF Bullish ---
    cmf_bullish = df["CMF"] > CMF_THRESHOLD

    # --- Akumulasi OBV (harga flat + OBV naik selama OBV_LOOKBACK hari) ---
    harga_flat_obv_pct = ((df["Close"] - df["Close"].shift(OBV_LOOKBACK)) / df["Close"].shift(OBV_LOOKBACK)) * 100
    obv_naik = df["OBV"] > df["OBV"].shift(OBV_LOOKBACK)
    akumulasi_obv = obv_naik & (harga_flat_obv_pct.abs() <= OBV_MAX_HARGA_FLAT_PCT)

    # --- A/D Divergence ---
    harga_flat_ad_pct = ((df["Close"] - df["Close"].shift(AD_LOOKBACK)) / df["Close"].shift(AD_LOOKBACK)) * 100
    ad_naik = df["AD"] > df["AD"].shift(AD_LOOKBACK)
    ad_divergence = ad_naik & (harga_flat_ad_pct.abs() <= AD_MAX_HARGA_FLAT_PCT)

    # --- Akumulasi Terkonfirmasi (OBV + A/D + CMF semua sejalan) ---
    sinyal_akumulasi = akumulasi_obv & ad_divergence & cmf_bullish & likuid

    # --- ARA Kemarin ---
    prev_close = df["Close"].shift(1)
    persen_kenaikan = ((df["Close"] - prev_close) / prev_close) * 100
    batas_ara = prev_close.apply(get_batas_ara)
    sinyal_ara = (persen_kenaikan >= (batas_ara - TOLERANSI_ARA_PCT)) & likuid

    return {
        "confluence": sinyal_confluence,
        "akumulasi": sinyal_akumulasi,
        "ara": sinyal_ara,
    }


# ============================================================
# 4. SIMULASI FORWARD RETURN
# ============================================================

def simulasikan_trades(ticker, df, sinyal_dict):
    """Untuk setiap tanggal sinyal = True, simulasikan beli di OPEN hari
    berikutnya, jual di CLOSE setelah N hari (horizon), untuk tiap horizon
    di HORIZON_HARI. Return list of dict per transaksi."""
    hasil = []
    n = len(df)

    for kategori, sinyal_series in sinyal_dict.items():
        indeks_sinyal = np.where(sinyal_series.fillna(False).values)[0]

        for idx in indeks_sinyal:
            entry_idx = idx + 1  # beli di open besoknya
            if entry_idx >= n:
                continue
            entry_price = df["Open"].iloc[entry_idx]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            for horizon in HORIZON_HARI:
                exit_idx = entry_idx + horizon - 1
                if exit_idx >= n:
                    continue
                exit_price = df["Close"].iloc[exit_idx]
                if pd.isna(exit_price):
                    continue

                return_pct = ((exit_price - entry_price) / entry_price) * 100
                hasil.append({
                    "Ticker": ticker,
                    "Kategori": kategori,
                    "Tanggal_Sinyal": df.index[idx].strftime("%Y-%m-%d"),
                    "Horizon_Hari": horizon,
                    "Entry": round(entry_price, 1),
                    "Exit": round(exit_price, 1),
                    "Return_%": round(return_pct, 2),
                })

    return hasil


# ============================================================
# 5. AMBIL DAFTAR SAHAM (sama seperti full_scanner_telegram.py)
# ============================================================

def get_watchlist():
    if GUNAKAN_DAFTAR_MANUAL_DULU:
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
# 6. JALANKAN BACKTEST
# ============================================================

def run_backtest():
    tickers = get_watchlist()
    print(f"Memulai backtest {len(tickers)} saham, periode {PERIODE_BACKTEST}...\n")

    semua_trade = []
    berhasil, gagal = 0, 0

    for i, ticker in enumerate(tickers):
        try:
            df = yf.download(ticker, period=PERIODE_BACKTEST, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 100:
                gagal += 1
                continue

            sinyal_dict = hitung_semua_sinyal(df)
            trades = simulasikan_trades(ticker.replace(".JK", ""), df, sinyal_dict)
            semua_trade.extend(trades)
            berhasil += 1
        except Exception as e:
            gagal += 1
        finally:
            time.sleep(JEDA_ANTAR_REQUEST)
            if (i + 1) % 10 == 0:
                print(f"   ...progress {i+1}/{len(tickers)}")

    print(f"\nSelesai. Berhasil diproses: {berhasil}, gagal/dilewati: {gagal}\n")

    if not semua_trade:
        print("Tidak ada sinyal historis yang ditemukan untuk saham-saham ini.")
        return

    df_trade = pd.DataFrame(semua_trade)

    # --- Ringkasan per kategori & horizon ---
    print(f"{'='*80}")
    print("RINGKASAN HASIL BACKTEST")
    print(f"{'='*80}")

    ringkasan = df_trade.groupby(["Kategori", "Horizon_Hari"])["Return_%"].agg(
        Jumlah_Sinyal="count",
        Win_Rate_Pct=lambda x: round((x > 0).mean() * 100, 1),
        Rata_Rata_Return_Pct=lambda x: round(x.mean(), 2),
        Median_Return_Pct=lambda x: round(x.median(), 2),
        Return_Terbaik=lambda x: round(x.max(), 2),
        Return_Terburuk=lambda x: round(x.min(), 2),
    ).reset_index()

    print(ringkasan.to_string(index=False))

    nama_file_ringkasan = f"backtest_ringkasan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    nama_file_detail = f"backtest_detail_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    ringkasan.to_csv(nama_file_ringkasan, index=False)
    df_trade.to_csv(nama_file_detail, index=False)

    print(f"\nRingkasan disimpan ke: {nama_file_ringkasan}")
    print(f"Detail semua transaksi disimpan ke: {nama_file_detail}")
    print("\nCara baca: Win_Rate_Pct = persentase sinyal yang untung (belum dikurangi")
    print("fee/pajak). Bandingkan antar kategori & horizon untuk lihat mana yang")
    print("paling konsisten - ingat keterbatasan yang dijelaskan di awal file ini.")


if __name__ == "__main__":
    run_backtest()
