"""
╔══════════════════════════════════════════════════════════════════╗
║          VERİ & İNDİKATÖR MODÜLÜ  —  veri_ve_indikatorler.py  ║
║  yfinance veri çekme + pandas-ta indikatör hesaplama           ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    raise ImportError("pip install yfinance")

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("pip install pandas-ta")


# ─────────────────────────────────────────────────────────────────────────────
#  BIST HİSSE LİSTESİ
# ─────────────────────────────────────────────────────────────────────────────

# BIST 100 bileşenleri + önemli BIST hisseleri (Yahoo Finance .IS eki)
BIST_HISSELER = {
    # BIST 30
    "AKBNK.IS": "Akbank",
    "ARCLK.IS": "Arçelik",
    "ASELS.IS": "Aselsan",
    "BIMAS.IS": "BİM Mağazalar",
    "EKGYO.IS": "Emlak Konut GYO",
    "EREGL.IS": "Ereğli Demir Çelik",
    "FROTO.IS": "Ford Otosan",
    "GARAN.IS": "Garanti BBVA",
    "HALKB.IS": "Halkbank",
    "ISCTR.IS": "İş Bankası C",
    "KCHOL.IS": "Koç Holding",
    "KOZAA.IS": "Koza Altın",
    "KOZAL.IS": "Koza Madencilik",
    "KRDMD.IS": "Kardemir",
    "MGROS.IS": "Migros",
    "ODAS.IS" : "Odaş Elektrik",
    "OYAKC.IS": "Oyak Çimento",
    "PETKM.IS": "Petkim",
    "PGSUS.IS": "Pegasus",
    "SAHOL.IS": "Sabancı Holding",
    "SASA.IS" : "SASA Polyester",
    "SISE.IS" : "Şişecam",
    "SOKM.IS" : "Şok Marketler",
    "TAVHL.IS": "TAV Havalimanları",
    "TCELL.IS": "Turkcell",
    "THYAO.IS": "Türk Hava Yolları",
    "TOASO.IS": "Tofaş",
    "TTKOM.IS": "Türk Telekom",
    "TUPRS.IS": "Tüpraş",
    "VAKBN.IS": "Vakıfbank",
    "YKBNK.IS": "Yapı Kredi",
    # BIST 50 ek hisseler
    "AEFES.IS": "Anadolu Efes",
    "AGHOL.IS": "Anadolu Grubu Holding",
    "AKSEN.IS": "Aksa Enerji",
    "ALARK.IS": "Alarko Holding",
    "ALBRK.IS": "Albaraka Türk",
    "ALFAS.IS": "Alfa Güneş Enerjisi",
    "ASUZU.IS": "Anadolu Isuzu",
    "BAGFS.IS": "Bagfaş",
    "BRISA.IS": "Brisa",
    "CEMTS.IS": "Çimentaş",
    "CIMSA.IS": "Çimsa",
    "DOHOL.IS": "Doğan Holding",
    "DYOBY.IS": "DYO Boya",
    "EGEEN.IS": "Ege Endüstri",
    "ENKAI.IS": "Enka İnşaat",
    "ERBOS.IS": "Erbosan",
    "EUPWR.IS": "Europower Enerji",
    "FENER.IS": "Fenerbahçe SK",
    "GESAN.IS": "Güneş Sigorta",
    "GENIL.IS": "Genilsan",
    "GLYHO.IS": "Global Yatırım Holding",
    "GOLTS.IS": "Göltaş Çimento",
    "GUBRF.IS": "Gübre Fabrikaları",
    "HEKTS.IS": "Hektaş",
    "INDES.IS": "İndeks Bilgisayar",
    "IPEKE.IS": "İpek Enerji",
    "ISGYO.IS": "İş GYO",
    "ISFIN.IS": "İş Finansal Kiralama",
    "KLNMA.IS": "Türkiye Kalkınma Bankası",
    "LOGO.IS" : "Logo Yazılım",
    "MAVI.IS" : "Mavi Giyim",
    "NTHOL.IS": "Net Holding",
    "OTKAR.IS": "Otokar",
    "PAPIL.IS": "Papilon Savunma",
    "PRKAB.IS": "Türk Prysmian Kablo",
    "QUAGR.IS": "QUA Granite",
    "REEDR.IS": "Reeder Teknoloji",
    "SAMAT.IS": "Samaş",
    "SELEC.IS": "Selçuk Ecza",
    "SKBNK.IS": "Şekerbank",
    "TATGD.IS": "Tat Gıda",
    "TBORG.IS": "Türk Tuborg",
    "TKFEN.IS": "Tekfen Holding",
    "TMSN.IS" : "Tamsan Transformatör",
    "TOASO.IS": "Tofaş",
    "TRGYO.IS": "Torunlar GYO",
    "TSKB.IS" : "TSKB",
    "TTRAK.IS": "Türk Traktör",
    "ULKER.IS": "Ülker Bisküvi",
    "VESBE.IS": "Vestel Beyaz Eşya",
    "VESTL.IS": "Vestel",
    "YATAS.IS": "Yataş",
    "ZRGYO.IS": "Ziraat GYO",
}


def bist_hisse_listesi() -> dict:
    """Kod → Ad eşleme sözlüğünü döndürür."""
    return BIST_HISSELER


# ─────────────────────────────────────────────────────────────────────────────
#  VERİ ÇEKME
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
#  ZAMAN DİLİMİ KONFİGÜRASYONU
# ─────────────────────────────────────────────────────────────────────────────

# Her zaman dilimi için: (yfinance interval kodu, varsayılan lookback, maks lookback gün, etiket)
ZAMAN_DILIMLERI = {
    "5m"  : dict(interval="5m",  varsayilan_gun=5,   maks_gun=60,  etiket="5 Dakika",  kisa="5d"),
    "15m" : dict(interval="15m", varsayilan_gun=10,  maks_gun=60,  etiket="15 Dakika", kisa="15d"),
    "1h"  : dict(interval="1h",  varsayilan_gun=30,  maks_gun=730, etiket="1 Saat",    kisa="1mo"),
    "1d"  : dict(interval="1d",  varsayilan_gun=365, maks_gun=3650,etiket="Günlük",    kisa="1y"),
}

# Her zaman dilimi için uygun lookback seçenekleri
LOOKBACK_SECENEKLER = {
    "5m"  : {"1 Gün": 1,  "3 Gün": 3,  "5 Gün": 5,   "10 Gün": 10},
    "15m" : {"3 Gün": 3,  "5 Gün": 5,  "10 Gün": 10,  "30 Gün": 30},
    "1h"  : {"1 Hafta": 7,"2 Hafta":14,"1 Ay": 30,    "3 Ay": 90},
    "1d"  : {"3 Ay": 90,  "6 Ay": 180, "1 Yıl": 365,  "2 Yıl": 730},
}

# Zaman dilimine göre pivot pencere ve tolerans ayarları
PIVOT_AYARLARI = {
    "5m"  : dict(pencere=3, tolerans=0.015),
    "15m" : dict(pencere=3, tolerans=0.020),
    "1h"  : dict(pencere=4, tolerans=0.030),
    "1d"  : dict(pencere=5, tolerans=0.050),
}

# İndikatör periyotları — kısa zaman dilimlerinde SMA200 yerine daha kısa periyotlar
INDIKTOR_PERIYOTLARI = {
    "5m"  : dict(sma_kisa=9,  sma_uzun=20,  rsi=9),
    "15m" : dict(sma_kisa=9,  sma_uzun=21,  rsi=14),
    "1h"  : dict(sma_kisa=20, sma_uzun=50,  rsi=14),
    "1d"  : dict(sma_kisa=50, sma_uzun=200, rsi=14),
}


def veri_cek(
    hisse_kodu: str,
    gun: int = 365,
    interval: str = "1d",
    period: Optional[str] = None,
) -> pd.DataFrame:
    """
    yfinance ile OHLCV verisi çeker.

    Parametreler
    ------------
    hisse_kodu : Yahoo Finance sembolü (örn. 'AKBNK.IS')
    gun        : Kaç günlük geçmiş
    interval   : Mum zaman dilimi — '5m', '15m', '1h', '1d'
    period     : yfinance period parametresi (verilirse gun/interval override edilir)

    Notlar
    ------
    - 5m / 15m : max 60 günlük geçmiş (yfinance kısıtı)
    - 1h       : max 730 günlük geçmiş
    - 1d       : 10 yıla kadar
    """
    try:
        ticker = yf.Ticker(hisse_kodu)

        if period:
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
        else:
            bitis     = datetime.today()
            # yfinance kısıtlarına göre lookback sınırla
            cfg       = ZAMAN_DILIMLERI.get(interval, ZAMAN_DILIMLERI["1d"])
            gercek_gun = min(gun, cfg["maks_gun"])
            baslangic = bitis - timedelta(days=gercek_gun)

            df = ticker.history(
                start=baslangic.strftime("%Y-%m-%d"),
                end=bitis.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
            )

        if df.empty:
            return pd.DataFrame()

        # Çok seviyeli sütunları düzelt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)

        # Timezone bilgisini kaldır (Streamlit uyumu için)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.sort_index(inplace=True)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                return pd.DataFrame()

        return df

    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
#  İNDİKATÖR HESAPLAMA
# ─────────────────────────────────────────────────────────────────────────────

def indiktor_hesapla(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """
    Zaman dilimine göre uyarlanmış SMA, RSI, MACD ve VWAP hesaplar.

    1d  → SMA 50/200, RSI 14
    1h  → SMA 20/50,  RSI 14
    15m → SMA 9/21,   RSI 14
    5m  → SMA 9/20,   RSI 9
    """
    if df.empty:
        return df

    df   = df.copy()
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()

    periyot = INDIKTOR_PERIYOTLARI.get(interval, INDIKTOR_PERIYOTLARI["1d"])
    p_kisa  = periyot["sma_kisa"]
    p_uzun  = periyot["sma_uzun"]
    p_rsi   = periyot["rsi"]

    # SMA — sütun adları sabit kalıyor (grafik kodu bunlara bağlı)
    df["SMA_50"]  = ta.sma(close, length=p_kisa)   # "kısa" SMA
    df["SMA_200"] = ta.sma(close, length=p_uzun)   # "uzun" SMA

    # SMA etiket bilgisi (grafik başlığı için)
    df.attrs["sma_kisa_periyot"] = p_kisa
    df.attrs["sma_uzun_periyot"] = p_uzun

    # RSI
    df["RSI_14"] = ta.rsi(close, length=p_rsi)
    df.attrs["rsi_periyot"] = p_rsi

    # MACD — kısa zaman dilimlerinde daha hassas parametreler
    if interval in ("5m", "15m"):
        macd_df = ta.macd(close, fast=8, slow=21, signal=5)
    else:
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)

    if macd_df is not None:
        for col in macd_df.columns:
            df[col] = macd_df[col].values

    # VWAP (kümülatif — intragün zaman dilimleri için anlamlı)
    typical_price = (high + low + close) / 3
    cum_vol  = volume.cumsum()
    cum_tpv  = (typical_price * volume).cumsum()
    df["VWAP"] = cum_tpv / cum_vol

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ÖZET METRİKLER
# ─────────────────────────────────────────────────────────────────────────────

def ozet_metrik(df: pd.DataFrame) -> dict:
    """
    Dashboard'da gösterilecek hızlı metrikler.
    """
    if df.empty:
        return {}

    close = df["Close"].squeeze()
    son   = float(close.iloc[-1])
    onceki = float(close.iloc[-2]) if len(close) > 1 else son
    degisim_pct = ((son - onceki) / onceki) * 100

    metrikler = {
        "son_fiyat"    : round(son, 2),
        "degisim_pct"  : round(degisim_pct, 2),
        "52h_yuksek"   : round(float(df["High"].max()), 2),
        "52h_dusuk"    : round(float(df["Low"].min()), 2),
        "ort_hacim"    : int(df["Volume"].mean()),
        "son_hacim"    : int(df["Volume"].iloc[-1]),
    }

    if "RSI_14" in df.columns:
        rsi_val = df["RSI_14"].dropna()
        if not rsi_val.empty:
            metrikler["rsi"] = round(float(rsi_val.iloc[-1]), 2)

    if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
        macd_val = df["MACD_12_26_9"].dropna()
        sig_val  = df["MACDs_12_26_9"].dropna()
        if not macd_val.empty:
            metrikler["macd"]   = round(float(macd_val.iloc[-1]), 4)
            metrikler["sinyal"] = round(float(sig_val.iloc[-1]), 4)

    return metrikler
