"""
╔══════════════════════════════════════════════════════════════════╗
║          FORMASYON MOTORU  —  formasyon_motoru.py               ║
║  Pivot tabanlı teknik formasyon tespit motoru                   ║
║  Dışarıya açık arayüz: formasyonlari_tara(df) → dict           ║
╚══════════════════════════════════════════════════════════════════╝

Desteklenen formasyonlar:
  1. Çanak/Kulp   (Cup and Handle)
  2. OBO / TOBO   (Head and Shoulders / Inverse H&S)
  3. İkili Tepe   (Double Top)
  4. İkili Dip    (Double Bottom)

Matriks / dış veri entegrasyonu için:
  formasyonlari_tara(df) fonksiyonu bir sözlük döndürür.
  Bu sözlüğü doğrudan takas verisiyle karşılaştırabilirsiniz.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  VERİ YAPILARI
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pivot:
    idx: int          # DataFrame içindeki satır indeksi
    tarih: object     # Gerçek tarih damgası
    fiyat: float
    tur: str          # 'tepe' | 'dip'


@dataclass
class Formasyon:
    ad: str
    tur: str               # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    guc: float             # 0-100 arası güven skoru
    baslangic_idx: int
    bitis_idx: int
    baslangic_tarih: object
    bitis_tarih: object
    hedef_fiyat: Optional[float] = None
    aciklama: str = ""
    pivotlar: List[Pivot] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  1. PİVOT NOKTALARI  (ZigZag mantığı)
# ─────────────────────────────────────────────────────────────────────────────

def pivot_bul(
    df: pd.DataFrame,
    pencere: int = 5,
    min_degisim_pct: float = 0.03,
) -> List[Pivot]:
    """
    Fiyat serisindeki yerel tepe ve dipleri tespit eder.

    Parametreler
    ------------
    pencere          : Her iki yanda bakılacak mum sayısı (yerellik tanımı)
    min_degisim_pct  : Pivot sayılmak için gereken minimum % değişim
    """
    close = df["Close"].values.flatten()
    high  = df["High"].values.flatten()
    low   = df["Low"].values.flatten()
    n     = len(close)
    pivotlar: List[Pivot] = []

    for i in range(pencere, n - pencere):
        # Yerel tepe kontrolü
        if high[i] == max(high[i - pencere: i + pencere + 1]):
            # Bir önceki pivottan yeterince uzak mı?
            if pivotlar and pivotlar[-1].tur == "tepe":
                if abs(high[i] - pivotlar[-1].fiyat) / pivotlar[-1].fiyat < min_degisim_pct:
                    # Daha yüksekse güncelle, değilse atla
                    if high[i] > pivotlar[-1].fiyat:
                        pivotlar[-1] = Pivot(i, df.index[i], high[i], "tepe")
                    continue
            pivotlar.append(Pivot(i, df.index[i], high[i], "tepe"))

        # Yerel dip kontrolü
        elif low[i] == min(low[i - pencere: i + pencere + 1]):
            if pivotlar and pivotlar[-1].tur == "dip":
                if abs(low[i] - pivotlar[-1].fiyat) / pivotlar[-1].fiyat < min_degisim_pct:
                    if low[i] < pivotlar[-1].fiyat:
                        pivotlar[-1] = Pivot(i, df.index[i], low[i], "dip")
                    continue
            pivotlar.append(Pivot(i, df.index[i], low[i], "dip"))

    return pivotlar


# ─────────────────────────────────────────────────────────────────────────────
#  2. ÇANAK / KULP  (Cup and Handle)
# ─────────────────────────────────────────────────────────────────────────────

def cup_and_handle_tara(
    df: pd.DataFrame,
    pivotlar: List[Pivot],
    tolerans: float = 0.05,
) -> List[Formasyon]:
    """
    Çanak/Kulp formasyonu:
      - Sol kenar tepe  → U şekli dip → Sağ kenar tepe  (≈ sol kenarla eşit)
      - Sağ kenardan sonra küçük düzeltme (kulp)
      - Kırılış: Kulpun üst bandını geçiş
    """
    sonuclar: List[Formasyon] = []
    tepeler = [p for p in pivotlar if p.tur == "tepe"]
    dipler  = [p for p in pivotlar if p.tur == "dip"]

    for i in range(len(tepeler) - 1):
        sol_kenar = tepeler[i]
        sag_kenar = tepeler[i + 1]

        # İki tepe arasında en az bir dip olmalı
        aradaki_dipler = [
            d for d in dipler
            if sol_kenar.idx < d.idx < sag_kenar.idx
        ]
        if not aradaki_dipler:
            continue

        canak_dibi = min(aradaki_dipler, key=lambda d: d.fiyat)

        # Çanak derinliği: sol kenarın en az %15'i
        derinlik = (sol_kenar.fiyat - canak_dibi.fiyat) / sol_kenar.fiyat
        if derinlik < 0.10:
            continue

        # Sağ kenar sol kenarın %5 toleransı içinde mi?
        fark = abs(sag_kenar.fiyat - sol_kenar.fiyat) / sol_kenar.fiyat
        if fark > tolerans:
            continue

        # Kulp: sağ kenardan sonra küçük dip var mı?
        kulp_dipler = [
            d for d in dipler
            if sag_kenar.idx < d.idx
            and (d.fiyat > canak_dibi.fiyat)        # dip çanak dibinden yüksek
            and (sag_kenar.fiyat - d.fiyat) / sag_kenar.fiyat < 0.15
        ]

        guc = 60.0
        if kulp_dipler:
            guc = 85.0  # Kulp varsa daha güçlü sinyal

        hedef = sag_kenar.fiyat + (sag_kenar.fiyat - canak_dibi.fiyat)

        sonuclar.append(Formasyon(
            ad="Çanak/Kulp (Cup & Handle)",
            tur="BULLISH",
            guc=guc,
            baslangic_idx=sol_kenar.idx,
            bitis_idx=sag_kenar.idx,
            baslangic_tarih=sol_kenar.tarih,
            bitis_tarih=sag_kenar.tarih,
            hedef_fiyat=round(hedef, 2),
            aciklama=(
                f"Sol Kenar: {sol_kenar.fiyat:.2f} | "
                f"Çanak Dibi: {canak_dibi.fiyat:.2f} | "
                f"Sağ Kenar: {sag_kenar.fiyat:.2f} | "
                f"{'Kulp Var ✓' if kulp_dipler else 'Kulp Yok'}"
            ),
            pivotlar=[sol_kenar, canak_dibi, sag_kenar],
        ))

    return sonuclar


# ─────────────────────────────────────────────────────────────────────────────
#  3. OBO / TOBO  (Head & Shoulders / Inverse)
# ─────────────────────────────────────────────────────────────────────────────

def obo_tobo_tara(
    df: pd.DataFrame,
    pivotlar: List[Pivot],
    tolerans: float = 0.06,
) -> List[Formasyon]:
    """
    OBO  (Omuz-Baş-Omuz)      : Üç tepe, orta tepe en yüksek
    TOBO (Ters OBO)            : Üç dip, orta dip en düşük
    Boyun çizgisi eğimi hesaplanır.
    """
    sonuclar: List[Formasyon] = []

    # ── OBO (Bearish) ─────────────────────────────────────────────────────────
    tepeler = [p for p in pivotlar if p.tur == "tepe"]
    dipler  = [p for p in pivotlar if p.tur == "dip"]

    for i in range(len(tepeler) - 2):
        sol_omuz = tepeler[i]
        bas      = tepeler[i + 1]
        sag_omuz = tepeler[i + 2]

        # Baş her iki omuzdan yüksek mi?
        if not (bas.fiyat > sol_omuz.fiyat and bas.fiyat > sag_omuz.fiyat):
            continue

        # Omuzlar birbirine yakın mı?
        omuz_fark = abs(sol_omuz.fiyat - sag_omuz.fiyat) / sol_omuz.fiyat
        if omuz_fark > tolerans:
            continue

        # Boyun çizgisi: sol-bas arası dip ve bas-sag arası dip
        sol_dip_adaylar = [
            d for d in dipler if sol_omuz.idx < d.idx < bas.idx
        ]
        sag_dip_adaylar = [
            d for d in dipler if bas.idx < d.idx < sag_omuz.idx
        ]
        if not sol_dip_adaylar or not sag_dip_adaylar:
            continue

        sol_dip = max(sol_dip_adaylar, key=lambda d: d.fiyat)
        sag_dip = max(sag_dip_adaylar, key=lambda d: d.fiyat)
        boyun   = (sol_dip.fiyat + sag_dip.fiyat) / 2

        yukseklik = bas.fiyat - boyun
        hedef     = boyun - yukseklik

        sonuclar.append(Formasyon(
            ad="OBO (Omuz-Baş-Omuz)",
            tur="BEARISH",
            guc=75.0,
            baslangic_idx=sol_omuz.idx,
            bitis_idx=sag_omuz.idx,
            baslangic_tarih=sol_omuz.tarih,
            bitis_tarih=sag_omuz.tarih,
            hedef_fiyat=round(hedef, 2),
            aciklama=(
                f"Sol Omuz: {sol_omuz.fiyat:.2f} | "
                f"Baş: {bas.fiyat:.2f} | "
                f"Sağ Omuz: {sag_omuz.fiyat:.2f} | "
                f"Boyun: {boyun:.2f}"
            ),
            pivotlar=[sol_omuz, bas, sag_omuz, sol_dip, sag_dip],
        ))

    # ── TOBO (Bullish) ────────────────────────────────────────────────────────
    for i in range(len(dipler) - 2):
        sol_omuz = dipler[i]
        bas      = dipler[i + 1]
        sag_omuz = dipler[i + 2]

        if not (bas.fiyat < sol_omuz.fiyat and bas.fiyat < sag_omuz.fiyat):
            continue

        omuz_fark = abs(sol_omuz.fiyat - sag_omuz.fiyat) / sol_omuz.fiyat
        if omuz_fark > tolerans:
            continue

        sol_tep_adaylar = [
            p for p in tepeler if sol_omuz.idx < p.idx < bas.idx
        ]
        sag_tep_adaylar = [
            p for p in tepeler if bas.idx < p.idx < sag_omuz.idx
        ]
        if not sol_tep_adaylar or not sag_tep_adaylar:
            continue

        sol_tep = min(sol_tep_adaylar, key=lambda p: p.fiyat)
        sag_tep = min(sag_tep_adaylar, key=lambda p: p.fiyat)
        boyun   = (sol_tep.fiyat + sag_tep.fiyat) / 2

        yukseklik = boyun - bas.fiyat
        hedef     = boyun + yukseklik

        sonuclar.append(Formasyon(
            ad="TOBO (Ters OBO)",
            tur="BULLISH",
            guc=75.0,
            baslangic_idx=sol_omuz.idx,
            bitis_idx=sag_omuz.idx,
            baslangic_tarih=sol_omuz.tarih,
            bitis_tarih=sag_omuz.tarih,
            hedef_fiyat=round(hedef, 2),
            aciklama=(
                f"Sol Omuz: {sol_omuz.fiyat:.2f} | "
                f"Baş: {bas.fiyat:.2f} | "
                f"Sağ Omuz: {sag_omuz.fiyat:.2f} | "
                f"Boyun: {boyun:.2f}"
            ),
            pivotlar=[sol_omuz, bas, sag_omuz, sol_tep, sag_tep],
        ))

    return sonuclar


# ─────────────────────────────────────────────────────────────────────────────
#  4. İKİLİ TEPE / DİP  (Double Top / Bottom)
# ─────────────────────────────────────────────────────────────────────────────

def ikili_tepe_dip_tara(
    df: pd.DataFrame,
    pivotlar: List[Pivot],
    tolerans: float = 0.04,
    min_aralik: int = 10,
) -> List[Formasyon]:
    """
    İkili Tepe : İki ardışık yakın tepe, aralarında belirgin dip
    İkili Dip  : İki ardışık yakın dip, aralarında belirgin tepe
    """
    sonuclar: List[Formasyon] = []
    tepeler = [p for p in pivotlar if p.tur == "tepe"]
    dipler  = [p for p in pivotlar if p.tur == "dip"]

    # ── İkili Tepe (Bearish) ──────────────────────────────────────────────────
    for i in range(len(tepeler) - 1):
        t1, t2 = tepeler[i], tepeler[i + 1]
        if (t2.idx - t1.idx) < min_aralik:
            continue
        fark = abs(t1.fiyat - t2.fiyat) / t1.fiyat
        if fark > tolerans:
            continue

        # Aradaki dip
        ara_dipler = [d for d in dipler if t1.idx < d.idx < t2.idx]
        if not ara_dipler:
            continue
        ara_dip = min(ara_dipler, key=lambda d: d.fiyat)

        # Dip yeterince derin mi?
        derinlik = (t1.fiyat - ara_dip.fiyat) / t1.fiyat
        if derinlik < 0.05:
            continue

        hedef = ara_dip.fiyat - (t1.fiyat - ara_dip.fiyat)

        sonuclar.append(Formasyon(
            ad="İkili Tepe (Double Top)",
            tur="BEARISH",
            guc=70.0,
            baslangic_idx=t1.idx,
            bitis_idx=t2.idx,
            baslangic_tarih=t1.tarih,
            bitis_tarih=t2.tarih,
            hedef_fiyat=round(hedef, 2),
            aciklama=(
                f"Tepe 1: {t1.fiyat:.2f} | "
                f"Ara Dip: {ara_dip.fiyat:.2f} | "
                f"Tepe 2: {t2.fiyat:.2f} | "
                f"Direnç: {(t1.fiyat+t2.fiyat)/2:.2f}"
            ),
            pivotlar=[t1, ara_dip, t2],
        ))

    # ── İkili Dip (Bullish) ───────────────────────────────────────────────────
    for i in range(len(dipler) - 1):
        d1, d2 = dipler[i], dipler[i + 1]
        if (d2.idx - d1.idx) < min_aralik:
            continue
        fark = abs(d1.fiyat - d2.fiyat) / d1.fiyat
        if fark > tolerans:
            continue

        ara_tepeler = [t for t in tepeler if d1.idx < t.idx < d2.idx]
        if not ara_tepeler:
            continue
        ara_tepe = max(ara_tepeler, key=lambda t: t.fiyat)

        yukselik = (ara_tepe.fiyat - d1.fiyat) / d1.fiyat
        if yukselik < 0.05:
            continue

        hedef = ara_tepe.fiyat + (ara_tepe.fiyat - d1.fiyat)

        sonuclar.append(Formasyon(
            ad="İkili Dip (Double Bottom)",
            tur="BULLISH",
            guc=70.0,
            baslangic_idx=d1.idx,
            bitis_idx=d2.idx,
            baslangic_tarih=d1.tarih,
            bitis_tarih=d2.tarih,
            hedef_fiyat=round(hedef, 2),
            aciklama=(
                f"Dip 1: {d1.fiyat:.2f} | "
                f"Ara Tepe: {ara_tepe.fiyat:.2f} | "
                f"Dip 2: {d2.fiyat:.2f} | "
                f"Destek: {(d1.fiyat+d2.fiyat)/2:.2f}"
            ),
            pivotlar=[d1, ara_tepe, d2],
        ))

    return sonuclar


# ─────────────────────────────────────────────────────────────────────────────
#  5. ANA ENTEGRASYON FONKSİYONU  (dışarıya açık API)
# ─────────────────────────────────────────────────────────────────────────────

def formasyonlari_tara(
    df: pd.DataFrame,
    pivot_pencere: int = 5,
    tolerans: float = 0.05,
) -> dict:
    """
    Verilen OHLCV DataFrame üzerinde tüm formasyonları tarar.

    Parametreler
    ------------
    df             : OHLCV verisi içeren DataFrame (yfinance formatı)
    pivot_pencere  : ZigZag için yerel pencere büyüklüğü
    tolerans       : Fiyat eşleşme toleransı (0.05 = %5)

    Döndürür
    --------
    dict:
        pivotlar     : List[Pivot]
        formasyonlar : List[Formasyon]   (tüm formasyonlar tek listede)
        ozet         : dict              (formasyon türlerine göre sayım)

    Matriks Entegrasyonu İçin:
        sonuc["formasyonlar"] içindeki her Formasyon nesnesi
        .baslangic_tarih ve .bitis_tarih alanlarıyla takas
        verisiyle zaman bazlı karşılaştırma yapılabilir.
    """
    if df is None or df.empty:
        return {"pivotlar": [], "formasyonlar": [], "ozet": {}}

    pivotlar = pivot_bul(df, pencere=pivot_pencere)

    cup_handle   = cup_and_handle_tara(df, pivotlar, tolerans)
    obo_tobo     = obo_tobo_tara(df, pivotlar, tolerans)
    ikili        = ikili_tepe_dip_tara(df, pivotlar, tolerans)

    tum_formasyonlar = cup_handle + obo_tobo + ikili
    # Başlangıç tarihine göre sırala
    tum_formasyonlar.sort(key=lambda f: f.baslangic_idx)

    ozet = {
        "Cup & Handle" : len(cup_handle),
        "OBO / TOBO"   : len(obo_tobo),
        "İkili Tepe/Dip": len(ikili),
        "Toplam"       : len(tum_formasyonlar),
        "BULLISH"      : sum(1 for f in tum_formasyonlar if f.tur == "BULLISH"),
        "BEARISH"      : sum(1 for f in tum_formasyonlar if f.tur == "BEARISH"),
    }

    return {
        "pivotlar"    : pivotlar,
        "formasyonlar": tum_formasyonlar,
        "ozet"        : ozet,
    }
