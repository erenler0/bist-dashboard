"""
╔══════════════════════════════════════════════════════════════════╗
║      BİST FİNANSAL ANALİZ DASHBOARD  —  dashboard.py           ║
║  Streamlit tabanlı iki katmanlı interaktif analiz arayüzü      ║
║                                                                  ║
║  Çalıştırma:                                                     ║
║      streamlit run dashboard.py                                  ║
║                                                                  ║
║  Gerekli paketler:                                               ║
║      pip install streamlit yfinance pandas pandas-ta plotly      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Yerel modüller
from veri_ve_indikatorler import (
    bist_hisse_listesi,
    veri_cek,
    indiktor_hesapla,
    ozet_metrik,
    ZAMAN_DILIMLERI,
    LOOKBACK_SECENEKLER,
    PIVOT_AYARLARI,
)
from formasyon_motoru import formasyonlari_tara, Formasyon, Pivot


# ─────────────────────────────────────────────────────────────────────────────
#  SAYFA YAPILANDIRMASI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BIST Finansal Analiz Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Özel CSS: karanlık borsa teması
st.markdown("""
<style>
/* Ana arka plan */
.stApp { background-color: #0D1117; color: #C9D1D9; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

/* Metrik kartları */
div[data-testid="metric-container"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    font-size: 1.4rem !important;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Tab başlıkları */
button[data-baseweb="tab"] {
    color: #8B949E !important;
    font-weight: 600;
    letter-spacing: 0.05em;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
}

/* Formasyon badge'leri */
.formasyon-bullish {
    background-color: rgba(38,166,154,0.15);
    border: 1px solid #26A69A;
    border-left: 4px solid #26A69A;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
}
.formasyon-bearish {
    background-color: rgba(239,83,80,0.1);
    border: 1px solid #EF5350;
    border-left: 4px solid #EF5350;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
}
.guc-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 6px;
}

/* Başlık */
h1, h2, h3 { color: #E6EDF3 !important; }
.stMarkdown p { color: #C9D1D9; }

/* Seçim kutuları */
.stSelectbox label, .stSlider label { color: #8B949E !important; }

/* Divider */
hr { border-color: #30363D; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def veri_yukle(hisse_kodu: str, gun: int, interval: str = "1d") -> pd.DataFrame:
    df = veri_cek(hisse_kodu, gun=gun, interval=interval)
    if not df.empty:
        df = indiktor_hesapla(df, interval=interval)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def formasyon_yukle(hisse_kodu: str, gun: int, pivot_pencere: int, interval: str = "1d") -> dict:
    df = veri_cek(hisse_kodu, gun=gun, interval=interval)
    if df.empty:
        return {"pivotlar": [], "formasyonlar": [], "ozet": {}}
    pivot_ayar = PIVOT_AYARLARI.get(interval, PIVOT_AYARLARI["1d"])
    return formasyonlari_tara(
        df,
        pivot_pencere=pivot_pencere,
        tolerans=pivot_ayar["tolerans"],
    )


def renk_al(deger: float, pozitif_yesil: bool = True) -> str:
    if deger > 0:
        return "#26A69A" if pozitif_yesil else "#EF5350"
    elif deger < 0:
        return "#EF5350" if pozitif_yesil else "#26A69A"
    return "#8B949E"


# ─────────────────────────────────────────────────────────────────────────────
#  ANA GRAFİK OLUŞTURUCU
# ─────────────────────────────────────────────────────────────────────────────

def ana_grafik_olustur(
    df: pd.DataFrame,
    hisse_kodu: str,
    formasyonlar: list,
    pivotlar: list,
    pivot_goster: bool = True,
    formasyon_goster: bool = True,
    gosterge_secim: list = None,
    interval: str = "1d",
) -> go.Figure:
    """
    3 panelli interaktif Plotly grafiği:
      - Panel 1: Candlestick + SMA50 + SMA200 + VWAP + Formasyonlar + Pivotlar
      - Panel 2: RSI (bant çizgileriyle)
      - Panel 3: MACD + Histogram + Sinyal
    """
    if gosterge_secim is None:
        gosterge_secim = ["SMA 50", "SMA 200", "VWAP"]

    MACD_COL = "MACD_12_26_9"
    MACS_COL = "MACDs_12_26_9"
    MACH_COL = "MACDh_12_26_9"

    has_rsi  = "RSI_14" in df.columns
    has_macd = all(c in df.columns for c in [MACD_COL, MACS_COL, MACH_COL])

    row_heights = [0.58, 0.21, 0.21]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
    )

    x = df.index

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=df["Close"].squeeze(),
        name="OHLC",
        increasing=dict(line=dict(color="#26A69A", width=1), fillcolor="#26A69A"),
        decreasing=dict(line=dict(color="#EF5350", width=1), fillcolor="#EF5350"),
        whiskerwidth=0.6,
    ), row=1, col=1)

    # ── Hareketli Ortalamalar ─────────────────────────────────────────────────
    sma_kisa_p = df.attrs.get("sma_kisa_periyot", 50)
    sma_uzun_p = df.attrs.get("sma_uzun_periyot", 200)

    if "SMA 50" in gosterge_secim and "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["SMA_50"].squeeze(),
            name=f"SMA {sma_kisa_p}", mode="lines",
            line=dict(color="#FF9800", width=1.5),
        ), row=1, col=1)

    if "SMA 200" in gosterge_secim and "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["SMA_200"].squeeze(),
            name=f"SMA {sma_uzun_p}", mode="lines",
            line=dict(color="#2196F3", width=1.5, dash="dash"),
        ), row=1, col=1)

    if "VWAP" in gosterge_secim and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["VWAP"].squeeze(),
            name="VWAP", mode="lines",
            line=dict(color="#CE93D8", width=1.5, dash="dot"),
        ), row=1, col=1)

    # ── Pivot Noktaları ───────────────────────────────────────────────────────
    if pivot_goster and pivotlar:
        tepeler = [p for p in pivotlar if p.tur == "tepe"]
        dipler  = [p for p in pivotlar if p.tur == "dip"]

        if tepeler:
            fig.add_trace(go.Scatter(
                x=[p.tarih for p in tepeler],
                y=[p.fiyat for p in tepeler],
                mode="markers",
                name="Tepe Pivot",
                marker=dict(
                    symbol="triangle-down",
                    size=8,
                    color="#EF5350",
                    line=dict(color="#FFFFFF", width=1),
                ),
            ), row=1, col=1)

        if dipler:
            fig.add_trace(go.Scatter(
                x=[p.tarih for p in dipler],
                y=[p.fiyat for p in dipler],
                mode="markers",
                name="Dip Pivot",
                marker=dict(
                    symbol="triangle-up",
                    size=8,
                    color="#26A69A",
                    line=dict(color="#FFFFFF", width=1),
                ),
            ), row=1, col=1)

    # ── Formasyonlar ──────────────────────────────────────────────────────────
    shapes = []
    annotations = []

    if formasyon_goster and formasyonlar:
        renk_map = {
            "BULLISH": "rgba(38,166,154,0.12)",
            "BEARISH": "rgba(239,83,80,0.10)",
        }
        cerceve_renk = {
            "BULLISH": "#26A69A",
            "BEARISH": "#EF5350",
        }
        etiket_renk = {
            "BULLISH": "#26A69A",
            "BEARISH": "#EF5350",
        }

        for f in formasyonlar:
            dolgu = renk_map.get(f.tur, "rgba(255,255,255,0.05)")
            cerceve = cerceve_renk.get(f.tur, "#888")

            # Bölge kutusu
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=f.baslangic_tarih,
                x1=f.bitis_tarih,
                y0=0, y1=1,
                fillcolor=dolgu,
                line=dict(color=cerceve, width=1.5, dash="dot"),
                layer="below",
            ))

            # Pivotları bağlayan çizgiler (üst 3 nokta)
            if len(f.pivotlar) >= 3:
                pivot_x = [p.tarih for p in f.pivotlar[:3]]
                pivot_y = [p.fiyat for p in f.pivotlar[:3]]
                fig.add_trace(go.Scatter(
                    x=pivot_x, y=pivot_y,
                    mode="lines+markers",
                    name=f.ad,
                    showlegend=False,
                    line=dict(color=cerceve, width=1.5, dash="dashdot"),
                    marker=dict(size=7, color=cerceve, symbol="circle"),
                ), row=1, col=1)

            # Hedef fiyat çizgisi
            if f.hedef_fiyat:
                shapes.append(dict(
                    type="line",
                    xref="x", yref="y",
                    x0=f.bitis_tarih,
                    x1=df.index[-1],
                    y0=f.hedef_fiyat,
                    y1=f.hedef_fiyat,
                    line=dict(color=cerceve, width=1, dash="dash"),
                ))

            # Etiket
            kisa_ad = f.ad.split("(")[0].strip()[:12]
            annotations.append(dict(
                x=f.baslangic_tarih,
                y=1.0,
                xref="x",
                yref="paper",
                text=f"▲ {kisa_ad}" if f.tur == "BULLISH" else f"▼ {kisa_ad}",
                showarrow=False,
                font=dict(size=10, color=etiket_renk.get(f.tur, "#888")),
                align="left",
                bgcolor="rgba(13,17,23,0.8)",
                bordercolor=cerceve,
                borderwidth=1,
                borderpad=3,
            ))

    # ── RSI ───────────────────────────────────────────────────────────────────
    if has_rsi:
        rsi_vals = df["RSI_14"].squeeze()
        fig.add_trace(go.Scatter(
            x=x, y=rsi_vals,
            name="RSI 14",
            line=dict(color="#9C27B0", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.06)",
        ), row=2, col=1)

        for seviye, renk in [(70, "rgba(239,83,80,0.4)"), (30, "rgba(38,166,154,0.4)"), (50, "rgba(100,100,100,0.3)")]:
            fig.add_hline(
                y=seviye,
                line=dict(color=renk, width=1, dash="dot"),
                row=2, col=1,
            )
        fig.update_yaxes(range=[0, 100], row=2, col=1,
                         tickvals=[30, 50, 70],
                         ticktext=["30", "50", "70"])

    # ── MACD ──────────────────────────────────────────────────────────────────
    if has_macd:
        hist = df[MACH_COL].squeeze()
        hist_renk = ["#26A69A" if v >= 0 else "#EF5350" for v in hist]

        fig.add_trace(go.Bar(
            x=x, y=hist,
            name="Histogram",
            marker_color=hist_renk,
            opacity=0.65,
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=df[MACD_COL].squeeze(),
            name="MACD",
            line=dict(color="#FF9800", width=1.5),
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=df[MACS_COL].squeeze(),
            name="Sinyal",
            line=dict(color="#2196F3", width=1.5),
        ), row=3, col=1)

        fig.add_hline(y=0, line=dict(color="rgba(150,150,150,0.4)", width=1), row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        template="plotly_dark",
        title=dict(
            text=f"<b>{hisse_kodu}</b>  ·  Teknik Analiz  ·  <span style='color:#58A6FF'>{interval}</span>",
            font=dict(size=16, color="#E6EDF3", family="monospace"),
            x=0.01,
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            bgcolor="rgba(22,27,34,0.9)",
            bordercolor="#30363D",
            borderwidth=1,
            font=dict(size=11, color="#C9D1D9"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#161B22",
            bordercolor="#30363D",
            font=dict(color="#C9D1D9", size=12),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        shapes=shapes,
        annotations=annotations,
        height=780,
    )

    # Eksen stilleri
    axis_style = dict(
        gridcolor="#21262D",
        gridwidth=0.5,
        zerolinecolor="#30363D",
        tickfont=dict(color="#8B949E", size=11),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # X ekseni: zaman dilimine göre boşlukları gizle
    if interval == "1d":
        # Günlük: hafta sonlarını gizle
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
        )
    else:
        # İntragün: hafta sonları + gece saatlerini (21:00–09:30 Türkiye) gizle
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[21, 9.5], pattern="hour"),
            ],
        )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  FORMASYON KARTLARI (HTML)
# ─────────────────────────────────────────────────────────────────────────────

def formasyon_karti_html(f: Formasyon) -> str:
    renk   = "#26A69A" if f.tur == "BULLISH" else "#EF5350"
    bg     = "rgba(38,166,154,0.08)" if f.tur == "BULLISH" else "rgba(239,83,80,0.08)"
    ok     = "▲" if f.tur == "BULLISH" else "▼"
    bas_t  = str(f.baslangic_tarih)[:10]
    bit_t  = str(f.bitis_tarih)[:10]
    hedef  = f"🎯 Hedef: <b>{f.hedef_fiyat:.2f}</b>" if f.hedef_fiyat else ""
    guc_w  = int(f.guc)

    return f"""
    <div style="
        background:{bg};
        border:1px solid {renk};
        border-left:4px solid {renk};
        border-radius:8px;
        padding:12px 14px;
        margin:8px 0;
        font-family: monospace;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:{renk};font-weight:700;font-size:0.95rem;">
                {ok} {f.ad}
            </span>
            <span style="
                background:{renk};
                color:#0D1117;
                border-radius:4px;
                padding:2px 8px;
                font-size:0.75rem;
                font-weight:700;
            ">{f.tur}</span>
        </div>
        <div style="color:#8B949E;font-size:0.8rem;margin-top:4px;">
            📅 {bas_t} → {bit_t}
        </div>
        <div style="color:#C9D1D9;font-size:0.82rem;margin-top:6px;">
            {f.aciklama}
        </div>
        <div style="color:{renk};font-size:0.85rem;margin-top:4px;">
            {hedef}
        </div>
        <div style="margin-top:8px;">
            <div style="display:flex;justify-content:space-between;
                        color:#8B949E;font-size:0.75rem;">
                <span>Güç Skoru</span><span>{f.guc:.0f}/100</span>
            </div>
            <div style="background:#21262D;border-radius:2px;height:4px;margin-top:3px;">
                <div style="
                    background:{renk};
                    width:{guc_w}%;
                    height:4px;
                    border-radius:2px;
                "></div>
            </div>
        </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_olustur():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-size:1.1rem;font-weight:700;
                        color:#E6EDF3;font-family:monospace;
                        letter-spacing:0.05em;">BIST ANALİZ</div>
            <div style="font-size:0.75rem;color:#8B949E;">
                Teknik Analiz Dashboard
            </div>
        </div>
        <hr style="border-color:#30363D;margin:8px 0 16px;">
        """, unsafe_allow_html=True)

        st.markdown("**🔍 Hisse Seçimi**")
        hisse_dict = bist_hisse_listesi()
        hisse_listesi = sorted(hisse_dict.keys())
        hisse_adlari = [f"{k}  ·  {hisse_dict[k]}" for k in hisse_listesi]

        hisse_map = {f"{k}  ·  {v}": k for k, v in hisse_dict.items()}

        secili_gosterim = st.selectbox(
            "Hisse",
            options=hisse_adlari,
            index=hisse_adlari.index("AKBNK.IS  ·  Akbank") if "AKBNK.IS  ·  Akbank" in hisse_adlari else 0,
            label_visibility="collapsed",
        )
        secili_hisse = hisse_map.get(secili_gosterim, "AKBNK.IS")

        # Manuel giriş
        st.markdown('<div style="color:#8B949E;font-size:0.78rem;margin-top:4px;">veya manuel giriş:</div>', unsafe_allow_html=True)
        manuel = st.text_input("", placeholder="SASA.IS", label_visibility="collapsed").strip().upper()
        if manuel:
            secili_hisse = manuel if manuel.endswith(".IS") else manuel + ".IS"

        st.markdown("---")

        # ── Zaman Dilimi ──────────────────────────────────────────────────────
        st.markdown("**⏱ Zaman Dilimi**")
        interval_secenekler = {
            "1d  · Günlük" : "1d",
            "1h  · Saatlik": "1h",
            "15m · 15 Dak" : "15m",
            "5m  · 5 Dak"  : "5m",
        }
        secili_interval_label = st.radio(
            "Zaman Dilimi",
            options=list(interval_secenekler.keys()),
            index=0,
            label_visibility="collapsed",
        )
        interval = interval_secenekler[secili_interval_label]

        # Seçilen zaman dilimine göre lookback seçenekleri
        st.markdown("**📅 Geriye Bakış Süresi**")
        lookback_map = LOOKBACK_SECENEKLER[interval]
        varsayilan_idx = min(2, len(lookback_map) - 1)
        secili_lookback = st.radio(
            "Geriye Bakış",
            options=list(lookback_map.keys()),
            index=varsayilan_idx,
            horizontal=True,
            label_visibility="collapsed",
        )
        gun = lookback_map[secili_lookback]

        # Kısa zaman dilimi uyarısı
        if interval in ("5m", "15m"):
            st.markdown(f"""
            <div style="background:rgba(255,152,0,0.1);border:1px solid #FF9800;
                        border-radius:6px;padding:8px 10px;font-size:0.75rem;color:#FF9800;
                        margin-top:4px;">
                ⚠️ {interval} verisi yfinance'te max 60 günlük geçmişe sahip.<br>
                Piyasa saatleri dışında veri gelmeyebilir.
            </div>
            """, unsafe_allow_html=True)
        elif interval == "1h":
            st.markdown(f"""
            <div style="background:rgba(33,150,243,0.08);border:1px solid #2196F3;
                        border-radius:6px;padding:8px 10px;font-size:0.75rem;color:#90CAF9;
                        margin-top:4px;">
                ℹ️ Saatlik veri max 730 günlük geçmiş sunar.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Göstergeler
        st.markdown("**📈 Göstergeler**")
        gosterge_secim = st.multiselect(
            "Göstergeler",
            options=["SMA 50", "SMA 200", "VWAP"],
            default=["SMA 50", "SMA 200", "VWAP"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Formasyon ayarları
        st.markdown("**🔎 Formasyon Tespiti**")
        pivot_pencere = st.slider(
            "Pivot Penceresi",
            min_value=3, max_value=15, value=5,
            help="Yerel tepe/dip için iki yanda bakılacak mum sayısı",
        )
        pivot_goster = st.checkbox("Pivot noktalarını göster", value=True)
        formasyon_goster = st.checkbox("Formasyonları göster", value=True)

        st.markdown("---")

        # Yenile butonu
        if st.button("🔄  Veriyi Yenile", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(f"""
        <div style="color:#8B949E;font-size:0.72rem;margin-top:16px;text-align:center;">
            Son güncelleme<br>{datetime.now().strftime('%H:%M:%S')}
        </div>
        """, unsafe_allow_html=True)

    return secili_hisse, gun, interval, gosterge_secim, pivot_pencere, pivot_goster, formasyon_goster


# ─────────────────────────────────────────────────────────────────────────────
#  ANA UYGULAMA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Tarayıcıdan hisse seçimi yönlendirmesi
    if "secili_hisse_override" in st.session_state:
        override = st.session_state.pop("secili_hisse_override")
        st.session_state["_override_hisse"] = override

    secili_hisse, gun, interval, gosterge_secim, pivot_pencere, pivot_goster, formasyon_goster = sidebar_olustur()

    # ── Veri yükleme ──────────────────────────────────────────────────────────
    zd_etiket = ZAMAN_DILIMLERI.get(interval, {}).get("etiket", interval)
    with st.spinner(f"📡 {secili_hisse} · {zd_etiket} verisi yükleniyor…"):
        df = veri_yukle(secili_hisse, gun, interval)

    if df.empty:
        st.error(
            f"❌ **{secili_hisse}** için veri alınamadı. "
            "Sembolün doğru olduğundan emin olun (örn. AKBNK.IS)."
        )
        return

    with st.spinner("🔎 Formasyonlar taranıyor…"):
        f_sonuc = formasyon_yukle(secili_hisse, gun, pivot_pencere, interval)

    formasyonlar = f_sonuc["formasyonlar"]
    pivotlar     = f_sonuc["pivotlar"]
    ozet         = f_sonuc["ozet"]
    metrikler    = ozet_metrik(df)

    # ── Başlık ────────────────────────────────────────────────────────────────
    hisse_dict = bist_hisse_listesi()
    hisse_adi  = hisse_dict.get(secili_hisse, "")
    degisim    = metrikler.get("degisim_pct", 0)
    degisim_ok = "▲" if degisim >= 0 else "▼"
    degisim_renk = "#26A69A" if degisim >= 0 else "#EF5350"

    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:16px;
                padding:4px 0 16px;border-bottom:1px solid #30363D;
                margin-bottom:16px;">
        <span style="font-size:1.8rem;font-weight:800;
                     color:#E6EDF3;font-family:monospace;">
            {secili_hisse}
        </span>
        <span style="font-size:1.1rem;color:#8B949E;">{hisse_adi}</span>
        <span style="background:#21262D;border:1px solid #30363D;
                     border-radius:5px;padding:2px 10px;
                     font-size:0.8rem;color:#58A6FF;font-family:monospace;
                     font-weight:700;">{interval} · {zd_etiket}</span>
        <span style="font-size:1.5rem;font-weight:700;
                     color:{degisim_renk};font-family:monospace;margin-left:auto;">
            {metrikler.get('son_fiyat', '—')} ₺
            <span style="font-size:0.95rem;">
                {degisim_ok} {abs(degisim):.2f}%
            </span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Üst metrikler ─────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Son Fiyat", f"{metrikler.get('son_fiyat', '—')} ₺",
                  delta=f"{degisim:+.2f}%")
    with m2:
        st.metric("52H Yüksek", f"{metrikler.get('52h_yuksek', '—')} ₺")
    with m3:
        st.metric("52H Düşük", f"{metrikler.get('52h_dusuk', '—')} ₺")
    with m4:
        rsi = metrikler.get("rsi", None)
        rsi_label = "Aşırı Alım" if rsi and rsi >= 70 else ("Aşırı Satım" if rsi and rsi <= 30 else "Nötr")
        st.metric("RSI (14)", f"{rsi:.1f}" if rsi else "—", delta=rsi_label)
    with m5:
        macd = metrikler.get("macd", None)
        sinyal = metrikler.get("sinyal", None)
        if macd is not None and sinyal is not None:
            macd_durum = "↑ Alım" if macd > sinyal else "↓ Satış"
            st.metric("MACD", f"{macd:.4f}", delta=macd_durum)
        else:
            st.metric("MACD", "—")
    with m6:
        st.metric("Formasyon", f"{ozet.get('Toplam', 0)} adet",
                  delta=f"↑{ozet.get('BULLISH',0)} / ↓{ozet.get('BEARISH',0)}")

    # ── Sekmeli içerik ────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Teknik Grafik",
        "🔷  Formasyonlar",
        "📋  Ham Veri & İhracat",
        "🔭  Piyasa Tarayıcı",
    ])

    # ── TAB 1: Grafik ─────────────────────────────────────────────────────────
    with tab1:
        fig = ana_grafik_olustur(
            df, secili_hisse,
            formasyonlar if formasyon_goster else [],
            pivotlar if pivot_goster else [],
            pivot_goster=pivot_goster,
            formasyon_goster=formasyon_goster,
            gosterge_secim=gosterge_secim,
            interval=interval,
        )
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png", "filename": secili_hisse,
                "height": 900, "width": 1600,
            },
        })

    # ── TAB 2: Formasyonlar ───────────────────────────────────────────────────
    with tab2:
        if not formasyonlar:
            st.info("🔍 Seçilen zaman aralığında ve pivot ayarlarıyla formasyon tespit edilemedi. "
                    "Pivot penceresi değerini düşürmeyi veya daha uzun bir dönem seçmeyi deneyin.")
        else:
            col_l, col_r = st.columns([1, 1])

            # ─ Sol: Özet sayım ─────────────────────────────────────────────
            with col_l:
                st.markdown("#### 📊 Tespit Özeti")
                ozet_cols = st.columns(3)
                for i, (k, v) in enumerate([
                    ("Cup & Handle", ozet.get("Cup & Handle", 0)),
                    ("OBO / TOBO",   ozet.get("OBO / TOBO", 0)),
                    ("İkili T/D",    ozet.get("İkili Tepe/Dip", 0)),
                ]):
                    with ozet_cols[i]:
                        st.markdown(f"""
                        <div style="background:#161B22;border:1px solid #30363D;
                                    border-radius:8px;padding:12px;text-align:center;">
                            <div style="font-size:1.6rem;font-weight:800;
                                        color:#E6EDF3;font-family:monospace;">{v}</div>
                            <div style="font-size:0.7rem;color:#8B949E;">{k}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("#### 🟢 Bullish Formasyonlar")
                bullish = [f for f in formasyonlar if f.tur == "BULLISH"]
                if bullish:
                    for f in bullish:
                        st.markdown(formasyon_karti_html(f), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#8B949E;font-style:italic;">Yok</div>', unsafe_allow_html=True)

            # ─ Sağ: Bearish formasyonlar + pivot tablo ─────────────────────
            with col_r:
                st.markdown("#### 🔴 Bearish Formasyonlar")
                bearish = [f for f in formasyonlar if f.tur == "BEARISH"]
                if bearish:
                    for f in bearish:
                        st.markdown(formasyon_karti_html(f), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#8B949E;font-style:italic;">Yok</div>', unsafe_allow_html=True)

                if pivotlar:
                    st.markdown("---")
                    st.markdown("#### 📍 Son 10 Pivot Noktası")
                    pivot_df = pd.DataFrame([{
                        "Tarih"  : str(p.tarih)[:10],
                        "Fiyat"  : round(p.fiyat, 2),
                        "Tür"    : "⬆ Tepe" if p.tur == "tepe" else "⬇ Dip",
                    } for p in pivotlar[-10:]])
                    st.dataframe(
                        pivot_df,
                        use_container_width=True,
                        hide_index=True,
                    )

        # ── Matriks entegrasyon bilgisi ────────────────────────────────────
        st.markdown("---")
        with st.expander("🔌 Matriks / Takas Verisi Entegrasyonu"):
            st.markdown("""
            **`formasyonlari_tara(df)` fonksiyonu aşağıdaki yapıyı döndürür:**
            ```python
            from formasyon_motoru import formasyonlari_tara

            sonuc = formasyonlari_tara(df)

            for f in sonuc["formasyonlar"]:
                print(f.ad, f.baslangic_tarih, f.bitis_tarih, f.tur)
                # Takas verisiyle karşılaştırma:
                # takas_df[takas_df["tarih"].between(f.baslangic_tarih, f.bitis_tarih)]
            ```
            Her `Formasyon` nesnesi şu alanları içerir:
            `ad`, `tur`, `guc`, `baslangic_tarih`, `bitis_tarih`,
            `hedef_fiyat`, `aciklama`, `pivotlar`
            """)

    # ── TAB 3: Ham Veri ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 📋 Son 60 Günlük Veri")

        gosterilecek_sutunlar = ["Open", "High", "Low", "Close", "Volume",
                                  "SMA_50", "SMA_200", "RSI_14", "VWAP"]
        mevcut_sutunlar = [c for c in gosterilecek_sutunlar if c in df.columns]

        gosterim_df = df[mevcut_sutunlar].tail(60).copy()
        gosterim_df.index = gosterim_df.index.strftime("%d.%m.%Y")

        # Sütun adlarını Türkçeleştir
        yeniden_adlandir = {
            "Open": "Açılış", "High": "Yüksek", "Low": "Düşük",
            "Close": "Kapanış", "Volume": "Hacim",
            "SMA_50": "SMA 50", "SMA_200": "SMA 200",
            "RSI_14": "RSI", "VWAP": "VWAP",
        }
        gosterim_df.rename(columns=yeniden_adlandir, inplace=True)

        fmt = {
            "Açılış": "{:.2f}", "Yüksek": "{:.2f}",
            "Düşük": "{:.2f}", "Kapanış": "{:.2f}",
            "Hacim": "{:,.0f}", "SMA 50": "{:.2f}",
            "SMA 200": "{:.2f}", "RSI": "{:.1f}", "VWAP": "{:.2f}",
        }
        fmt_mevcut = {k: v for k, v in fmt.items() if k in gosterim_df.columns}
        st.dataframe(
            gosterim_df.style.format(fmt_mevcut, na_rep="—"),
            use_container_width=True,
            height=400,
        )

        # CSV indirme
        csv = df.tail(200).to_csv(encoding="utf-8-sig")
        st.download_button(
            "⬇️  Son 200 Günü CSV Olarak İndir",
            data=csv,
            file_name=f"{secili_hisse}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=False,
        )

    # ── TAB 4: Piyasa Tarayıcı ────────────────────────────────────────────────
    tarayici_tab(tab4, pivot_pencere)


# ─────────────────────────────────────────────────────────────────────────────
#  PİYASA TARAYICI  (Tab 4)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def tum_bist_tara(gun: int, pivot_pencere: int, secili_formasyonlar: tuple) -> list:
    """
    Tüm BIST hisselerini paralel olmayan sıralı şekilde tarar.
    Sonuçlar 10 dakika cache'lenir.
    """
    hisse_dict = bist_hisse_listesi()
    sonuclar = []

    for kod, ad in hisse_dict.items():
        try:
            df = veri_cek(kod, gun=gun)
            if df.empty or len(df) < 60:
                continue

            tarama = formasyonlari_tara(df, pivot_pencere=pivot_pencere)
            bulunan = tarama["formasyonlar"]

            if not bulunan:
                continue

            # Sadece seçili formasyon türlerini filtrele
            if secili_formasyonlar:
                bulunan = [
                    f for f in bulunan
                    if any(sf in f.ad for sf in secili_formasyonlar)
                ]

            if not bulunan:
                continue

            # En son formasyonu al
            en_son = max(bulunan, key=lambda f: f.bitis_idx)
            close  = df["Close"].values.flatten()
            son_fiyat = float(close[-1])
            onceki    = float(close[-2]) if len(close) > 1 else son_fiyat
            degisim   = ((son_fiyat - onceki) / onceki) * 100

            sonuclar.append({
                "Kod"          : kod,
                "Şirket"       : ad,
                "Formasyon"    : en_son.ad,
                "Yön"          : en_son.tur,
                "Güç"          : en_son.guc,
                "Başlangıç"    : str(en_son.baslangic_tarih)[:10],
                "Bitiş"        : str(en_son.bitis_tarih)[:10],
                "Hedef"        : en_son.hedef_fiyat,
                "Son Fiyat"    : round(son_fiyat, 2),
                "Değişim %"    : round(degisim, 2),
                "Toplam"       : len(bulunan),
                "_aciklama"    : en_son.aciklama,
            })
        except Exception:
            continue

    # Güç skoru ve bitiş tarihine göre sırala
    sonuclar.sort(key=lambda x: (x["Güç"], x["Bitiş"]), reverse=True)
    return sonuclar


def tarayici_tab(tab, pivot_pencere: int):
    with tab:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#161B22,#0D1117);
                    border:1px solid #30363D;border-radius:10px;
                    padding:16px 20px;margin-bottom:20px;">
            <div style="font-size:1.1rem;font-weight:700;color:#E6EDF3;">
                🔭 Piyasa Tarayıcı
            </div>
            <div style="color:#8B949E;font-size:0.85rem;margin-top:4px;">
                Tüm BIST hisselerini aynı anda formasyon için tarar.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tarayıcıya özel zaman dilimi & dönem seçici ───────────────────────
        td1, td2 = st.columns([2, 3])
        with td1:
            st.markdown("**⏱ Tarama Zaman Dilimi**")
            tarama_interval_map = {
                "1d · Günlük" : "1d",
                "1h · Saatlik": "1h",
                "15m · 15 Dak": "15m",
                "5m  · 5 Dak" : "5m",
            }
            tarama_interval_label = st.radio(
                "Tarama Zaman Dilimi",
                options=list(tarama_interval_map.keys()),
                index=0,
                label_visibility="collapsed",
            )
            tarama_interval = tarama_interval_map[tarama_interval_label]

        with td2:
            st.markdown("**📅 Tarama Dönemi**")
            tarama_lookback_map = LOOKBACK_SECENEKLER[tarama_interval]
            tarama_lookback_label = st.radio(
                "Tarama Dönemi",
                options=list(tarama_lookback_map.keys()),
                index=min(2, len(tarama_lookback_map) - 1),
                horizontal=True,
                label_visibility="collapsed",
            )
            tarama_gun = tarama_lookback_map[tarama_lookback_label]

            if tarama_interval in ("5m", "15m"):
                st.markdown("""
                <div style="background:rgba(255,152,0,0.1);border:1px solid #FF9800;
                            border-radius:5px;padding:6px 10px;font-size:0.72rem;color:#FF9800;">
                    ⚠️ İntragün tarama daha uzun sürebilir ve bazı hisselerde veri gelmeyebilir.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Tarama ayarları ───────────────────────────────────────────────────
        ayar1, ayar2, ayar3 = st.columns([2, 2, 1])

        with ayar1:
            formasyon_filtre = st.multiselect(
                "Formasyon Filtresi",
                options=["Çanak/Kulp", "OBO", "TOBO", "İkili Tepe", "İkili Dip"],
                default=[],
                placeholder="Tümü (boş bırakın)",
            )

        with ayar2:
            yon_filtre = st.radio(
                "Yön Filtresi",
                options=["Tümü", "Sadece BULLISH 🟢", "Sadece BEARISH 🔴"],
                horizontal=True,
            )

        with ayar3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            tara_btn = st.button("🔍  Tüm BIST'i Tara", use_container_width=True, type="primary")

        # ── Tarama durumu ─────────────────────────────────────────────────────
        if "tarama_yapildi" not in st.session_state:
            st.session_state.tarama_yapildi = False
        if "tarama_sonuclari" not in st.session_state:
            st.session_state.tarama_sonuclari = []

        if tara_btn:
            st.session_state.tarama_yapildi = False
            ilerleme = st.progress(0, text="Tarama başlıyor…")
            durum_yazi = st.empty()

            hisse_dict = bist_hisse_listesi()
            toplam = len(hisse_dict)
            gecici_sonuclar = []

            pivot_ayar   = PIVOT_AYARLARI.get(tarama_interval, PIVOT_AYARLARI["1d"])
            min_mum      = 20  # intragün için daha az mum yeterli

            for i, (kod, ad) in enumerate(hisse_dict.items()):
                ilerleme.progress(
                    (i + 1) / toplam,
                    text=f"Tarıyor: **{kod}** ({i+1}/{toplam})"
                )
                durum_yazi.markdown(
                    f'<div style="color:#8B949E;font-size:0.8rem;">📡 {ad} · {tarama_interval}</div>',
                    unsafe_allow_html=True
                )
                try:
                    df_t = veri_cek(kod, gun=tarama_gun, interval=tarama_interval)
                    if df_t.empty or len(df_t) < min_mum:
                        continue

                    tarama_sonuc = formasyonlari_tara(
                        df_t,
                        pivot_pencere=pivot_ayar["pencere"],
                        tolerans=pivot_ayar["tolerans"],
                    )
                    bulunan = tarama_sonuc["formasyonlar"]
                    if not bulunan:
                        continue

                    if formasyon_filtre:
                        bulunan = [f for f in bulunan if any(sf in f.ad for sf in formasyon_filtre)]
                    if not bulunan:
                        continue

                    en_son = max(bulunan, key=lambda f: f.bitis_idx)
                    close  = df_t["Close"].values.flatten()
                    son_fiyat = float(close[-1])
                    onceki    = float(close[-2]) if len(close) > 1 else son_fiyat
                    degisim   = ((son_fiyat - onceki) / onceki) * 100

                    gecici_sonuclar.append({
                        "Kod"       : kod,
                        "Şirket"    : ad,
                        "Zaman"     : tarama_interval,
                        "Formasyon" : en_son.ad,
                        "Yön"       : en_son.tur,
                        "Güç"       : en_son.guc,
                        "Başlangıç" : str(en_son.baslangic_tarih)[:16],
                        "Bitiş"     : str(en_son.bitis_tarih)[:16],
                        "Hedef ₺"   : en_son.hedef_fiyat,
                        "Fiyat ₺"   : round(son_fiyat, 2),
                        "Değişim %" : round(degisim, 2),
                        "Açıklama"  : en_son.aciklama,
                    })
                except Exception:
                    continue

            gecici_sonuclar.sort(key=lambda x: (x["Güç"], x["Bitiş"]), reverse=True)
            st.session_state.tarama_sonuclari = gecici_sonuclar
            st.session_state.tarama_yapildi = True
            ilerleme.empty()
            durum_yazi.empty()

        # ── Sonuçları göster ──────────────────────────────────────────────────
        if st.session_state.tarama_yapildi and st.session_state.tarama_sonuclari:
            sonuclar = st.session_state.tarama_sonuclari

            # Yön filtresi uygula
            if yon_filtre == "Sadece BULLISH 🟢":
                sonuclar = [s for s in sonuclar if s["Yön"] == "BULLISH"]
            elif yon_filtre == "Sadece BEARISH 🔴":
                sonuclar = [s for s in sonuclar if s["Yön"] == "BEARISH"]

            # ── Özet banner ───────────────────────────────────────────────────
            bullish_sayi = sum(1 for s in sonuclar if s["Yön"] == "BULLISH")
            bearish_sayi = sum(1 for s in sonuclar if s["Yön"] == "BEARISH")

            b1, b2, b3, b4 = st.columns(4)
            for col, label, val, renk in [
                (b1, "Toplam Sinyal",   len(sonuclar),   "#58A6FF"),
                (b2, "🟢 Bullish",      bullish_sayi,    "#26A69A"),
                (b3, "🔴 Bearish",      bearish_sayi,    "#EF5350"),
                (b4, "Taranan Hisse",   len(bist_hisse_listesi()), "#8B949E"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style="background:#161B22;border:1px solid #30363D;
                                border-radius:8px;padding:12px;text-align:center;">
                        <div style="font-size:1.8rem;font-weight:800;
                                    color:{renk};font-family:monospace;">{val}</div>
                        <div style="font-size:0.72rem;color:#8B949E;
                                    text-transform:uppercase;letter-spacing:0.08em;">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Kart grid ─────────────────────────────────────────────────────
            st.markdown("#### 📋 Tespit Edilen Formasyonlar")

            for s in sonuclar:
                yon_renk = "#26A69A" if s["Yön"] == "BULLISH" else "#EF5350"
                yon_bg   = "rgba(38,166,154,0.08)" if s["Yön"] == "BULLISH" else "rgba(239,83,80,0.07)"
                ok       = "▲" if s["Yön"] == "BULLISH" else "▼"
                degisim_renk = "#26A69A" if s["Değişim %"] >= 0 else "#EF5350"
                degisim_ok   = "▲" if s["Değişim %"] >= 0 else "▼"
                hedef_str = f"🎯 {s['Hedef ₺']:.2f} ₺" if s["Hedef ₺"] else ""
                guc_w = int(s["Güç"])

                col_kart, col_btn = st.columns([11, 1])
                with col_kart:
                    st.markdown(f"""
                    <div style="background:{yon_bg};border:1px solid {yon_renk};
                                border-left:5px solid {yon_renk};border-radius:8px;
                                padding:12px 16px;margin:4px 0;font-family:monospace;">
                        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                            <div>
                                <span style="font-size:1.05rem;font-weight:800;color:#E6EDF3;">
                                    {s['Kod']}
                                </span>
                                <span style="color:#8B949E;font-size:0.85rem;margin-left:8px;">
                                    {s['Şirket']}
                                </span>
                            </div>
                            <div style="display:flex;gap:12px;align-items:center;">
                                <span style="color:{degisim_renk};font-size:0.9rem;font-weight:600;">
                                    {s['Fiyat ₺']:.2f} ₺ &nbsp;
                                    {degisim_ok} {abs(s['Değişim %']):.2f}%
                                </span>
                                <span style="background:{yon_renk};color:#0D1117;
                                             border-radius:4px;padding:2px 8px;
                                             font-size:0.72rem;font-weight:700;">
                                    {ok} {s['Yön']}
                                </span>
                            </div>
                        </div>
                        <div style="color:{yon_renk};font-size:0.88rem;
                                    font-weight:600;margin-top:6px;">
                            {s['Formasyon']}
                        </div>
                        <div style="color:#8B949E;font-size:0.78rem;margin-top:3px;">
                            📅 {s['Başlangıç']} → {s['Bitiş']}
                            &nbsp;&nbsp;{hedef_str}
                        </div>
                        <div style="color:#C9D1D9;font-size:0.78rem;margin-top:3px;">
                            {s['Açıklama']}
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    color:#8B949E;font-size:0.72rem;margin-top:8px;">
                            <span>Güç Skoru</span><span>{s['Güç']:.0f}/100</span>
                        </div>
                        <div style="background:#21262D;border-radius:2px;height:3px;margin-top:3px;">
                            <div style="background:{yon_renk};width:{guc_w}%;
                                        height:3px;border-radius:2px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_btn:
                    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
                    if st.button("📈", key=f"git_{s['Kod']}", help=f"{s['Kod']} grafiğine git"):
                        st.session_state["secili_hisse_override"] = s["Kod"]
                        st.rerun()

            # ── CSV ihracat ───────────────────────────────────────────────────
            st.markdown("---")
            ihracat_df = pd.DataFrame([{
                k: v for k, v in s.items() if k != "Açıklama"
            } for s in sonuclar])

            csv_tarama = ihracat_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "⬇️  Tarama Sonuçlarını CSV Olarak İndir",
                data=csv_tarama,
                file_name=f"bist_tarama_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

        elif st.session_state.tarama_yapildi:
            st.warning("🔍 Seçilen kriterlere uyan formasyon bulunamadı. Filtreleri gevşetin veya daha uzun dönem seçin.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#8B949E;">
                <div style="font-size:3rem;margin-bottom:12px;">🔭</div>
                <div style="font-size:1.1rem;font-weight:600;color:#C9D1D9;">
                    Tüm BIST hisselerini tek tıkla tarayın
                </div>
                <div style="font-size:0.85rem;margin-top:8px;">
                    Yukarıdaki <b style="color:#58A6FF;">Tüm BIST'i Tara</b> butonuna basın.<br>
                    ~60 hisse için yaklaşık 30-60 saniye sürer.
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
