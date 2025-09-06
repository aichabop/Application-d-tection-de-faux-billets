# streamlit_app.py — fond complet RVB (248,142,85)
# ---------------------------------------------------------------------------------
# Lancer :
#   pip install streamlit pandas numpy requests altair pillow
#   streamlit run streamlit_app.py
# ---------------------------------------------------------------------------------

import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

EXPECTED_COLS = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

# ============================
# CONFIG APP
# ============================
st.set_page_config(
    page_title="Détection de faux billets – ML",
    page_icon="💶",
    layout="wide",
)

# ============================
# STYLES — fond uniforme RVB (248,142,85)
# ============================
CUSTOM_CSS = """
<style>
:root{
  --primary-rgb: 248,142,85;
  --primary: rgb(var(--primary-rgb));
  --card:#FFFFFF;
  --bd:#E6E6E6;
  --ink:#0F172A;
  --muted:#475569;
}

/* fond uniforme */
.stApp{background: rgb(var(--primary-rgb));}
.block-container{padding-top:1rem;}

/* cartes glass */
.glass{background:var(--card);border:1px solid var(--bd);border-radius:18px;box-shadow:0 10px 28px rgba(var(--primary-rgb),.15);padding:1.1rem 1.2rem}
.hero-title{margin:0;color:var(--ink);font-size:2.3rem;font-weight:800;text-align:center}
.hero-sub{color:var(--muted);margin:.35rem auto 0;font-size:1.08rem;max-width:900px;text-align:center}

/* boutons */
.stButton>button{border-radius:12px;font-weight:700;padding:.7rem 1.1rem;border:1px solid var(--bd);
                 background:linear-gradient(90deg,rgb(248,142,85),rgb(255,180,120)); color:#fff}

/* métriques */
.metric-card{display:flex;gap:12px;align-items:center}
.metric-emoji{font-size:1.4rem}
.metric-body h3{margin:.1rem 0 0 0;font-size:1.6rem}
.badge-pill{display:inline-block;padding:.3rem .7rem;border-radius:999px;border:1px solid var(--bd);font-weight:700}

.dataframe th, .dataframe td {color: var(--ink) !important;}
hr{border:none;border-top:2px solid var(--bd);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================
# IMAGES
# ============================
HERO_IMG = "https://images.pexels.com/photos/6266518/pexels-photo-6266518.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=1600"
GRID_IMG_1 = "https://images.pexels.com/photos/4386370/pexels-photo-4386370.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=1200"
GRID_IMG_2 = "https://images.pexels.com/photos/730547/pexels-photo-730547.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=1200"
GRID_IMG_3 = "https://images.pexels.com/photos/15206825/pexels-photo-15206825.jpeg?auto=compress&cs=tinysrgb&dpr=2&w=1200"

# ============================
# HEADER
# ============================
st.markdown(
    f"""
    <div class='glass'>
      <h1 class='hero-title'>💶 Détection automatique de faux billets</h1>
      <p class='hero-sub'>IA & FastAPI au service de la <b>sécurité des transactions</b> : chargez un CSV, lancez les prédictions,
      et obtenez des <b>visualisations</b> et un export des résultats.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.image(HERO_IMG, use_container_width=True, caption="Vérification sous lumière UV – Image Pexels (gratuite)")
st.markdown("---")

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.title("⚙️ Réglages API")
    api_base = st.text_input("URL de l'API", value="https://application-d-tection-de-faux-billets.onrender.com")
    endpoint = st.text_input("Endpoint", value="/predict")
    mode = st.radio("Mode d'envoi", ["Fichier CSV (multipart)", "JSON (records)"])
    st.markdown("---")
    label_true = st.text_input("Libellé classe 1", value="Vrai")
    label_false = st.text_input("Libellé classe 0", value="Faux")

# ============================
# API helpers
# ============================
def post_csv(url, df):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = requests.post(url, files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def post_json(url, df):
    payload = df[EXPECTED_COLS].to_dict(orient="records")
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

# ============================
# SECTION UPLOAD
# ============================
st.header("📥 Charger vos données (CSV)")
uploaded = st.file_uploader("Uploader un fichier .csv", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.markdown("<div class='glass'><b>Aperçu du CSV</b></div>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    if st.button("🚀 Lancer les prédictions", use_container_width=True):
        try:
            url = api_base.rstrip("/") + "/" + endpoint.lstrip("/")
            resp = post_csv(url, df) if mode.startswith("Fichier") else post_json(url, df)

            preds = resp.get("predictions", []) if isinstance(resp, dict) else []
            probas = resp.get("probabilities", []) if isinstance(resp, dict) else []
            pred_df = pd.DataFrame({"prediction": preds, "proba": probas})

            out = df.copy()
            if len(pred_df) == len(out):
                out["Probabilité_classe_1"] = pred_df["proba"]
                out["Classe"] = np.where(pred_df["prediction"].astype(int)==1, label_true, label_false)
            else:
                out["Classe"] = np.nan

            st.success("✔️ Prédictions reçues")
            st.balloons()

            tabs = st.tabs(["Table", "Statistiques", "Graphiques", "Export", "Aide"])

            # --- TABLE
            with tabs[0]:
                st.dataframe(out, use_container_width=True)

            # --- STATISTIQUES
            with tabs[1]:
                total = len(out)
                n_true = int((out.get("Classe","_") == label_true).sum())
                n_false = int((out.get("Classe","_") == label_false).sum())

                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown(f"<div class='glass metric-card'><div class='metric-emoji'>📦</div><div class='metric-body'><div class='badge-pill' style='background:#FFEFE0;'>Total</div><h3>{total} lignes</h3></div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='glass metric-card'><div class='metric-emoji'>✅</div><div class='metric-body'><div class='badge-pill' style='background:#FFE5D0;color:#0f5132;'> {label_true} </div><h3>{n_true}</h3></div></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='glass metric-card'><div class='metric-emoji'>⚠️</div><div class='metric-body'><div class='badge-pill' style='background:#FFF0F0;color:#842029;'> {label_false} </div><h3>{n_false}</h3></div></div>", unsafe_allow_html=True)

                # Bar chart horizontal
                if "Classe" in out.columns:
                    bar_df = out["Classe"].value_counts().rename_axis("Classe").reset_index(name="Nombre")
                    domain = [label_true, label_false] if label_true in bar_df["Classe"].values else list(bar_df["Classe"].values)
                    range_colors = ["rgb(248,142,85)", "#FF6B6B"] if len(domain)>=2 else ["rgb(248,142,85)"]
                    bar = (alt.Chart(bar_df)
                           .mark_bar(cornerRadius=8)
                           .encode(
                               y=alt.Y("Classe:N", sort=domain, title=""),
                               x=alt.X("Nombre:Q", title="Nombre d'observations"),
                               color=alt.Color("Classe:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
                               tooltip=["Classe", "Nombre"],
                           ).properties(height=160))
                    st.altair_chart(bar, use_container_width=True)

                if "Probabilité_classe_1" in out.columns:
                    st.markdown("**Résumé des probabilités (classe 1)**")
                    st.dataframe(out["Probabilité_classe_1"].describe().to_frame(), use_container_width=True)

            # --- GRAPHIQUES
            with tabs[2]:
                if "Probabilité_classe_1" in out.columns:
                    base = alt.Chart(out.reset_index()).properties(height=320)
                    hist = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                        x=alt.X("Probabilité_classe_1:Q", bin=alt.Bin(maxbins=30), title="Probabilité (classe 1)"),
                        y=alt.Y("count():Q", title="Effectif"),
                        color=alt.value("rgb(248,142,85)"),
                        tooltip=["count()"],
                    )
                    st.altair_chart(hist, use_container_width=True)

                    dens = alt.Chart(out).transform_density(
                        "Probabilité_classe_1", as_=["Probabilité_classe_1", "Densité"], extent=[0,1], steps=200
                    ).mark_area(opacity=0.35).encode(
                        x="Probabilité_classe_1:Q", y="Densité:Q", color=alt.value("rgba(248,142,85,0.6)")
                    ).properties(height=220)
                    st.altair_chart(dens, use_container_width=True)

                    line = base.mark_line(point=alt.OverlayMarkDef(filled=True, size=45)).encode(
                        x=alt.X("index:Q", title="Index"),
                        y=alt.Y("Probabilité_classe_1:Q", title="Probabilité (classe 1)"),
                        color=alt.Color("Classe:N", scale=alt.Scale(domain=[label_true,label_false], range=["rgb(248,142,85)","#FF6B6B"]), title="Classe"),
                        tooltip=["index", "Probabilité_classe_1", "Classe"],
                    )
                    st.altair_chart(line, use_container_width=True)

                if set(["diagonal","length"]).issubset(out.columns):
                    scatter = alt.Chart(out).mark_circle(size=80, opacity=0.75).encode(
                        x=alt.X("diagonal:Q", title="Diagonal"),
                        y=alt.Y("length:Q", title="Longueur"),
                        color=alt.Color("Classe:N", scale=alt.Scale(domain=[label_true,label_false], range=["rgb(248,142,85)","#FF6B6B"]), legend=alt.Legend(title="Classe")),
                        tooltip=list(out.columns)
                    ).properties(height=320)
                    st.altair_chart(scatter, use_container_width=True)

            # --- EXPORT
            with tabs[3]:
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Télécharger les résultats (CSV)", data=csv_bytes, file_name="predictions_billets.csv", mime="text/csv")

            # --- A PROPOS
            with tabs[4]:
                st.markdown(
                    """
                    **Notes**
                    Votre allié contre la fausse monnaie.
                    Vérifiez vos billets rapidement et en toute confiance.
                    """
                )

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez l'URL et que FastAPI tourne.")
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur HTTP: {e}")
        except Exception as e:
            st.error("Une erreur est survenue pendant l'inférence.")
            st.exception(e)
else:
    st.info("Chargez un CSV pour commencer.")

# ============================
# GALERIE
# ============================
st.markdown("---")
st.subheader("🔎 Galerie illustrative (libre d'utilisation)")
col1, col2, col3 = st.columns(3)
for c, url in zip([col1, col2, col3], [GRID_IMG_1, GRID_IMG_2, GRID_IMG_3]):
    with c:
        st.image(url, use_container_width=True)
