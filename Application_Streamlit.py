# streamlit_app.py – version PREMIUM SIMPLIFIÉE avec design ultra moderne
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
    page_title="💎 AI Banknote Detective",
    page_icon="🔮",
    layout="wide",
)

# ============================
# STYLES ULTRA PREMIUM
# ============================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
  --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  --danger: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
  --bg: #0a0a0f;
  --glass: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.12);
  --text: #ffffff;
  --shadow: 0 20px 40px rgba(0,0,0,0.3);
  --glow: 0 0 30px rgba(102,126,234,0.4);
}

@keyframes float { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-15px); }}
@keyframes glow { 0%,100% { box-shadow: var(--shadow); } 50% { box-shadow: var(--glow); }}
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); }}

.stApp {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  background-image: 
    radial-gradient(circle at 20% 50%, rgba(102,126,234,0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 50%, rgba(240,147,251,0.1) 0%, transparent 50%);
  color: var(--text);
}

/* HERO SECTION */
.hero {
  background: var(--glass);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 3rem 2rem;
  margin: 2rem 0;
  text-align: center;
  animation: fadeIn 0.8s ease, glow 3s infinite;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  background: var(--primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 1rem;
  animation: float 4s infinite;
}

.hero-text {
  font-size: 1.3rem;
  opacity: 0.8;
  max-width: 700px;
  margin: 0 auto 2rem;
  line-height: 1.6;
}

.badge {
  background: var(--success);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 50px;
  font-weight: 600;
  display: inline-block;
  margin-bottom: 1rem;
}

/* SECTIONS */
.section-title {
  font-size: 2.5rem;
  font-weight: 800;
  margin: 3rem 0 2rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.section-icon {
  background: var(--primary);
  width: 50px;
  height: 50px;
  border-radius: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
}

.glass-card {
  background: var(--glass);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem;
  margin: 1rem 0;
  box-shadow: var(--shadow);
  transition: all 0.3s ease;
}

.glass-card:hover {
  transform: translateY(-5px);
  box-shadow: var(--glow);
}

/* BUTTONS */
.stButton>button {
  background: var(--primary) !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 1rem 2rem !important;
  font-weight: 700 !important;
  color: white !important;
  box-shadow: var(--shadow) !important;
  transition: all 0.3s ease !important;
  text-transform: uppercase !important;
}

.stButton>button:hover {
  transform: translateY(-3px) !important;
  box-shadow: var(--glow) !important;
}

/* METRICS */
.metric {
  background: var(--glass);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2rem;
  text-align: center;
  transition: all 0.3s ease;
}

.metric:hover { transform: translateY(-5px); }

.metric-number {
  font-size: 2.5rem;
  font-weight: 900;
  background: var(--primary);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.metric-label {
  font-size: 0.9rem;
  opacity: 0.7;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: 0.5rem;
}

/* SIDEBAR */
.css-1d391kg { background: var(--glass) !important; backdrop-filter: blur(20px) !important; }

/* FILE UPLOADER */
.stFileUploader {
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
  background: var(--glass) !important;
  padding: 2rem !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
  background: var(--glass);
  padding: 0.5rem;
  border-radius: 12px;
  gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
  border-radius: 8px;
  font-weight: 600;
}

.stTabs [aria-selected="true"] {
  background: var(--primary);
  color: white;
}

/* ALERTS */
.stSuccess, .stError, .stWarning, .stInfo {
  border-radius: 12px !important;
  border: none !important;
  box-shadow: var(--shadow) !important;
}

.stSuccess { background: var(--success) !important; }
.stError { background: var(--danger) !important; }

/* GALLERY */
.gallery img {
  border-radius: 16px;
  box-shadow: var(--shadow);
  transition: all 0.3s ease;
}

.gallery img:hover {
  transform: translateY(-10px) scale(1.02);
  box-shadow: var(--glow);
}
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
# HERO SECTION
# ============================
st.markdown("""
<div class='hero'>
  <div class='badge'>🤖 Powered by AI</div>
  <h1 class='hero-title'>💎 AI Banknote Detective</h1>
  <p class='hero-text'>
    Détection automatique de faux billets avec l'intelligence artificielle.
    Chargez vos données, obtenez des prédictions instantanées et visualisez les résultats.
  </p>
</div>
""", unsafe_allow_html=True)

st.image(HERO_IMG, use_container_width=True, caption="Vérification sous lumière UV")

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); width: 60px; height: 60px; 
                    border-radius: 15px; display: inline-flex; align-items: center; justify-content: center; 
                    font-size: 1.5rem; margin-bottom: 1rem;'>⚙️</div>
        <h2 style='margin: 0;'>Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    api_base = st.text_input("URL de l'API", value="https://application-d-tection-de-faux-billets.onrender.com")
    endpoint = st.text_input("Endpoint", value="/predict/")
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
# UPLOAD SECTION
# ============================
st.markdown("""
<div class='section-title'>
  <div class='section-icon'>📁</div>
  <span>Upload de données</span>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Chargez votre fichier CSV", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.markdown("<div class='glass-card'><h3>📊 Aperçu des données</h3></div>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

    missing_cols = [col for col in EXPECTED_COLS if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Colonnes manquantes ajoutées avec NaN : {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan

    if st.button("🚀 Lancer les prédictions", use_container_width=True):
        try:
            url = api_base.rstrip("/") + "/" + endpoint.lstrip("/")
            resp = post_json(url, df)

            preds = resp.get("predictions", []) if isinstance(resp, dict) else []
            probas = resp.get("probabilities", []) if isinstance(resp, dict) else []
            pred_df = pd.DataFrame({"prediction": preds, "proba": probas})

            out = df.copy()
            if len(pred_df) == len(out):
                out["Probabilité_classe_1"] = pred_df["proba"]
                out["Classe"] = np.where(pred_df["prediction"].astype(int)==1, label_true, label_false)
            else:
                out["Classe"] = np.nan

            st.success("🎉 Prédictions terminées !")
            st.balloons()

            tabs = st.tabs(["📋 Résultats", "📊 Statistiques", "📈 Graphiques", "💾 Export", "ℹ️ Aide"])

            with tabs[0]:
                st.markdown("<div class='glass-card'><h3>🔍 Résultats détaillés</h3></div>", unsafe_allow_html=True)
                st.dataframe(out, use_container_width=True)

            with tabs[1]:
                total = len(out)
                n_true = int((out.get("Classe","_") == label_true).sum())
                n_false = int((out.get("Classe","_") == label_false).sum())

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                        <div class='metric'>
                            <div style='font-size: 2rem;'>📦</div>
                            <div class='metric-number'>{total}</div>
                            <div class='metric-label'>Total</div>
                        </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                        <div class='metric'>
                            <div style='font-size: 2rem;'>✅</div>
                            <div class='metric-number'>{n_true}</div>
                            <div class='metric-label'>{label_true}</div>
                        </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                        <div class='metric'>
                            <div style='font-size: 2rem;'>⚠️</div>
                            <div class='metric-number'>{n_false}</div>
                            <div class='metric-label'>{label_false}</div>
                        </div>""", unsafe_allow_html=True)

                if "Classe" in out.columns:
                    bar_df = out["Classe"].value_counts().rename_axis("Classe").reset_index(name="Nombre")
                    domain = [label_true, label_false] if label_true in bar_df["Classe"].values else list(bar_df["Classe"].values)
                    range_colors = ["#4facfe", "#fa709a"] if len(domain)>=2 else ["#4facfe"]
                    
                    bar = alt.Chart(bar_df).mark_bar(cornerRadius=8).encode(
                        y=alt.Y("Classe:N", sort=domain, title=""),
                        x=alt.X("Nombre:Q", title="Nombre d'observations"),
                        color=alt.Color("Classe:N", scale=alt.Scale(domain=domain, range=range_colors), legend=None),
                        tooltip=["Classe", "Nombre"],
                    ).properties(height=200)
                    st.altair_chart(bar, use_container_width=True)

            with tabs[2]:
                if "Probabilité_classe_1" in out.columns:
                    # Histogramme
                    hist = alt.Chart(out.reset_index()).mark_bar(cornerRadius=6).encode(
                        x=alt.X("Probabilité_classe_1:Q", bin=alt.Bin(maxbins=25), title="Probabilité (classe 1)"),
                        y=alt.Y("count():Q", title="Fréquence"),
                        color=alt.value("#667eea")
                    ).properties(height=300)
                    st.altair_chart(hist, use_container_width=True)

                    # Nuage de points
                    if set(["diagonal","length"]).issubset(out.columns):
                        scatter = alt.Chart(out).mark_circle(size=80).encode(
                            x=alt.X("diagonal:Q", title="Diagonal"),
                            y=alt.Y("length:Q", title="Longueur"),
                            color=alt.Color("Classe:N", scale=alt.Scale(domain=[label_true,label_false], range=["#4facfe","#fa709a"])),
                            tooltip=list(out.columns)
                        ).properties(height=400)
                        st.altair_chart(scatter, use_container_width=True)

            with tabs[3]:
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Télécharger les résultats", data=csv_bytes, file_name="predictions_billets.csv", mime="text/csv")

            with tabs[4]:
                st.markdown("""
                <div class='glass-card'>
                    <h3>ℹ️ À propos</h3>
                    <p>Cette application utilise l'intelligence artificielle pour détecter les faux billets.
                    Chargez un fichier CSV avec les caractéristiques des billets et obtenez des prédictions instantanées.</p>
                </div>
                """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API")
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur HTTP: {e}")
        except Exception as e:
            st.error("Une erreur est survenue")
            st.exception(e)
else:
    st.info("Chargez un CSV pour commencer l'analyse")

# ============================
# GALERIE
# ============================
st.markdown("---")
st.markdown("""
<div class='section-title'>
  <div class='section-icon'>🖼️</div>
  <span>Galerie</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='gallery'><img src='{GRID_IMG_1}' style='width:100%; border-radius:16px;'></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='gallery'><img src='{GRID_IMG_2}' style='width:100%; border-radius:16px;'></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='gallery'><img src='{GRID_IMG_3}' style='width:100%; border-radius:16px;'></div>", unsafe_allow_html=True)