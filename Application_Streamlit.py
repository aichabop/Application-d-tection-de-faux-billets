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
# SIDEBAR — Réglages API
# ============================
with st.sidebar:
    st.title("⚙️ Réglages API")
    api_base = st.text_input(
        "URL de l'API", 
        value="https://application-d-tection-de-faux-billets.onrender.com"  # URL Render
    )
    mode = st.radio("Mode d'envoi", ["Fichier CSV (multipart)", "JSON (records)"])
    st.markdown("---")
    label_true = st.text_input("Libellé classe 1", value="Vrai")
    label_false = st.text_input("Libellé classe 0", value="Faux")

    # Choix automatique de l’endpoint
    if mode.startswith("Fichier"):
        endpoint = "/predict_csv/"
    else:
        endpoint = "/predict/"

    # Vérifier si l'API est joignable
    try:
        r = requests.get(api_base.rstrip("/") + "/")
        if r.status_code == 200:
            st.success("API joignable ✅")
        else:
            st.warning("API répond mais pas OK ⚠️")
    except:
        st.error("API non joignable ❌")

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
    st.dataframe(df.head(50), use_container_width=True)

    if st.button("🚀 Lancer les prédictions"):
        try:
            # Construire l’URL complète
            url = api_base.rstrip("/") + "/" + endpoint.lstrip("/")
            st.write(f"👉 URL appelée : {url}")   # DEBUG : afficher l’URL exacte

            # Appel API
            resp = post_csv(url, df) if mode.startswith("Fichier") else post_json(url, df)

            preds = resp.get("predictions", [])
            probas = resp.get("probabilities", [])
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
                    st.metric("Total lignes", total)
                with c2:
                    st.metric(f"{label_true}", n_true)
                with c3:
                    st.metric(f"{label_false}", n_false)

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
                    Vérifiez vos billets rapidement et en toute confiance grâce à cette application.
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
