import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Morocco Smart Guide", layout="wide")


# ------------------------------------------
# Sidebar navigation
# ------------------------------------------

st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Aller à :", ["Générateur d’itinéraire", "Chatbot IA"])

st.sidebar.markdown("---")
st.sidebar.caption("Morocco Smart Guide – IA Tourisme 🇲🇦")


# ==================================================
# PAGE 1 : GENERATEUR D’ITINERAIRE
# ==================================================

if page == "Générateur d’itinéraire":
    st.title("🧭 Générer un itinéraire personnalisé")

    city = st.text_input("Ville principale (ex : Marrakech)")
    duration = st.number_input("Durée (jours)", min_value=1, value=3)
    budget = st.selectbox("Budget", ["low", "medium", "high"])
    interests = st.text_input("Centres d’intérêt (culture, shopping, gastronomy...)")
    constraints = st.text_area("Contraintes (optionnel)", "")
    language = st.selectbox("Langue", ["fr", "en"], index=0)

    if st.button("Générer l’itinéraire", use_container_width=True):
        with st.spinner("Génération en cours…"):
            payload = {
                "city": city or None,
                "duration_days": duration,
                "budget": budget,
                "interests": [s.strip() for s in interests.split(",") if s.strip()],
                "constraints": constraints,
                "language": language,
            }

            try:
                resp = requests.post(f"{API_URL}/itinerary", json=payload)
                resp.raise_for_status()

                itinerary = resp.json()
                st.success("Itinéraire généré !")

                st.subheader(f"📍 Ville : {itinerary['city']}")
                st.markdown(f"**Durée :** {itinerary['duration_days']} jours")

                for day in itinerary["days"]:
                    st.markdown(f"## 🗓️ Jour {day['day_number']}")

                    cols = st.columns(3)

                    cols[0].markdown("### 🌅 Matin")
                    for a in day["morning"]:
                        cols[0].info(f"**{a['name']}**\n\n{a.get('description','')}")

                    cols[1].markdown("### 🌞 Après-midi")
                    for a in day["afternoon"]:
                        cols[1].info(f"**{a['name']}**\n\n{a.get('description','')}")

                    cols[2].markdown("### 🌙 Soir")
                    for a in day["evening"]:
                        cols[2].info(f"**{a['name']}**\n\n{a.get('description','')}")

            except Exception as e:
                st.error(f"Erreur : {e}")


# ==================================================
# PAGE 2 : CHATBOT IA
# ==================================================

if page == "Chatbot IA":
    st.title("💬 Chatbot Touristique IA")

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    user_msg = st.text_input("Votre message :", "")

    if st.button("Envoyer", use_container_width=True):
        if user_msg.strip():
            payload = {
                "session_id": st.session_state.chat_session_id,
                "message": user_msg,
            }

            try:
                resp = requests.post(f"{API_URL}/chat", json=payload)
                resp.raise_for_status()

                data = resp.json()

                # sauvegarde de la session
                st.session_state.chat_session_id = data["session_id"]

                st.markdown(f"**Vous :** {user_msg}")
                st.markdown(f"**Assistant :** {data['answer']}")

            except Exception as e:
                st.error(f"Erreur : {e}")
