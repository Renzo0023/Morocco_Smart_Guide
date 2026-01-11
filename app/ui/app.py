# app/ui/app.py

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Morocco Smart Guide",
    layout="wide",
    page_icon="🇲🇦"
)

# ------------------------------------------
# Navigation
# ------------------------------------------

st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Aller à :", ["🧭 Générateur d’itinéraire", "💬 Chatbot IA"])

st.sidebar.markdown("---")
st.sidebar.caption("Morocco Smart Guide – IA Tourisme 🇲🇦")


# ==================================================
# PAGE 1 : GENERATEUR D’ITINERAIRE
# ==================================================

if page == "🧭 Générateur d’itinéraire":
    st.title("🧭 Générer un itinéraire personnalisé")

    city = st.text_input("Ville principale", placeholder="ex : Marrakech")
    duration = st.number_input("Durée (jours)", min_value=1, value=3)
    budget = st.selectbox("Budget", ["low", "medium", "high"], index=1)
    interests = st.text_input("Centres d’intérêt", placeholder="culture, gastronomy, shopping …")
    constraints = st.text_area("Contraintes", placeholder="optionnel")
    language = st.selectbox("Langue", ["fr", "en"], index=0)

    if st.button("Générer l’itinéraire", use_container_width=True):
        payload = {
            "city": city or None,
            "duration_days": int(duration),
            "budget": budget,
            "interests": [s.strip() for s in interests.split(",") if s.strip()],
            "constraints": constraints,
            "language": language,
        }

        try:
            with st.spinner("Génération de l’itinéraire…"):
                resp = requests.post(f"{API_URL}/itinerary", json=payload)
                resp.raise_for_status()

            itinerary = resp.json()
            st.success("Itinéraire généré avec succès !")

            st.subheader(f"📍 Ville : **{itinerary['city']}**")
            st.markdown(f"**Durée :** {itinerary['duration_days']} jours")

            for day in itinerary["days"]:
                st.markdown(f"## 🗓️ Jour {day['day_number']}")

                col_morning, col_afternoon, col_evening = st.columns(3)

                col_morning.markdown("### 🌅 Matin (09:00 – 13:00)")
                for a in day["morning"]:
                    time_range = ""
                    if a.get("start_time") and a.get("end_time"):
                        time_range = f"🕒 {a['start_time']} – {a['end_time']}\n\n"

                    url = a.get("maps_url")
                    if url:
                        title = f"[{a['name']}]({url})"
                    else:
                        title = a['name']

                    col_morning.info(
                        f"{time_range}**{title}**\n\n{a.get('description','')}"
                    )

                col_afternoon.markdown("### 🌞 Après-midi (14:00 – 18:00)")
                for a in day["afternoon"]:
                    time_range = ""
                    if a.get("start_time") and a.get("end_time"):
                        time_range = f"🕒 {a['start_time']} – {a['end_time']}\n\n"

                    url = a.get("maps_url")
                    if url:
                        title = f"[{a['name']}]({url})"
                    else:
                        title = a['name']

                    col_afternoon.info(
                        f"{time_range}**{title}**\n\n{a.get('description','')}"
                    )

                col_evening.markdown("### 🌙 Soir (à partir de 18:00)")
                for a in day["evening"]:
                    time_range = ""
                    if a.get("start_time") and a.get("end_time"):
                        time_range = f"🕒 {a['start_time']} – {a['end_time']}\n\n"

                    url = a.get("maps_url")
                    if url:
                        title = f"[{a['name']}]({url})"
                    else:
                        title = a['name']

                    col_evening.info(
                        f"{time_range}**{title}**\n\n{a.get('description','')}"
                    )

                
        except Exception as e:
            st.error(f"❌ Erreur : {e}")


# ==================================================
# PAGE 2 : CHATBOT IA
# ==================================================

if page == "💬 Chatbot IA":
    st.title("💬 Chatbot Touristique – IA Maroc")

    # Initialisation session
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Saisie
    user_msg = st.text_input("Votre message :", placeholder="Posez une question sur une ville, un lieu, un itinéraire...")

    if st.button("Envoyer", use_container_width=True):
        if user_msg.strip():
            payload = {
                "session_id": st.session_state.chat_session_id,
                "message": user_msg,
                "language": "fr"
            }

            try:
                resp = requests.post(f"{API_URL}/chat", json=payload)
                resp.raise_for_status()

                data = resp.json()

                # Mise à jour session ID
                st.session_state.chat_session_id = data["session_id"]

                # Stockage de l'historique
                st.session_state.chat_history.append(("user", user_msg))
                st.session_state.chat_history.append(("assistant", data["answer"]))

            except Exception as e:
                st.error(f"Erreur : {e}")

    # Affichage de l'historique
    st.markdown("### 💬 Historique")

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧍 Vous :** {message}")
        else:
            st.markdown(f"**🤖 Assistant :** {message}")
