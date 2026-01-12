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
page = st.sidebar.radio("Go to:", ["🧭 Itinerary Generator", "💬 AI Chatbot"])

st.sidebar.markdown("---")
st.sidebar.caption("Morocco Smart Guide – Tourism AI 🇲🇦")


# ==================================================
# PAGE 1: ITINERARY GENERATOR
# ==================================================

if page == "🧭 Itinerary Generator":
    st.title("🧭 Generate a personalized itinerary")

    city = st.text_input("Main city", placeholder="e.g.: Marrakech")
    duration = st.number_input("Duration (days)", min_value=1, value=3)
    budget = st.selectbox("Budget", ["low", "medium", "high"], index=1)
    interests = st.text_input("Interests", placeholder="culture, gastronomy, shopping …")
    constraints = st.text_area("Constraints", placeholder="optional")
    language = st.selectbox("Language", ["en", "fr"], index=0)

    if st.button("Generate itinerary", use_container_width=True):
        payload = {
            "city": city or None,
            "duration_days": int(duration),
            "budget": budget,
            "interests": [s.strip() for s in interests.split(",") if s.strip()],
            "constraints": constraints,
            "language": language,
        }

        try:
            with st.spinner("Generating itinerary…"):
                resp = requests.post(f"{API_URL}/itinerary", json=payload)
                resp.raise_for_status()

            itinerary = resp.json()
            st.success("Itinerary successfully generated!")

            st.subheader(f"📍 City: **{itinerary['city']}**")
            st.markdown(f"**Duration:** {itinerary['duration_days']} days")

            for day in itinerary["days"]:
                st.markdown(f"## 🗓️ Day {day['day_number']}")

                col_morning, col_afternoon, col_evening = st.columns(3)

                col_morning.markdown("### 🌅 Morning")
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

                col_afternoon.markdown("### 🌞 Afternoon")
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

                col_evening.markdown("### 🌙 Evening")
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
            st.error(f"❌ Error: {e}")


# ==================================================
# PAGE 2: AI CHATBOT
# ==================================================

if page == "💬 AI Chatbot":
    st.title("💬 Tourism Chatbot – Morocco AI")

    # Session initialization
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input
    user_msg = st.text_input(
        "Your message:",
        placeholder="Ask a question about a city, a place, an itinerary..."
    )

    if st.button("Send", use_container_width=True):
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

                # Update session ID
                st.session_state.chat_session_id = data["session_id"]

                # Store history
                st.session_state.chat_history.append(("user", user_msg))
                st.session_state.chat_history.append(("assistant", data["answer"]))

            except Exception as e:
                st.error(f"Error: {e}")

    # Display history
    st.markdown("### 💬 History")

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧍 You:** {message}")
        else:
            st.markdown(f"**🤖 Assistant:** {message}")
