import streamlit as st

st.set_page_config(
    page_title="AI Ecosystem Health Monitoring",
    page_icon="🌍",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main{
    background-color:#f4f8fb;
}

.title{
    text-align:center;
    color:#0B3C5D;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:20px;
}

div[data-testid="stTextInput"] input{
    background-color:white;
    color:black;
}

.card{
    background-color:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("<p class='title'>AI Ecosystem Health Monitoring System</p>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Real-Time Environmental Monitoring using Artificial Intelligence</p>", unsafe_allow_html=True)

st.write("")

# ---------- City ----------
city = st.text_input("Enter City Name", placeholder="Example: Hyderabad")

analyze = st.button("Analyze")

st.write("---")

# ---------- Cards ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weather")
    st.write("Temperature : --")
    st.write("Humidity : --")
    st.write("Pressure : --")
    st.write("Wind Speed : --")
    st.write("Weather : --")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Air Quality")
    st.write("AQI : --")
    st.write("PM2.5 : --")
    st.write("PM10 : --")
    st.write("CO : --")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Prediction")
    st.info("Waiting for Analysis...")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

st.subheader("Sustainability Recommendations")

st.success("""
• Recommendations will appear here after AI analysis.
""")