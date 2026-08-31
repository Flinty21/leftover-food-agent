import streamlit as st
from main import app as kitchen_app

st.set_page_config(page_title="Food-Waste Reduction Agent", page_icon="🥗", layout="wide")

st.title("🥗 Autonomous Food-Waste Reduction Multi-Agent System")
st.caption("Powered by LangGraph, Tool Calling (Open Food Facts API) & Local Ollama (Llama 3.2)")

st.sidebar.header("Input Panel")
ingredients_input = st.sidebar.text_area(
    "Available Leftover Ingredients", 
    value="Leftover cooked rice, fresh eggs, onion, bell pepper, gluten-free tamari",
    help="Comma-separated list of ingredients in your fridge."
)

dietary_input = st.sidebar.text_input("Dietary Restrictions", value="Gluten-Free")

if st.sidebar.button("Generate Safe Recipe", type="primary"):
    initial_state = {
        "ingredients": ingredients_input,
        "dietary_restrictions": dietary_input,
        "revision_count": 0
    }
    
    with st.spinner("Executing 4-Agent Pipeline & Querying Food Database..."):
        result = kitchen_app.invoke(initial_state)
    
    st.success("Pipeline Execution Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Food Safety Audit")
        st.warning(result.get("safety_notes", "No safety issues identified."))
        
        st.subheader("🔬 Database & Nutrition Research (Tool Output)")
        st.info(result.get("nutritional_data", "No tool data retrieved."))
        
        st.subheader("Verified Safe Ingredients")
        st.write(result.get("safe_ingredients", "None"))

    with col2:
        st.subheader("👨‍🍳 Executive Chef Recipe")
        st.markdown(result.get("recipe", "No recipe generated."))
        
        st.subheader("🔍 Quality Control Status")
        if result.get("is_approved"):
            st.success("Status: APPROVED ✅")
        else:
            st.error("Status: REVISION / REJECTED ⚠️")