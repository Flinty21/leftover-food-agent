from typing import TypedDict
import requests
from urllib.parse import quote_plus
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# 1. Initialize Local Ollama Model (3B Model for strong reasoning)
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Define External Tool: Open Food Facts API Tool
@tool
def search_food_database(food_item: str) -> str:
    """
    Search the global Open Food Facts database for nutritional info, allergens, and ingredients.
    """
    try:
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={quote_plus(food_item)}&search_simple=1&action=process&json=1"
        # Adding a User-Agent header bypasses bot-blockers
        headers = {"User-Agent": "LeftoverFoodAgent/1.0 (StudentProject)"} 
        
        res = requests.get(url, headers=headers, timeout=5)
        
        # Guard clause to ensure we actually got a successful response before parsing JSON
        if res.status_code != 200:
            return f"API blocked the request (Status {res.status_code})"
            
        data = res.json()
        products = data.get("products", [])
        
        if products:
            p = products[0]
            name = p.get("product_name", food_item)
            nutriscore = p.get("nutriscore_grade", "unknown").upper()
            allergens = p.get("allergens", "none listed")
            allergens = allergens if allergens else "none listed"
            return f"Item: {name} | Nutri-Score (Health Grade): {nutriscore} | Allergens: {allergens}"
            
        return f"No database records found for {food_item}, treating as raw fresh ingredient."
    except Exception as e:
        return f"Database lookup warning: {e}"

# 3. Define Shared System State Across Agents
class KitchenState(TypedDict):
    ingredients: str
    dietary_restrictions: str
    safe_ingredients: str
    safety_notes: str
    nutritional_data: str
    recipe: str
    qc_feedback: str
    is_approved: bool
    revision_count: int

# 4. Agent 1: Food Safety Agent Node
def food_safety_agent(state: KitchenState) -> KitchenState:
    print("\n[🛡️ Food Safety Agent] Auditing ingredients for safety and freshness...")
    prompt = f"""You are a Food Safety Specialist. 
Analyze these leftover ingredients: {state['ingredients']}

Identify any spoiled, unsafe, or expired items that should NOT be consumed.
List the verified SAFE ingredients for cooking, and note any discarded items.

Respond in this exact format:
SAFE INGREDIENTS: <comma separated list of safe items>
SAFETY NOTES: <brief explanation of discarded or flagged items>
"""
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    safe_part = response
    notes_part = "None"
    if "SAFETY NOTES:" in response:
        parts = response.split("SAFETY NOTES:")
        safe_part = parts[0].replace("SAFE INGREDIENTS:", "").strip()
        notes_part = parts[1].strip()
    else:
        safe_part = response.replace("SAFE INGREDIENTS:", "").strip()
    
    return {
        "safe_ingredients": safe_part,
        "safety_notes": notes_part
    }

# 5. Agent 2: Nutritional Research Agent (Utilizes Tool)
def nutrition_research_agent(state: KitchenState) -> KitchenState:
    print("\n[🔬 Nutritional Research Agent] Querying Global Food Database via API Tool...")
    items = state["safe_ingredients"].split(",")
    research_results = []
    
    for item in items[:3]: # Query top ingredients using our tool
        item_clean = item.strip()
        if item_clean:
            tool_output = search_food_database.invoke({"food_item": item_clean})
            research_results.append(tool_output)
            
    research_summary = "\n".join(research_results)
    return {"nutritional_data": research_summary}

# 6. Agent 3: Executive Chef Node
def chef_agent(state: KitchenState) -> KitchenState:
    print("\n[👨‍🍳 Executive Chef Agent] Drafting sustainable recipe...")
    feedback_context = f"\nPrevious QC Feedback to fix: {state.get('qc_feedback', '')}" if state.get('qc_feedback') else ""
    
    prompt = f"""You are an Executive Chef specializing in zero-waste cooking.
Create a recipe using ONLY these verified safe ingredients: {state['safe_ingredients']}
Nutritional & Allergen Insights from Database: {state['nutritional_data']}
Dietary Restrictions: {state['dietary_restrictions']}{feedback_context}

Provide a recipe with:
1. Recipe Name
2. Ingredients Used 
3. Step-by-Step Instructions
"""
    recipe = llm.invoke([HumanMessage(content=prompt)]).content
    return {"recipe": recipe}

# 7. Agent 4: Quality Control Agent Node
def quality_control_agent(state: KitchenState) -> KitchenState:
    print("\n[🔍 Quality Control Agent] Auditing recipe compliance...")
    prompt = f"""You are a Quality Control Auditor in a commercial kitchen.
Review this generated recipe:
---
{state['recipe']}
---

Strict Criteria:
1. Does it respect these dietary restrictions: {state['dietary_restrictions']}?
2. Are all ingredients derived from the safe list: {state['safe_ingredients']}?

Respond in this exact format:
STATUS: APPROVED or REJECTED
FEEDBACK: <If rejected, explain what to fix. If approved, write None>
"""
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    is_approved = "STATUS: APPROVED" in response.upper() or "APPROVED" in response.split("\n")[0].upper()
    rev_count = state.get("revision_count", 0) + 1
    
    return {
        "is_approved": is_approved,
        "qc_feedback": response,
        "revision_count": rev_count
    }

# 8. Routing Logic (Conditional Edge)
def route_qc(state: KitchenState) -> str:
    if state["is_approved"] or state.get("revision_count", 0) >= 2:
        return "end"
    return "chef"

# 9. Build LangGraph Workflow
workflow = StateGraph(KitchenState)

# Add Nodes
workflow.add_node("food_safety", food_safety_agent)
workflow.add_node("nutrition_research", nutrition_research_agent)
workflow.add_node("chef", chef_agent)
workflow.add_node("qc", quality_control_agent)

# Add Edges
workflow.add_edge(START, "food_safety")
workflow.add_edge("food_safety", "nutrition_research")
workflow.add_edge("nutrition_research", "chef")
workflow.add_edge("chef", "qc")

# Add Conditional Loop back to Chef if QC fails
workflow.add_conditional_edges(
    "qc",
    route_qc,
    {
        "end": END,
        "chef": "chef"
    }
)

app = workflow.compile()