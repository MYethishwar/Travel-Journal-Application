# agents.py — Vertex AI + Google ADK + LangGraph + LangSmith

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

import vertexai
from vertexai.generative_models import GenerativeModel
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langsmith import traceable

load_dotenv()

# ── Vertex AI Setup ────────────────────────────────────────────────────────────
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-east5"),
)
model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

import time

def llm_call(prompt: str) -> str:
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"Rate limited, waiting 30s... (attempt {attempt+1})")
                time.sleep(30)
            else:
                raise
    raise Exception("Max retries exceeded")

# ── Search Tool ────────────────────────────────────────────────────────────────
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
search_tool = Tool(name="google_search", func=search.run, description="Search Google")

# ── State ──────────────────────────────────────────────────────────────────────
class TravelState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    destination: str
    research_result: str
    plan_result: str
    budget_result: str
    final_summary: str
    next_agent: str

# ── Agent 1: Orchestrator ──────────────────────────────────────────────────────
@traceable(name="Orchestrator Agent")
def orchestrator_agent(state: TravelState) -> TravelState:
    print("\n[ORCHESTRATOR] Analyzing query...")
    prompt = f"""You are an orchestrator for a travel assistant.
User query: "{state['query']}"
Extract destination and decide which agents are needed (researcher, planner, budgeter).
Respond EXACTLY:
DESTINATION: <city/country or unknown>
AGENTS: <comma-separated: researcher, planner, budgeter>
REASON: <one sentence>"""

    text = llm_call(prompt)
    destination = "unknown"
    for line in text.split("\n"):
        if line.startswith("DESTINATION:"):
            destination = line.replace("DESTINATION:", "").strip()

    print(f"[ORCHESTRATOR] → {destination}")
    return {**state, "destination": destination, "next_agent": "researcher",
            "messages": state["messages"] + [AIMessage(content=text)]}

# ── Agent 2: Researcher ────────────────────────────────────────────────────────
@traceable(name="Researcher Agent")
def researcher_agent(state: TravelState) -> TravelState:
    dest = state.get("destination", "unknown")
    print(f"\n[RESEARCHER] Searching {dest}...")
    try:
        search_results = search_tool.run(f"{dest} travel guide top attractions best time to visit")
    except Exception as e:
        search_results = f"Search unavailable: {e}"

    result = llm_call(f"""You are a Travel Researcher.
Destination: {dest}
Search results: {search_results}
Give: top 5 attractions, best time to visit, culture tips, visa info. Max 300 words.""")

    print("[RESEARCHER] Done.")
    return {**state, "research_result": result, "next_agent": "planner",
            "messages": state["messages"] + [AIMessage(content=result)]}

# ── Agent 3: Planner ───────────────────────────────────────────────────────────
@traceable(name="Planner Agent")
def planner_agent(state: TravelState) -> TravelState:
    dest = state.get("destination", "unknown")
    print(f"\n[PLANNER] Planning {dest}...")
    try:
        search_results = search_tool.run(f"{dest} 5 day itinerary popular route")
    except Exception:
        search_results = "Search unavailable"

    result = llm_call(f"""You are a Travel Planner.
Destination: {dest}
Research: {state.get('research_result', '')}
Search: {search_results}
Create a 3-5 day itinerary: day-by-day, morning/afternoon/evening. Max 350 words.""")

    print("[PLANNER] Done.")
    return {**state, "plan_result": result, "next_agent": "budgeter",
            "messages": state["messages"] + [AIMessage(content=result)]}

# ── Agent 4: Budgeter ──────────────────────────────────────────────────────────
@traceable(name="Budgeter Agent")
def budgeter_agent(state: TravelState) -> TravelState:
    dest = state.get("destination", "unknown")
    print(f"\n[BUDGETER] Estimating budget for {dest}...")
    try:
        search_results = search_tool.run(f"{dest} travel cost per day budget 2024")
    except Exception:
        search_results = "Search unavailable"

    result = llm_call(f"""You are a Travel Budget Agent.
Destination: {dest}
Search: {search_results}
Give: flights from India, accommodation (budget/mid/luxury), food/day, transport/day, activities. TOTAL range. Max 250 words.""")

    print("[BUDGETER] Done.")
    return {**state, "budget_result": result, "next_agent": "summarizer",
            "messages": state["messages"] + [AIMessage(content=result)]}

# ── Agent 5: Summarizer ────────────────────────────────────────────────────────
@traceable(name="Summarizer Agent")
def summarizer_agent(state: TravelState) -> TravelState:
    print(f"\n[SUMMARIZER] Compiling final guide...")

    result = llm_call(f"""You are a Travel Summary Agent.
Combine these 3 reports into one clean travel guide for {state.get('destination')}.

RESEARCH: {state.get('research_result')}
ITINERARY: {state.get('plan_result')}
BUDGET: {state.get('budget_result')}

Use clear headings. Blog post style. No repetition. Max 500 words.""")

    print("[SUMMARIZER] Done!")
    return {**state, "final_summary": result, "next_agent": "end",
            "messages": state["messages"] + [AIMessage(content=result)]}

# ── LangGraph ──────────────────────────────────────────────────────────────────
def build_travel_graph():
    g = StateGraph(TravelState)
    g.add_node("orchestrator", orchestrator_agent)
    g.add_node("researcher",   researcher_agent)
    g.add_node("planner",      planner_agent)
    g.add_node("budgeter",     budgeter_agent)
    g.add_node("summarizer",   summarizer_agent)
    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", "researcher")
    g.add_edge("researcher",   "planner")
    g.add_edge("planner",      "budgeter")
    g.add_edge("budgeter",     "summarizer")
    g.add_edge("summarizer",   END)
    return g.compile()

@traceable(name="Travel Agent Workflow")
def run_travel_agents(user_query: str) -> dict:
    graph = build_travel_graph()
    final_state = graph.invoke({
        "messages": [HumanMessage(content=user_query)],
        "query": user_query,
        "destination": "",
        "research_result": "",
        "plan_result": "",
        "budget_result": "",
        "final_summary": "",
        "next_agent": "orchestrator",
    })
    return {
        "query":       user_query,
        "destination": final_state["destination"],
        "research":    final_state["research_result"],
        "itinerary":   final_state["plan_result"],
        "budget":      final_state["budget_result"],
        "summary":     final_state["final_summary"],
    }