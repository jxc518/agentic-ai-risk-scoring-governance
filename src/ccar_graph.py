
from typing import TypedDict, List, Dict, Any

import pandas as pd

from IPython.display import display, Markdown
from langgraph.graph import StateGraph, END

from ccar_agents import (
    _01_agent_upload,
    _02_agent_DataValidator,
    _03_agent_FeatureEngineer,
    _04_agent_Model_Scorer,
    _05_agent_PerformanceEvaluator,
    _06_agent_conditional_router,
    _07_agent_reporting,
    _08_agent_RAG_governance_monitoring,
    _09_agent_recommendation,
    _10_agent_human_approval,
    _11_agent_DeterministicSQLAgent,
    _12_agent_ExecutiveSummaryAgent
)


##### Step 01: Agent State

class AgentState(TypedDict, total=False):

    # ---------- Agent 1: Upload ----------
    input_file: str
    scoring_vintage: str
    required_columns: List[str]
    numeric_columns: List[str]
    target_col: str

    dataset_id: str
    upload_report: pd.DataFrame
    upload_summary: Dict[str, Any]

    # ---------- Agent 2: Data Validation ----------
    validation_report: pd.DataFrame

    # ---------- Agent 3: Feature Engineering ----------
    feature_dataset_id: str
    feature_summary: Dict[str, Any]

    scored_dataset_id: str
    scoring_report: pd.DataFrame
    scoring_summary: Dict[str, Any]

    performance_report: pd.DataFrame

    # ---------- Agent 4: Monthly Scoring ----------
    model_path: str
    model_x_list: List[str]
    pd_threshold: float
    scoring_dataset_id: str
    scoring_report: pd.DataFrame

    # ---------- Agent 5: Performance Evaluation ----------
    performance_report: pd.DataFrame

    # ---------- Agent 6: Conditional Router ----------
    route_result: Dict[str, Any]
    router_decision: str
    router_reason: str
    router_result: str

    # ---------- Agent 7: Reporting ----------
    technical_report: str

    # ---------- Agent 8: RAG for goverance ----------
    knowledge_base: Any
    _rag_query: str
    _rag_documents: List[Any]
    rag_answer: str

    # ---------- Agent 9: Recommendation ----------
    recommendation_report: pd.DataFrame

    # ---------- Agent 10: Human-in-the-Loop ----------
    approval_status: str
    approval_reason: str

    # ---------- Agent 11: Deterministic SQL ----------
    openai_client: Any
    llm_model_name: str
    semantic_map: Any
    sql_conn: Any
    sql_user_query: str
    sql_query: str
    sql_result: pd.DataFrame

    # ---------- Agent 12: Executive Summary ----------
    executive_summary: str

    # ---------- Engineering Error Handling ----------
    error_agent: str
    error_type: str
    error_message: str
    pipeline_status: str


##### Step 02: Couter_decision(), agent_cleanup(), graph helpers

# ==================================================
# IV. Helper Functions
# ==================================================

def router_decision(state: AgentState):

    decision = state["route_result"]["route_decision"]

    if decision == "REPORTING_AGENT":
        return "approve"

    elif decision in ["HUMAN_REVIEW", "RAG_AGENT", "RECOMMENDATION_AGENT"]:
        return "manual_review"

    elif decision == "STOP":
        return "decline"

    else:
        return "manual_review"

def agent_cleanup(state: AgentState):

    for key in list(state.keys()):
        if key.startswith("_"):
            state.pop(key, None)

    return state

def safe_display_any(obj):
    if obj is None:
        display(Markdown("*Not available.*"))
    elif isinstance(obj, pd.DataFrame):
        display(obj)
    elif isinstance(obj, dict):
        display(pd.DataFrame(list(obj.items()), columns=["Item", "Value"]))
    else:
        display(Markdown(str(obj)))


def print_dict_table(d):
    if d is None:
        print("N/A")
        return

    df = pd.DataFrame(
        list(d.items()),
        columns=["Item", "Value"]
    )

    display(df)


def safe_display_any(obj):
    if obj is None:
        display(Markdown("*Not available.*"))
    elif isinstance(obj, pd.DataFrame):
        display(obj)
    elif isinstance(obj, dict):
        display(pd.DataFrame(list(obj.items()), columns=["Item", "Value"]))
    else:
        display(Markdown(str(obj)))

def display_workflow_result(state):

    display(Markdown("# 🚀 Agentic AI Credit Risk Decisioning Platform\n---"))

    display(Markdown("## 📁 Agent 1 — Data Upload"))
    safe_display_any(state.get("upload_summary"))
    safe_display_any(state.get("upload_report"))

    display(Markdown("## ✅ Agent 2 — Data Validation"))
    safe_display_any(state.get("validation_report"))

    display(Markdown("## ⚙️ Agent 3 — Feature Engineering"))
    safe_display_any(state.get("feature_summary"))

    display(Markdown("## 📈 Agent 4 — Model Scoring"))
    safe_display_any(state.get("scoring_report"))
    safe_display_any(state.get("scoring_summary"))

    display(Markdown("## 📊 Agent 5 — Performance Evaluation"))
    safe_display_any(state.get("performance_report"))

    display(Markdown("## 🚦 Agent 6 — Conditional Router"))
    safe_display_any(state.get("route_result"))

    display(Markdown("## 📝 Agent 7 — Technical Report"))
    safe_display_any(state.get("technical_report"))

    display(Markdown("## 📚 Agent 8 — RAG-Goverance-Monitoring"))
    safe_display_any(state.get("rag_answer"))

    display(Markdown("## 💡 Agent 9 — Recommendation"))
    safe_display_any(state.get("recommendation_report"))

    display(Markdown("## 👤 Agent 10 — Human Approval"))
    approval = pd.DataFrame([{
        "Approval Status": state.get("approval_status", "N/A"),
        "Reason": state.get("approval_reason", "")
    }])
    display(approval)

    display(Markdown("## 🗄️ Agent 11 — SQL Agent"))
    safe_display_any(state.get("sql_result"))

    display(Markdown("## ⭐ Agent 12 — Executive Summary"))
    safe_display_any(state.get("executive_summary"))

    display(Markdown("---\n## ✅ Workflow Completed Successfully"))


##### Step 03: StateGraph construction, add_node(), add_edge(), add_contional_edge(), compile()

pd.reset_option("display.max_colwidth")
pd.reset_option("display.max_columns")
pd.reset_option("display.width")


def build_ccar_graph():

    # 1. Ensure my lang-graph is built first
    workflow = StateGraph(AgentState)

    # 2. Add 12 nodes (Agent 1 through 12)

    workflow.add_node("uploader"                 ,  _01_agent_upload )
    workflow.add_node("data_validator"           ,  _02_agent_DataValidator)
    workflow.add_node("featureengineer"          ,  _03_agent_FeatureEngineer)
    workflow.add_node("scorer"                   ,  _04_agent_Model_Scorer)
    workflow.add_node("perf_evaluator"           ,  _05_agent_PerformanceEvaluator)
    workflow.add_node("router"                   ,  _06_agent_conditional_router)
    # workflow.add_node("debug_before_reporter"    ,  debug_before_reporter)
    workflow.add_node("reporter"                 ,  _07_agent_reporting)
    workflow.add_node("rag_governance_monitoring", _08_agent_RAG_governance_monitoring)
    workflow.add_node("recommendation"           ,  _09_agent_recommendation)
    workflow.add_node("human_approval"           ,  _10_agent_human_approval)
    workflow.add_node("deterministic_sql"        ,  _11_agent_DeterministicSQLAgent)
    workflow.add_node("executives_summary"       ,  _12_agent_ExecutiveSummaryAgent)

    workflow.add_node("cleanup", agent_cleanup)

    workflow.set_entry_point("uploader")

    workflow.add_edge("uploader", "data_validator")
    workflow.add_edge("data_validator", "featureengineer")
    workflow.add_edge("featureengineer", "scorer")
    workflow.add_edge("scorer", "perf_evaluator")
    workflow.add_edge("perf_evaluator", "router")

    workflow.add_conditional_edges(
        "router",
        router_decision,
        {
            # "approve": "reporter",
            # "manual_review": "recommendation",
            "approve": "recommendation",
            "manual_review": "reporter",
            "decline": "executives_summary",
        }
    )

    # approve path
    # workflow.add_edge("debug_before_reporter", "reporter")
    workflow.add_edge("reporter", "rag_governance_monitoring")
    workflow.add_edge("rag_governance_monitoring", "recommendation")

    # manual_review path
    workflow.add_edge("recommendation", "human_approval")

    # continue after review / recommendation
    workflow.add_edge("human_approval", "deterministic_sql")
    workflow.add_edge("deterministic_sql", "executives_summary")

    # decline path
    workflow.add_edge("executives_summary", "cleanup")
    workflow.add_edge("cleanup", END)


    # 4. CRITICAL: Compile the graph to create the 'app' variable
    app = workflow.compile()

    # 5. Here I can invoke the app
    # initial_state = {
    #     # Agent 1
    #     "input_file": pd_input_file,
    #     "scoring_vintage": "201604",

    #     # Agent 2 / 3
    #     "required_columns": x_list_v2,
    #     "numeric_columns": x_list_v3,
    #     "target_col": pd_y,

    #     # Agent 4
    #     "model_path": pd_model_path,
    #     "model_x_list": pd_x_list,
    #     "pd_threshold": pd_threshold,

    #     # Agent 8
    #     "knowledge_base": sample_knowledge_base,

    #     # Agent 11
    #     "openai_client": openai_client,
    #     "llm_model_name": "gpt-4o-mini",
    #     "semantic_map": semantic_map,
    #     "sql_conn": conn,
    #     "sql_user_query": "please let me know the total original balance for HELOC"
    # }

    # result = app.invoke(initial_state)

    # display_workflow_result(result)

    return app




