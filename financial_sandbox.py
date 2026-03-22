import os
import sys
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional

# Make predictor.py importable without installing the deployment package
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "credit-risk-prediction-project/deployment"))
from predictor import CreditRiskPredictor

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg

# --- CONFIGURATION ---
_API_KEY_MISSING = not os.environ.get("OPENAI_API_KEY")
if _API_KEY_MISSING:
    print("WARNING: OPENAI_API_KEY not set. Chatbot will be disabled.")

# --- FEATURE DEFINITIONS ---
# Ordered dict matching the Gradio /predict_loan parameter order exactly.
FEATURE_DEFAULTS: Dict[str, Any] = {
    'loan_amnt': 15000,
    'int_rate': 12.5,
    'grade': 'C',
    'emp_length': '5 years',
    'annual_inc': 75000,
    'dti': 18.5,
    'revol_util': 45,
    'delinq_2yrs': 0,
    'inq_last_6mths': 2,
    'open_acc': 8,
    'total_acc': 25,
    'revol_bal': 5000,
    'total_bc_limit': 20000,
    'total_bal_ex_mort': 30000,
    'avg_cur_bal': 2500,
    'mo_sin_old_il_acct': 60,
    'mo_sin_old_rev_tl_op': 48,
    'mo_sin_rcnt_rev_tl_op': 12,
    'mths_since_recent_bc': 6,
    'mths_since_recent_inq': 3,
    'pct_tl_nvr_dlq': 95,
    'last_fico_range_low': 680,
    'last_fico_range_high': 684,
    'years_since_earliest_cr': 10,
    'addr_state': 'CA',
    'home_ownership': 'RENT',
    'purpose': 'debt_consolidation',
    'verification_status': 'Verified',
    'title': 'Debt consolidation loan',
}

# --- PREDICTOR (module-level, loaded once) ---
_predictor = CreditRiskPredictor(
    os.path.join(os.path.dirname(__file__),
                 "credit-risk-prediction-project/deployment/model_artifacts")
)

def _profile_to_predictor_input(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Convert our internal profile dict to the format predictor.py expects."""
    d = dict(profile)
    # predictor.py expects revol_util as a string like "45%" (mirrors app.py line 102)
    if not str(d['revol_util']).endswith('%'):
        d['revol_util'] = f"{d['revol_util']}%"
    # Explicit type casts that app.py applies before calling predictor.predict()
    for k in ('loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_bal', 'total_bc_limit',
              'total_bal_ex_mort', 'avg_cur_bal', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
              'mo_sin_rcnt_rev_tl_op', 'mths_since_recent_bc', 'mths_since_recent_inq',
              'pct_tl_nvr_dlq', 'last_fico_range_low', 'last_fico_range_high',
              'years_since_earliest_cr'):
        d[k] = float(d[k])
    for k in ('delinq_2yrs', 'inq_last_6mths', 'open_acc', 'total_acc'):
        d[k] = int(d[k])
    return d

# --- STATE ---
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_profile: Dict[str, Any]

# --- TOOLS ---

@tool
def predict_credit_risk(
    updates: Dict[str, Any],
    profile: Annotated[Dict[str, Any], InjectedToolArg],
) -> str:
    """
    Predict loan approval based on the current profile, with optional feature updates.

    Call this to answer questions like:
    - "What is my current risk?" → pass updates={}
    - "If my income is $90k and DTI is 10?" → pass updates={"annual_inc": 90000, "dti": 10}

    Args:
        updates: Features to change before predicting. Use exact names and valid values:
            Categorical — must use these exact strings:
            - grade: "A", "B", "C", "D", "E", "F", "G"  (A=best, G=worst)
            - home_ownership: "RENT", "MORTGAGE", "OWN", "OTHER"
            - purpose: "debt_consolidation", "credit_card", "home_improvement",
                       "major_purchase", "medical", "car", "wedding"
            - verification_status: "Verified", "Source Verified", "Not Verified"
            - addr_state: 2-letter US state code e.g. "CA", "NY", "TX"
            - emp_length: "< 1 year", "1 year", "2 years", ..., "10+ years"
            Numeric — pass as numbers (not strings):
            - loan_amnt, int_rate, annual_inc, dti, revol_util, delinq_2yrs,
              inq_last_6mths, open_acc, total_acc, revol_bal, total_bc_limit,
              total_bal_ex_mort, avg_cur_bal, last_fico_range_low, last_fico_range_high,
              years_since_earliest_cr, pct_tl_nvr_dlq
    """
    working = dict(profile)
    applied = []
    unknown = []
    for k, v in updates.items():
        if k in working:
            working[k] = v
            applied.append(f"{k}={v}")
        else:
            unknown.append(k)

    try:
        result = _predictor.predict(_profile_to_predictor_input(working))
        if not result['success']:
            return json.dumps({"error": result.get('error', 'Prediction failed')})

        output = {
            "decision": result['decision'],
            "default_probability": f"{result['default_probability']:.2%}",
            "risk_level": result['risk_level'],
            "threshold": f"{result['optimal_threshold']:.2%}",
            "updates_applied": applied,
        }
        if unknown:
            output["unknown_features_ignored"] = unknown
        return json.dumps(output)

    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {e}"})


@tool
def get_current_profile(
    profile: Annotated[Dict[str, Any], InjectedToolArg],
) -> str:
    """
    Return all current profile values that will be used for the next prediction.
    Call this when the user asks about their current settings, values, or profile.
    """
    return json.dumps(profile, indent=2)


@tool
def reset_profile(
    profile: Annotated[Dict[str, Any], InjectedToolArg],
) -> str:
    """
    Reset all profile values back to the original defaults.
    Call this when the user asks to reset, start over, or go back to baseline.
    """
    # Actual state reset is handled in tool_node; this returns the confirmation payload.
    return json.dumps({"status": "reset", "new_profile": FEATURE_DEFAULTS})


# --- LLM (module-level, instantiated once) ---
# Uses OpenRouter — base URL and model prefix required
_llm = None
_llm_with_tools = None
if not _API_KEY_MISSING:
    _llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    _tools = [predict_credit_risk, get_current_profile, reset_profile]
    _llm_with_tools = _llm.bind_tools(_tools)


def _make_system_prompt(profile: Dict[str, Any]) -> str:
    key_fields = ["loan_amnt", "int_rate", "grade", "annual_inc", "dti",
                  "last_fico_range_low", "home_ownership", "purpose"]
    summary = {k: profile[k] for k in key_fields}
    return f"""You are a helpful Financial Credit Risk Assistant with access to a What-If simulation tool.

Current profile summary:
{json.dumps(summary, indent=2)}

Available tools:
- predict_credit_risk: Run a prediction, optionally changing features first
- get_current_profile: Show the user all current profile values
- reset_profile: Revert all values to the original defaults

Always summarize results clearly. When comparing scenarios, highlight the change in probability.
Use exact feature names and valid categorical values as documented in predict_credit_risk.
Important: the model was trained on Lending Club data (2013-2014) with loan amounts up to $40,000. If the user requests inputs far outside this range (e.g. loan_amnt > $40k, income implausibly low for the loan size), warn them that the prediction is unreliable."""


# --- AGENT NODES ---

def chatbot(state: AgentState) -> dict:
    profile = state.get("current_profile", dict(FEATURE_DEFAULTS))
    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    messages = [SystemMessage(content=_make_system_prompt(profile))] + non_system
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    profile = dict(state.get("current_profile", FEATURE_DEFAULTS))
    outputs = []

    for tc in last_message.tool_calls:
        name = tc["name"]

        if name == "predict_credit_risk":
            raw = tc["args"]
            # LLMs sometimes pass features directly as top-level args instead of nested
            # inside the 'updates' dict — handle both calling patterns.
            if raw.get("updates"):
                updates = raw["updates"]
            else:
                updates = {k: v for k, v in raw.items() if k in profile}
            # Apply updates to the persisted profile
            for k, v in updates.items():
                if k in profile:
                    profile[k] = v
            result = predict_credit_risk.invoke({"updates": updates, "profile": profile})

        elif name == "get_current_profile":
            result = get_current_profile.invoke({"profile": profile})

        elif name == "reset_profile":
            profile = dict(FEATURE_DEFAULTS)
            result = reset_profile.invoke({"profile": profile})

        else:
            result = json.dumps({"error": f"Unknown tool: {name}"})

        outputs.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=name))

    return {"messages": outputs, "current_profile": profile}


# --- GRAPH ---
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", chatbot)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    def should_continue(state: AgentState):
        return "tools" if state["messages"][-1].tool_calls else END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# --- MAIN LOOP ---
def run_interactive_session():
    if _API_KEY_MISSING:
        print("Cannot start session: OPENAI_API_KEY not set.")
        return
    graph = build_graph()

    state: AgentState = {
        "messages": [],
        "current_profile": dict(FEATURE_DEFAULTS),
    }

    print("Financial Sandbox Agent Ready! (Type 'quit' to exit)")
    print("------------------------------------------------------")
    print("Baseline: Income=$75k, Loan=$15k, DTI=18.5, Grade=C, FICO=680-684")

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

        print("Thinking...", end="", flush=True)

        try:
            final = graph.invoke(state)
            state["messages"] = final["messages"]
            state["current_profile"] = final["current_profile"]

            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n\nAgent: {msg.content}")
                    break

        except Exception as e:
            print(f"\nError: {e}")
            # Continue — don't kill the session on transient errors


if __name__ == "__main__":
    run_interactive_session()
