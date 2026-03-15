import os
import json
from typing import Dict, Any, List, TypedDict, Annotated

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from predictor import CreditRiskPredictor

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI


st.set_page_config(
    page_title="Credit Risk Predictor + LLM Sandbox",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def get_predictor() -> CreditRiskPredictor:
    return CreditRiskPredictor("model_artifacts")


predictor = get_predictor()

FEATURE_INFO = {
    "loan_amnt": "Total amount of the loan applied for",
    "int_rate": "Interest rate on the loan",
    "grade": "LC assigned loan grade (A=best, G=worst)",
    "emp_length": "Employment length in years",
    "annual_inc": "Self-reported annual income",
    "dti": "Debt-to-income ratio",
    "revol_util": "Revolving line utilization rate",
    "delinq_2yrs": "Number of delinquencies in past 2 years",
    "inq_last_6mths": "Number of credit inquiries in past 6 months",
    "open_acc": "Number of open credit lines",
    "total_acc": "Total number of credit lines",
    "revol_bal": "Total credit revolving balance",
    "total_bc_limit": "Total bankcard limit",
    "total_bal_ex_mort": "Total balance excluding mortgage",
    "avg_cur_bal": "Average current balance",
    "mo_sin_old_il_acct": "Months since oldest installment account opened",
    "mo_sin_old_rev_tl_op": "Months since oldest revolving account opened",
    "mo_sin_rcnt_rev_tl_op": "Months since most recent revolving account opened",
    "mths_since_recent_bc": "Months since most recent bankcard account opened",
    "mths_since_recent_inq": "Months since most recent inquiry",
    "pct_tl_nvr_dlq": "Percent of trades never delinquent",
    "last_fico_range_low": "Lower bound of the last FICO range",
    "last_fico_range_high": "Upper bound of the last FICO range",
    "years_since_earliest_cr": "Years since earliest credit line opened",
    "addr_state": "State of the borrower (2-letter code)",
    "home_ownership": "Home ownership status",
    "purpose": "Purpose of the loan",
    "verification_status": "Income verification status",
    "title": "Loan title/description",
}

FEATURE_DEFAULTS: Dict[str, Any] = {
    "loan_amnt": 15000,
    "int_rate": 12.5,
    "grade": "C",
    "emp_length": "5 years",
    "annual_inc": 75000,
    "dti": 18.5,
    "revol_util": 45,
    "delinq_2yrs": 0,
    "inq_last_6mths": 2,
    "open_acc": 8,
    "total_acc": 25,
    "revol_bal": 5000,
    "total_bc_limit": 20000,
    "total_bal_ex_mort": 30000,
    "avg_cur_bal": 2500,
    "mo_sin_old_il_acct": 60,
    "mo_sin_old_rev_tl_op": 48,
    "mo_sin_rcnt_rev_tl_op": 12,
    "mths_since_recent_bc": 6,
    "mths_since_recent_inq": 3,
    "pct_tl_nvr_dlq": 95,
    "last_fico_range_low": 680,
    "last_fico_range_high": 684,
    "years_since_earliest_cr": 10,
    "addr_state": "CA",
    "home_ownership": "RENT",
    "purpose": "debt_consolidation",
    "verification_status": "Verified",
    "title": "Debt consolidation loan",
}

EXAMPLES = {
    "Low Risk": {
        "loan_amnt": 10000, "int_rate": 8.5, "grade": "A", "emp_length": "10+ years",
        "annual_inc": 120000, "dti": 12.0, "revol_util": 30, "delinq_2yrs": 0,
        "inq_last_6mths": 1, "open_acc": 5, "total_acc": 20, "revol_bal": 3000,
        "total_bc_limit": 15000, "total_bal_ex_mort": 25000, "avg_cur_bal": 3000,
        "mo_sin_old_il_acct": 120, "mo_sin_old_rev_tl_op": 96, "mo_sin_rcnt_rev_tl_op": 24,
        "mths_since_recent_bc": 12, "mths_since_recent_inq": 6, "pct_tl_nvr_dlq": 98,
        "last_fico_range_low": 720, "last_fico_range_high": 724, "years_since_earliest_cr": 15,
        "addr_state": "CA", "home_ownership": "OWN", "purpose": "debt_consolidation",
        "verification_status": "Verified", "title": "Debt consolidation"
    },
    "High Risk": {
        "loan_amnt": 35000, "int_rate": 25.0, "grade": "F", "emp_length": "< 1 year",
        "annual_inc": 30000, "dti": 35.0, "revol_util": 95, "delinq_2yrs": 3,
        "inq_last_6mths": 8, "open_acc": 15, "total_acc": 40, "revol_bal": 20000,
        "total_bc_limit": 5000, "total_bal_ex_mort": 10000, "avg_cur_bal": 1000,
        "mo_sin_old_il_acct": 6, "mo_sin_old_rev_tl_op": 12, "mo_sin_rcnt_rev_tl_op": 1,
        "mths_since_recent_bc": 1, "mths_since_recent_inq": 1, "pct_tl_nvr_dlq": 60,
        "last_fico_range_low": 580, "last_fico_range_high": 590, "years_since_earliest_cr": 2,
        "addr_state": "NV", "home_ownership": "RENT", "purpose": "credit_card",
        "verification_status": "Not Verified", "title": "Credit card payoff"
    },
    "Borderline": {
        "loan_amnt": 20000, "int_rate": 15.0, "grade": "D", "emp_length": "3 years",
        "annual_inc": 55000, "dti": 22.0, "revol_util": 75, "delinq_2yrs": 1,
        "inq_last_6mths": 4, "open_acc": 10, "total_acc": 30, "revol_bal": 10000,
        "total_bc_limit": 10000, "total_bal_ex_mort": 20000, "avg_cur_bal": 2000,
        "mo_sin_old_il_acct": 36, "mo_sin_old_rev_tl_op": 48, "mo_sin_rcnt_rev_tl_op": 6,
        "mths_since_recent_bc": 6, "mths_since_recent_inq": 3, "pct_tl_nvr_dlq": 85,
        "last_fico_range_low": 650, "last_fico_range_high": 660, "years_since_earliest_cr": 5,
        "addr_state": "TX", "home_ownership": "MORTGAGE", "purpose": "home_improvement",
        "verification_status": "Source Verified", "title": "Home renovation loan"
    }
}


def ensure_profile_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = dict(FEATURE_DEFAULTS)
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_profile" not in st.session_state:
        st.session_state.agent_profile = dict(FEATURE_DEFAULTS)


ensure_profile_state()


def create_visualization(default_prob: float, threshold: float = 0.28):
    fig, ax = plt.subplots(figsize=(8, 2))
    x = np.linspace(0, 1, 100)
    colors = plt.cm.RdYlGn_r(x)
    for i in range(len(x) - 1):
        ax.fill_between([x[i], x[i + 1]], 0, 1, color=colors[i], alpha=0.7)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.0%})")
    ax.plot(default_prob, 0.5, "ro", markersize=12, label=f"Prediction ({default_prob:.1%})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Default Probability")
    ax.set_title("Risk Assessment")
    ax.legend(loc="upper right")
    ax.set_yticks([])
    plt.tight_layout()
    return fig


def profile_to_predictor_input(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loan_amnt": float(profile["loan_amnt"]),
        "int_rate": float(profile["int_rate"]),
        "grade": str(profile["grade"]),
        "emp_length": str(profile["emp_length"]),
        "annual_inc": float(profile["annual_inc"]),
        "dti": float(profile["dti"]),
        "revol_util": f"{float(profile['revol_util'])}%",
        "delinq_2yrs": int(profile["delinq_2yrs"]),
        "inq_last_6mths": int(profile["inq_last_6mths"]),
        "open_acc": int(profile["open_acc"]),
        "total_acc": int(profile["total_acc"]),
        "revol_bal": float(profile["revol_bal"]),
        "total_bc_limit": float(profile["total_bc_limit"]),
        "total_bal_ex_mort": float(profile["total_bal_ex_mort"]),
        "avg_cur_bal": float(profile["avg_cur_bal"]),
        "mo_sin_old_il_acct": float(profile["mo_sin_old_il_acct"]),
        "mo_sin_old_rev_tl_op": float(profile["mo_sin_old_rev_tl_op"]),
        "mo_sin_rcnt_rev_tl_op": float(profile["mo_sin_rcnt_rev_tl_op"]),
        "mths_since_recent_bc": float(profile["mths_since_recent_bc"]),
        "mths_since_recent_inq": float(profile["mths_since_recent_inq"]),
        "pct_tl_nvr_dlq": float(profile["pct_tl_nvr_dlq"]) / 100.0,
        "last_fico_range_low": float(profile["last_fico_range_low"]),
        "last_fico_range_high": float(profile["last_fico_range_high"]),
        "years_since_earliest_cr": float(profile["years_since_earliest_cr"]),
        "addr_state": str(profile["addr_state"]),
        "home_ownership": str(profile["home_ownership"]),
        "purpose": str(profile["purpose"]),
        "verification_status": str(profile["verification_status"]),
        "title": str(profile["title"]),
    }


def run_prediction(profile: Dict[str, Any]) -> Dict[str, Any]:
    return predictor.predict(profile_to_predictor_input(profile))


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_profile: Dict[str, Any]


def get_llm():
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    # Prefer OpenRouter if provided
    if openrouter_key:
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Credit Risk Predictor",
            },
        )

    # Fallback: direct OpenAI
    if openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_key,
        )

    return None

def _make_system_prompt(profile: Dict[str, Any]) -> str:
    key_fields = ["loan_amnt", "int_rate", "grade", "annual_inc", "dti", "last_fico_range_low", "home_ownership", "purpose"]
    summary = {k: profile[k] for k in key_fields}
    return f'''You are a helpful Financial Credit Risk Assistant with access to a what-if simulation tool.

Current profile summary:
{json.dumps(summary, indent=2)}

Available tools:
- predict_credit_risk: run a prediction, optionally changing features first
- get_current_profile: show all current profile values
- reset_profile: revert all values to the original defaults

Always summarize results clearly.
When comparing scenarios, highlight the change in probability.
Use exact feature names and valid categorical values as documented in predict_credit_risk.
Important: the model was trained on Lending Club data (2013-2014) with loan amounts up to $40,000.
If the user requests values far outside this range, warn that the prediction may be unreliable.
'''


@tool
def predict_credit_risk(updates: Dict[str, Any], profile: Annotated[Dict[str, Any], InjectedToolArg]) -> str:
    """Run a credit risk prediction using the current profile, with optional field updates."""
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
        result = run_prediction(working)
        if not result["success"]:
            return json.dumps({"error": result.get("error", "Prediction failed")})
        output = {
            "decision": result["decision"],
            "default_probability": f"{result['default_probability']:.2%}",
            "risk_level": result["risk_level"],
            "threshold": f"{result['optimal_threshold']:.2%}",
            "updates_applied": applied,
        }
        if unknown:
            output["unknown_features_ignored"] = unknown
        return json.dumps(output)
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {e}"})


@tool
def get_current_profile(profile: Annotated[Dict[str, Any], InjectedToolArg]) -> str:
    """Return the current borrower profile used by the chatbot."""
    return json.dumps(profile, indent=2)


@tool
def reset_profile(profile: Annotated[Dict[str, Any], InjectedToolArg]) -> str:
    """Reset the chatbot profile back to the default values."""
    return json.dumps({"status": "reset", "new_profile": FEATURE_DEFAULTS})

@st.cache_resource
def build_graph():
    llm = get_llm()
    if llm is None:
        return None
    tools = [predict_credit_risk, get_current_profile, reset_profile]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: AgentState) -> dict:
        profile = state.get("current_profile", dict(FEATURE_DEFAULTS))
        non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        messages = [SystemMessage(content=_make_system_prompt(profile))] + non_system
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        profile = dict(state.get("current_profile", FEATURE_DEFAULTS))
        outputs = []

        for tc in last_message.tool_calls:
            name = tc["name"]
            if name == "predict_credit_risk":
                raw = tc["args"]
                if raw.get("updates"):
                    updates = raw["updates"]
                else:
                    updates = {k: v for k, v in raw.items() if k in profile}
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

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", chatbot)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    def should_continue(state: AgentState):
        return "tools" if state["messages"][-1].tool_calls else END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def ask_agent(user_input: str) -> str:
    graph = build_graph()
    if graph is None:
        return "OpenRouter API key is not set. Please export OPENROUTER_API_KEY (or OPENAI_API_KEY) before launching Streamlit."

    state: AgentState = {
        "messages": st.session_state.chat_messages[:],
        "current_profile": dict(st.session_state.agent_profile),
    }
    state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    try:
        final = graph.invoke(state)
    except Exception as e:
        return f"LLM request failed: {e}"

    st.session_state.chat_messages = final["messages"]
    st.session_state.agent_profile = final["current_profile"]

    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "No response was returned by the model."


st.title("🏦 Credit Risk Predictor + LLM Sandbox")
st.caption("The left panel is the credit risk assessment form. The right panel is an OpenRouter-powered chat sandbox.")

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Loan Application")
    col_a, col_b = st.columns(2)

    with col_a:
        loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, int(st.session_state.profile["loan_amnt"]), 500, help=FEATURE_INFO["loan_amnt"])
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, float(st.session_state.profile["int_rate"]), 0.1, help=FEATURE_INFO["int_rate"])
        grade = st.radio("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=["A", "B", "C", "D", "E", "F", "G"].index(st.session_state.profile["grade"]), horizontal=True, help=FEATURE_INFO["grade"])
        emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"], index=["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"].index(st.session_state.profile["emp_length"]), help=FEATURE_INFO["emp_length"])
        annual_inc = st.slider("Annual Income ($)", 20000, 1000000, int(st.session_state.profile["annual_inc"]), 1000, help=FEATURE_INFO["annual_inc"])
        dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, float(st.session_state.profile["dti"]), 0.1, help=FEATURE_INFO["dti"])

    with col_b:
        revol_util = st.slider("Credit Utilization (%)", 0, 100, int(st.session_state.profile["revol_util"]), 1, help=FEATURE_INFO["revol_util"])
        delinq_2yrs = st.slider("Delinquencies (last 2 years)", 0, 10, int(st.session_state.profile["delinq_2yrs"]), 1, help=FEATURE_INFO["delinq_2yrs"])
        inq_last_6mths = st.slider("Credit Inquiries (last 6 months)", 0, 10, int(st.session_state.profile["inq_last_6mths"]), 1, help=FEATURE_INFO["inq_last_6mths"])
        open_acc = st.slider("Open Credit Lines", 0, 50, int(st.session_state.profile["open_acc"]), 1, help=FEATURE_INFO["open_acc"])
        total_acc = st.slider("Total Credit Lines", 0, 100, int(st.session_state.profile["total_acc"]), 1, help=FEATURE_INFO["total_acc"])

    with st.expander("Advanced Features (Optional)", expanded=False):
        adv1, adv2 = st.columns(2)
        with adv1:
            revol_bal = st.slider("Revolving Balance ($)", 0, 100000, int(st.session_state.profile["revol_bal"]), 1000, help=FEATURE_INFO["revol_bal"])
            total_bc_limit = st.slider("Total Bankcard Limit ($)", 0, 100000, int(st.session_state.profile["total_bc_limit"]), 1000, help=FEATURE_INFO["total_bc_limit"])
            total_bal_ex_mort = st.slider("Total Balance Excl. Mortgage ($)", 0, 200000, int(st.session_state.profile["total_bal_ex_mort"]), 1000, help=FEATURE_INFO["total_bal_ex_mort"])
            avg_cur_bal = st.slider("Average Current Balance ($)", 0, 50000, int(st.session_state.profile["avg_cur_bal"]), 100, help=FEATURE_INFO["avg_cur_bal"])
            mo_sin_old_il_acct = st.slider("Months since oldest installment account", 0, 300, int(st.session_state.profile["mo_sin_old_il_acct"]), 1, help=FEATURE_INFO["mo_sin_old_il_acct"])
            mo_sin_old_rev_tl_op = st.slider("Months since oldest revolving account", 0, 300, int(st.session_state.profile["mo_sin_old_rev_tl_op"]), 1, help=FEATURE_INFO["mo_sin_old_rev_tl_op"])
            mo_sin_rcnt_rev_tl_op = st.slider("Months since newest revolving account", 0, 300, int(st.session_state.profile["mo_sin_rcnt_rev_tl_op"]), 1, help=FEATURE_INFO["mo_sin_rcnt_rev_tl_op"])
        with adv2:
            mths_since_recent_bc = st.slider("Months since newest bankcard", 0, 120, int(st.session_state.profile["mths_since_recent_bc"]), 1, help=FEATURE_INFO["mths_since_recent_bc"])
            mths_since_recent_inq = st.slider("Months since newest inquiry", 0, 120, int(st.session_state.profile["mths_since_recent_inq"]), 1, help=FEATURE_INFO["mths_since_recent_inq"])
            pct_tl_nvr_dlq = st.slider("% of trades never delinquent", 0, 100, int(st.session_state.profile["pct_tl_nvr_dlq"]), 1, help=FEATURE_INFO["pct_tl_nvr_dlq"])
            last_fico_range_low = st.slider("Lowest recent FICO score", 300, 850, int(st.session_state.profile["last_fico_range_low"]), 10, help=FEATURE_INFO["last_fico_range_low"])
            last_fico_range_high = st.slider("Highest recent FICO score", 300, 850, int(st.session_state.profile["last_fico_range_high"]), 10, help=FEATURE_INFO["last_fico_range_high"])
            years_since_earliest_cr = st.slider("Years since first credit line", 0, 50, int(st.session_state.profile["years_since_earliest_cr"]), 1, help=FEATURE_INFO["years_since_earliest_cr"])
            addr_state = st.text_input("State (2 letters)", value=st.session_state.profile["addr_state"], help=FEATURE_INFO["addr_state"]).upper()

        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(st.session_state.profile["home_ownership"]), help=FEATURE_INFO["home_ownership"])
        purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "car", "wedding"], index=["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "car", "wedding"].index(st.session_state.profile["purpose"]), help=FEATURE_INFO["purpose"])
        verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"], index=["Verified", "Source Verified", "Not Verified"].index(st.session_state.profile["verification_status"]), help=FEATURE_INFO["verification_status"])
        title = st.text_input("Loan Title", value=st.session_state.profile["title"], help=FEATURE_INFO["title"])

    current_profile = {
        "loan_amnt": loan_amnt, "int_rate": int_rate, "grade": grade, "emp_length": emp_length,
        "annual_inc": annual_inc, "dti": dti, "revol_util": revol_util, "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "total_acc": total_acc,
        "revol_bal": revol_bal, "total_bc_limit": total_bc_limit, "total_bal_ex_mort": total_bal_ex_mort,
        "avg_cur_bal": avg_cur_bal, "mo_sin_old_il_acct": mo_sin_old_il_acct,
        "mo_sin_old_rev_tl_op": mo_sin_old_rev_tl_op, "mo_sin_rcnt_rev_tl_op": mo_sin_rcnt_rev_tl_op,
        "mths_since_recent_bc": mths_since_recent_bc, "mths_since_recent_inq": mths_since_recent_inq,
        "pct_tl_nvr_dlq": pct_tl_nvr_dlq, "last_fico_range_low": last_fico_range_low,
        "last_fico_range_high": last_fico_range_high, "years_since_earliest_cr": years_since_earliest_cr,
        "addr_state": addr_state, "home_ownership": home_ownership, "purpose": purpose,
        "verification_status": verification_status, "title": title,
    }
    st.session_state.profile = current_profile

    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if st.button("🔍 Assess Credit Risk", use_container_width=True):
            st.session_state.prediction_result = run_prediction(st.session_state.profile)
    with btn2:
        if st.button("🔄 Reset Defaults", use_container_width=True):
            st.session_state.profile = dict(FEATURE_DEFAULTS)
            st.session_state.prediction_result = None
            st.rerun()
    with btn3:
        example_name = st.selectbox("Load Example", list(EXAMPLES.keys()), label_visibility="collapsed")
    with btn4:
        if st.button("📥 Apply Example", use_container_width=True):
            st.session_state.profile = dict(EXAMPLES[example_name])
            st.session_state.prediction_result = None
            st.rerun()

    st.markdown("---")
    st.subheader("Assessment Results")
    result = st.session_state.prediction_result
    if result is None:
        st.info("Click Assess Credit Risk to view the prediction result here.")
    elif not result.get("success"):
        st.error(result.get("error", "Prediction failed"))
    else:
        decision = result["decision"]
        prob = result["default_probability"]
        risk_level = result["risk_level"]
        conf = result["confidence"]
        threshold = result["optimal_threshold"]

        if decision == "APPROVE":
            st.success("✅ LOAN APPROVED")
        else:
            st.error("❌ LOAN REJECTED")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Default Probability", f"{prob:.2%}")
        m2.metric("Risk Level", risk_level)
        m3.metric("Confidence", f"{conf:.0%}")
        m4.metric("Optimal Threshold", f"{threshold:.0%}")

        st.write("**Explanation**")
        st.write(result.get("explanation", ""))
        fig = create_visualization(prob, threshold)
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Features used: {len(predictor.feature_list) if predictor.feature_list else 'Unknown'} | Threshold optimized for profit: {threshold:.0%}")

with right:
    st.subheader("🤖 LLM Chatbot")
    st.caption("The right panel is an OpenRouter-powered chat sandbox.")

    key_present = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if not key_present:
        st.warning("No API key was detected. Please export OPENROUTER_API_KEY (or OPENAI_API_KEY) before launching Streamlit.")
        st.code("export OPENROUTER_API_KEY='your-key-here'")
    else:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.agent_profile = dict(FEATURE_DEFAULTS)
            st.rerun()

        for msg in st.session_state.chat_messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage) and msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        prompt = st.chat_input("Ask about risk, scenarios, or update features...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = ask_agent(prompt)
                    st.markdown(reply)

    st.markdown("---")
    st.write("**Useful prompts**")
    st.markdown(
        "- What is my current risk?\n"
        "- If my income is 90000 and dti is 10, what happens?\n"
        "- Reset my profile\n"
        "- Show my current profile\n"
        "- Compare grade C vs grade B"
    )
