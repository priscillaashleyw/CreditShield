import os
import sys
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from gradio_client import Client
import json

# --- CONFIGURATION ---
# Check for OpenAI API Key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = ""
if "OPENAI_API_KEY" not in os.environ:
    print("⚠️  OPENAI_API_KEY not found in environment variables.")
    print("Please set it: export OPENAI_API_KEY='sk-...'")
    # For demo purposes, we might let it fail if not set, 
    # or prompt the user (but in this non-interactive shell we can't).

# --- FEATURE MAPPING ---
# Maps feature names to their index in the API list [0..28]
FEATURE_MAP = {
    'loan_amnt': 0, 'int_rate': 1, 'grade': 2, 'emp_length': 3, 'annual_inc': 4,
    'dti': 5, 'revol_util': 6, 'delinq_2yrs': 7, 'inq_last_6mths': 8, 'open_acc': 9,
    'total_acc': 10, 'revol_bal': 11, 'total_bc_limit': 12, 'total_bal_ex_mort': 13,
    'avg_cur_bal': 14, 'mo_sin_old_il_acct': 15, 'mo_sin_old_rev_tl_op': 16,
    'mo_sin_rcnt_rev_tl_op': 17, 'mths_since_recent_bc': 18, 'mths_since_recent_inq': 19,
    'pct_tl_nvr_dlq': 20, 'last_fico_range_low': 21, 'last_fico_range_high': 22,
    'years_since_earliest_cr': 23, 'addr_state': 24, 'home_ownership': 25,
    'purpose': 26, 'verification_status': 27, 'title': 28
}

# Default profile (Baseline)
DEFAULT_PROFILE = [
    15000, 12.5, "C", "5 years", 75000, 18.5, 45, 0, 2, 8, 25, 
    5000, 20000, 30000, 2500, 60, 48, 12, 6, 3, 95, 680, 684, 
    10, "CA", "RENT", "debt_consolidation", "Verified", "Debt consolidation loan"
]

# --- STATE MANAGEMENT ---
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_profile: List[Any]

# --- TOOLS ---

@tool
def predict_credit_risk(updates: dict[str, Any] = {}, profile_list: list[Any] = None) -> str:
    """
    Predicts credit risk based on the current loan profile, optionally updating specific features.
    
    Args:
        updates: A dictionary of feature names and their new values (e.g., {"annual_inc": 80000, "dti": 15}).
                Available features: loan_amnt, int_rate, annual_inc, dti, revol_util, grade, etc.
        profile_list: (Injected by state, do not set manually) The current full profile list.
    
    Returns:
        JSON string containing the decision (APPROVE/REJECT), default probability, and risk level.
    """
    # This function effectively acts as both "update profile" and "predict"
    # In a real agent, we might separate them, but bundling is efficient here.
    
    # We need access to the current state. 
    # NOTE: LangChain tools don't inherently see the graph state unless passed explicitly or via global/closure.
    # For simplicity in this demo, we'll assume the agent passes the current profile 
    # OR we use a global variable (less clean) OR we use the 'updates' to modify a base.
    
    # Let's use the `profile_list` argument which we will inject from the agent node.
    # If not provided, use default (fallback).
    current_list = profile_list.copy() if profile_list else DEFAULT_PROFILE.copy()
    
    # Apply updates
    updated_features = []
    for feature, value in updates.items():
        if feature in FEATURE_MAP:
            idx = FEATURE_MAP[feature]
            current_list[idx] = value
            updated_features.append(f"{feature}={value}")
    
    # Call API
    try:
        client = Client("http://localhost:7860")
        result = client.predict(*current_list, api_name="/predict_loan")
        
        # Result format from app: [decision_html, results_md, color, fig]
        decision_html = result[0]
        results_md = result[1]
        
        decision = "APPROVE" if "LOAN APPROVED" in decision_html else "REJECT"
        
        # Extract probability (rough extraction from markdown)
        prob = "Unknown"
        threshold = "Unknown"
        for line in results_md.split('\n'):
            if "Default Probability" in line:
                prob = line.split('|')[2].strip()
            if "Optimal Threshold" in line:
                threshold = line.split('|')[2].strip()
                
        return json.dumps({
            "decision": decision,
            "probability": prob,
            "threshold": threshold,
            "updates_applied": updated_features
        })
        
    except Exception as e:
        return f"Error calling prediction API: {str(e)}"

# --- AGENT NODES ---

def chatbot(state: AgentState):
    """
    The main chatbot node that invokes the LLM.
    """
    messages = state['messages']
    profile = state.get('current_profile', DEFAULT_PROFILE)
    
    # We must inject the current profile into the tool call if the LLM decides to call it.
    # However, standard OpenAI function calling doesn't let us easily inject hidden args *after* generation.
    # STRATEGY: The LLM generates the 'updates' dict. The 'tool_node' will handle the merging.
    
    llm = ChatOpenAI(model="openai/gpt-3.5-turbo", temperature=0)
    llm_with_tools = llm.bind_tools([predict_credit_risk])
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    """
    Executes tools requested by the LLM.
    Handles state updates (modifying the profile).
    """
    messages = state['messages']
    last_message = messages[-1]
    profile = state.get('current_profile', list(DEFAULT_PROFILE)) # Copy!
    
    outputs = []
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "predict_credit_risk":
            # Extract arguments provided by LLM
            updates = tool_call["args"].get("updates", {})
            
            # Update the ACTUAL state profile
            for feature, value in updates.items():
                if feature in FEATURE_MAP:
                    idx = FEATURE_MAP[feature]
                    profile[idx] = value
            
            # Execute the tool fn
            # We explicitly pass the *updated* profile to the tool so it predicts on the new state
            result = predict_credit_risk.invoke({"updates": updates, "profile_list": profile})
            
            outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
            )
            
    # Return the tool outputs AND the updated profile to the state
    return {
        "messages": outputs, 
        "current_profile": profile  # Persist the changes!
    }

# --- GRAPH DEFINITION ---
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", chatbot)
    workflow.add_node("tools", tool_node)
    
    workflow.set_entry_point("agent")
    
    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# --- MAIN LOOP ---
def run_interactive_session():
    graph = build_graph()
    
    # Initialize state with a system message and default profile
    initial_state = {
        "messages": [
            SystemMessage(content="""You are a helpful Financial Credit Risk Assistant. 
            You have access to a 'What-If' simulation tool that can predict loan approval probability.
            
            Users will ask you questions like:
            - "What is my current risk?"
            - "If I increase my income to 90k, will I be approved?"
            - "What if I lower my DTI to 10?"
            
            Identify which features the user wants to change, call the `predict_credit_risk` tool with those updates.
            Always summarize the result clearly to the user, highlighting the change in probability if applicable.
            """)
        ],
        "current_profile": list(DEFAULT_PROFILE)
    }
    
    print("🤖 Financial Sandbox Agent Ready! (Type 'quit' to exit)")
    print("-----------------------------------------------------")
    print(f"Current Profile Baseline: Income=$75k, Loan=$15k, DTI=18.5, Grade=C")
    
    # We maintain the conversation history in `current_state`
    current_state = initial_state
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            # Append user message to history
            current_state["messages"].append(HumanMessage(content=user_input))
            
            print("Thinking...", end="", flush=True)
            
            # Run the graph until it produces a final response
            for event in graph.stream(current_state):
                for key, value in event.items():
                    if key == "tools":
                        print(" [Running Simulation] ", end="", flush=True)
                    elif key == "agent":
                        print(".", end="", flush=True)
                    
                    if "messages" in value:
                        current_state["messages"] = add_messages(current_state["messages"], value["messages"])
                    if "current_profile" in value:
                        current_state["current_profile"] = value["current_profile"]
            
            print("\n")
            
            last_msg = current_state['messages'][-1]
            print(f"Agent: {last_msg.content}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    run_interactive_session()
