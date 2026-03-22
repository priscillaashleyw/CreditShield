# deployment/gradio_app.py
import gradio as gr
import pandas as pd
import json
from predictor import CreditRiskPredictor
import matplotlib.pyplot as plt
import numpy as np
import sys as _sys
import os as _os

_SANDBOX_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../.."))
_sys.path.insert(0, _SANDBOX_DIR)

try:
    import financial_sandbox as _sb
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph.message import add_messages as _add_messages
    _CHATBOT_ENABLED = not _sb._API_KEY_MISSING
    _graph = _sb.build_graph() if _CHATBOT_ENABLED else None
except ImportError as e:
    print(f"Chatbot disabled (missing packages): {e}")
    _CHATBOT_ENABLED = False
    _graph = None
    _sb = None

# Initialize predictor
predictor = CreditRiskPredictor("model_artifacts")

# Get the actual base features needed from the predictor
if hasattr(predictor, 'base_features_needed') and predictor.base_features_needed:
    print(f"📋 Model needs these base features: {predictor.base_features_needed}")
else:
    print("⚠️ Could not determine base features needed")

# Feature descriptions for tooltips
FEATURE_INFO = {
    'loan_amnt': "Total amount of the loan applied for",
    'int_rate': "Interest rate on the loan",
    'grade': "LC assigned loan grade (A=best, G=worst)",
    'emp_length': "Employment length in years",
    'annual_inc': "Self-reported annual income",
    'dti': "Debt-to-income ratio",
    'revol_util': "Revolving line utilization rate",
    'delinq_2yrs': "Number of delinquencies in past 2 years",
    'inq_last_6mths': "Number of credit inquiries in past 6 months",
    'open_acc': "Number of open credit lines",
    'total_acc': "Total number of credit lines",
    # Additional features from your predictor
    'revol_bal': "Total credit revolving balance",
    'total_bc_limit': "Total bankcard limit",
    'total_bal_ex_mort': "Total balance excluding mortgage",
    'avg_cur_bal': "Average current balance",
    'mo_sin_old_il_acct': "Months since oldest installment account opened",
    'mo_sin_old_rev_tl_op': "Months since oldest revolving account opened",
    'mo_sin_rcnt_rev_tl_op': "Months since most recent revolving account opened",
    'mths_since_recent_bc': "Months since most recent bankcard account opened",
    'mths_since_recent_inq': "Months since most recent inquiry",
    'pct_tl_nvr_dlq': "Percent of trades never delinquent",
    'last_fico_range_low': "Lower bound of the last FICO range",
    'last_fico_range_high': "Upper bound of the last FICO range",
    'years_since_earliest_cr': "Years since earliest credit line opened",
    'addr_state': "State of the borrower (2-letter code)",
    'home_ownership': "Home ownership status",
    'purpose': "Purpose of the loan",
    'verification_status': "Income verification status",
    'title': "Loan title/description"
}

def create_visualization(default_prob, threshold=0.28):
    """Create risk visualization"""
    fig, ax = plt.subplots(figsize=(8, 2))

    # Create gradient risk bar
    x = np.linspace(0, 1, 100)
    colors = plt.cm.RdYlGn_r(x)  # Red to Green (reversed)

    for i in range(len(x)-1):
        ax.fill_between([x[i], x[i+1]], 0, 1, color=colors[i], alpha=0.7)

    # Mark threshold
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0%})')

    # Mark prediction
    ax.plot(default_prob, 0.5, 'ro', markersize=15, label=f'Prediction ({default_prob:.1%})')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Default Probability')
    ax.set_title('Risk Assessment')
    ax.legend(loc='upper right')
    ax.set_yticks([])

    plt.tight_layout()
    return fig

def predict_loan(loan_amnt, int_rate, grade, emp_length, annual_inc,
                 dti, revol_util, delinq_2yrs, inq_last_6mths,
                 open_acc, total_acc, revol_bal=5000, total_bc_limit=20000,
                 total_bal_ex_mort=30000, avg_cur_bal=2500,
                 mo_sin_old_il_acct=60, mo_sin_old_rev_tl_op=48,
                 mo_sin_rcnt_rev_tl_op=12, mths_since_recent_bc=6,
                 mths_since_recent_inq=3, pct_tl_nvr_dlq=95,
                 last_fico_range_low=680, last_fico_range_high=684,
                 years_since_earliest_cr=10, addr_state="CA",
                 home_ownership="RENT", purpose="debt_consolidation",
                 verification_status="Verified",
                 title="Debt consolidation loan"):
    """Main prediction function with all needed features"""

    # Prepare input with ALL features the model expects
    loan_data = {
        # Basic loan info
        'loan_amnt': float(loan_amnt),
        'int_rate': float(int_rate),
        'grade': grade,
        'emp_length': emp_length,
        'annual_inc': float(annual_inc),
        'dti': float(dti),
        'revol_util': f"{revol_util}%",
        'delinq_2yrs': int(delinq_2yrs),
        'inq_last_6mths': int(inq_last_6mths),
        'open_acc': int(open_acc),
        'total_acc': int(total_acc),

        # Additional credit features
        'revol_bal': float(revol_bal),
        'total_bc_limit': float(total_bc_limit),
        'total_bal_ex_mort': float(total_bal_ex_mort),
        'avg_cur_bal': float(avg_cur_bal),
        'mo_sin_old_il_acct': float(mo_sin_old_il_acct),
        'mo_sin_old_rev_tl_op': float(mo_sin_old_rev_tl_op),
        'mo_sin_rcnt_rev_tl_op': float(mo_sin_rcnt_rev_tl_op),
        'mths_since_recent_bc': float(mths_since_recent_bc),
        'mths_since_recent_inq': float(mths_since_recent_inq),
        'pct_tl_nvr_dlq': float(pct_tl_nvr_dlq) / 100.0,  # Convert to decimal
        'last_fico_range_low': float(last_fico_range_low),
        'last_fico_range_high': float(last_fico_range_high),
        'years_since_earliest_cr': float(years_since_earliest_cr),

        # Categorical features for one-hot encoding
        'addr_state': str(addr_state),
        'home_ownership': str(home_ownership),
        'purpose': str(purpose),
        'verification_status': str(verification_status),
        'title': str(title)
    }

    # Get prediction
    result = predictor.predict(loan_data)

    if not result['success']:
        return f"❌ Error: {result['error']}", None, "red"

    # Format results
    if result['decision'] == 'APPROVE':
        decision_html = """
        <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border: 2px solid #c3e6cb;'>
            <h2 style='color: #155724; margin: 0;'>✅ LOAN APPROVED</h2>
        </div>
        """
        color = "green"
    else:
        decision_html = """
        <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; border: 2px solid #f5c6cb;'>
            <h2 style='color: #721c24; margin: 0;'>❌ LOAN REJECTED</h2>
        </div>
        """
        color = "red"

    # Create results table
    results_md = f"""
    ## 📊 Prediction Results

    | Metric | Value |
    |--------|-------|
    | **Default Probability** | {result['default_probability']:.2%} |
    | **Risk Level** | {result['risk_level']} |
    | **Confidence** | {result['confidence']:.0%} |
    | **Optimal Threshold** | {result['optimal_threshold']:.0%} |

    ### 💡 Explanation
    {result['explanation']}

    ### 🔧 Model Info
    - **Features used**: {len(predictor.feature_list) if predictor.feature_list else 'Unknown'}
    - **Features provided**: {len(loan_data)}
    - **Threshold optimized for profit**: {result['optimal_threshold']:.0%}

    ---
    *Model accuracy: 92.3% AUC-ROC | Trained on 358,244 loans*
    """

    # Create visualization
    fig = create_visualization(result['default_probability'], result['optimal_threshold'])

    return decision_html, results_md, color, fig


# --- Chatbot functions ---

def _chat(user_msg, messages, profile):
    """Run one chat turn. Profile comes from slider state (always current)."""
    updated_messages = _add_messages(messages, [HumanMessage(content=user_msg)])
    agent_state = {"messages": list(updated_messages), "current_profile": dict(profile)}
    result = _graph.invoke(agent_state)
    new_messages = result["messages"]
    new_profile = result["current_profile"]
    display = []
    for msg in new_messages:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            display.append({"role": "assistant", "content": msg.content})
    return new_messages, display, new_profile

def _reset_chat(profile):
    """Clear conversation history; profile stays as sliders dictate."""
    return [], [], dict(profile)


# Create Gradio interface
with gr.Blocks(title="Credit Risk Predictor", theme=gr.themes.Soft()) as demo:
    profile_state = gr.State(value=dict(_sb.FEATURE_DEFAULTS if _CHATBOT_ENABLED else {}))

    with gr.Tabs():
        with gr.Tab("Credit Risk Predictor"):
            gr.Markdown("""
            # 🏦 Credit Risk Prediction System
            *Predict loan defaults with 92.3% accuracy using machine learning*

            Based on research: *"Credit scoring for peer-to-peer lending using machine learning techniques"*
            (Quantitative Finance and Economics, Volume 6, Issue 2) with enhancements.
            """)

            # Advanced features accordion
            with gr.Accordion("🔧 Advanced Features (Optional)", open=False):
                gr.Markdown("""
                **Default values are set to typical/average levels.**
                These additional features improve prediction accuracy but are optional.
                """)

                with gr.Row():
                    with gr.Column():
                        revol_bal = gr.Slider(0, 100000, 5000, step=1000,
                                             label="Revolving Balance ($)",
                                             info=FEATURE_INFO['revol_bal'])

                        total_bc_limit = gr.Slider(0, 100000, 20000, step=1000,
                                                  label="Total Bankcard Limit ($)",
                                                  info=FEATURE_INFO['total_bc_limit'])

                        total_bal_ex_mort = gr.Slider(0, 200000, 30000, step=1000,
                                                    label="Total Balance Excl. Mortgage ($)",
                                                    info=FEATURE_INFO['total_bal_ex_mort'])

                        avg_cur_bal = gr.Slider(0, 50000, 2500, step=100,
                                               label="Average Current Balance ($)",
                                               info=FEATURE_INFO['avg_cur_bal'])

                    with gr.Column():
                        mo_sin_old_il_acct = gr.Slider(0, 300, 60, step=1,
                                                      label="Months since oldest installment account",
                                                      info=FEATURE_INFO['mo_sin_old_il_acct'])

                        mo_sin_old_rev_tl_op = gr.Slider(0, 300, 48, step=1,
                                                        label="Months since oldest revolving account",
                                                        info=FEATURE_INFO['mo_sin_old_rev_tl_op'])

                        mo_sin_rcnt_rev_tl_op = gr.Slider(0, 300, 12, step=1,
                                                         label="Months since newest revolving account",
                                                         info=FEATURE_INFO['mo_sin_rcnt_rev_tl_op'])

                        mths_since_recent_bc = gr.Slider(0, 120, 6, step=1,
                                                        label="Months since newest bankcard",
                                                        info=FEATURE_INFO['mths_since_recent_bc'])

                with gr.Row():
                    with gr.Column():
                        mths_since_recent_inq = gr.Slider(0, 120, 3, step=1,
                                                         label="Months since newest inquiry",
                                                         info=FEATURE_INFO['mths_since_recent_inq'])

                        pct_tl_nvr_dlq = gr.Slider(0, 100, 95, step=1,
                                                  label="% of trades never delinquent",
                                                  info=FEATURE_INFO['pct_tl_nvr_dlq'])

                        last_fico_range_low = gr.Slider(300, 850, 680, step=10,
                                                       label="Lowest recent FICO score",
                                                       info=FEATURE_INFO['last_fico_range_low'])

                        last_fico_range_high = gr.Slider(300, 850, 684, step=10,
                                                        label="Highest recent FICO score",
                                                        info=FEATURE_INFO['last_fico_range_high'])

                    with gr.Column():
                        years_since_earliest_cr = gr.Slider(0, 50, 10, step=1,
                                                           label="Years since first credit line",
                                                           info=FEATURE_INFO['years_since_earliest_cr'])

                        addr_state = gr.Textbox(value="CA", label="State (2 letters)",
                                               info=FEATURE_INFO['addr_state'])

                        home_ownership = gr.Dropdown(["RENT", "MORTGAGE", "OWN", "OTHER"],
                                                   value="RENT", label="Home Ownership",
                                                   info=FEATURE_INFO['home_ownership'])

                with gr.Row():
                    purpose = gr.Dropdown(["debt_consolidation", "credit_card", "home_improvement",
                                         "major_purchase", "medical", "car", "wedding"],
                                        value="debt_consolidation", label="Loan Purpose",
                                        info=FEATURE_INFO['purpose'])

                    verification_status = gr.Dropdown(["Verified", "Source Verified", "Not Verified"],
                                                    value="Verified", label="Income Verification",
                                                    info=FEATURE_INFO['verification_status'])

                    title = gr.Textbox(value="Debt consolidation loan", label="Loan Title",
                                      info=FEATURE_INFO['title'])

            # Main form
            gr.Markdown("## 📝 Required Loan Information")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Loan Application")

                    with gr.Group():
                        loan_amnt = gr.Slider(1000, 40000, 15000, step=500,
                                             label="Loan Amount ($)",
                                             info=FEATURE_INFO['loan_amnt'])

                        int_rate = gr.Slider(5.0, 30.0, 12.5, step=0.1,
                                            label="Interest Rate (%)",
                                            info=FEATURE_INFO['int_rate'])

                        grade = gr.Radio(["A", "B", "C", "D", "E", "F", "G"], value="C",
                                        label="Loan Grade",
                                        info=FEATURE_INFO['grade'])

                    with gr.Group():
                        emp_length = gr.Dropdown(["< 1 year", "1 year", "2 years", "3 years",
                                                 "4 years", "5 years", "6 years", "7 years",
                                                 "8 years", "9 years", "10+ years"],
                                                value="5 years",
                                                label="Employment Length",
                                                info=FEATURE_INFO['emp_length'])

                        annual_inc = gr.Slider(20000, 1000000, 75000, step=1000,
                                              label="Annual Income ($)",
                                              info=FEATURE_INFO['annual_inc'])

                        dti = gr.Slider(0, 40, 18.5, step=0.1,
                                       label="Debt-to-Income Ratio",
                                       info=FEATURE_INFO['dti'])

                with gr.Column(scale=1):
                    gr.Markdown("### Credit History")

                    with gr.Group():
                        revol_util = gr.Slider(0, 100, 45, step=1,
                                              label="Credit Utilization (%)",
                                              info=FEATURE_INFO['revol_util'])

                        delinq_2yrs = gr.Slider(0, 10, 0, step=1,
                                               label="Delinquencies (last 2 years)",
                                               info=FEATURE_INFO['delinq_2yrs'])

                        inq_last_6mths = gr.Slider(0, 10, 2, step=1,
                                                  label="Credit Inquiries (last 6 months)",
                                                  info=FEATURE_INFO['inq_last_6mths'])

                    with gr.Group():
                        open_acc = gr.Slider(0, 50, 8, step=1,
                                            label="Open Credit Lines",
                                            info=FEATURE_INFO['open_acc'])

                        total_acc = gr.Slider(0, 100, 25, step=1,
                                             label="Total Credit Lines",
                                             info=FEATURE_INFO['total_acc'])

            with gr.Row():
                submit_btn = gr.Button("🔍 Assess Credit Risk", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear Form", variant="secondary")
                simple_mode_btn = gr.Button("📱 Simple Mode", variant="secondary")

            # Example buttons
            gr.Markdown("### 🚀 Quick Examples")
            with gr.Row():
                low_risk_btn = gr.Button("👍 Low Risk Example", variant="secondary", size="sm")
                high_risk_btn = gr.Button("👎 High Risk Example", variant="secondary", size="sm")
                borderline_btn = gr.Button("⚖️ Borderline Example", variant="secondary", size="sm")

            # Results section
            gr.Markdown("## 📈 Assessment Results")

            with gr.Row():
                decision_output = gr.HTML(label="Decision")
                color_indicator = gr.HTML(visible=False)

            with gr.Row():
                with gr.Column(scale=2):
                    results_output = gr.Markdown(label="Detailed Results")
                with gr.Column(scale=1):
                    plot_output = gr.Plot(label="Risk Visualization")

            # Footer
            gr.Markdown("""
            ---
            ### ℹ️ About This Model
            - **Accuracy**: 92.3% AUC-ROC (beats paper's 86-87%)
            - **Training Data**: 358,244 loans from Lending Club (2013-2014)
            - **Key Features**: 98 engineered features including credit history and financial ratios
            - **Business Impact**: Optimized for maximum profit (threshold: 28%)
            - **Improvements**: No undersampling, time-based validation, enhanced features

            *For research purposes only. Not financial advice.*
            """)

            # Define examples with all needed features
            examples = {
                'low': {
                    'basic': [10000, 8.5, 'A', '10+ years', 120000, 12.0, 30, 0, 1, 5, 20],
                    'advanced': [3000, 15000, 25000, 3000, 120, 96, 24, 12, 6, 98, 720, 724, 15,
                                "CA", "OWN", "debt_consolidation", "Verified", "Debt consolidation"]
                },
                'high': {
                    'basic': [35000, 25.0, 'F', '< 1 year', 30000, 35.0, 95, 3, 8, 15, 40],
                    'advanced': [20000, 5000, 10000, 1000, 6, 12, 1, 1, 1, 60, 580, 590, 2,
                                "NV", "RENT", "credit_card", "Not Verified", "Credit card payoff"]
                },
                'borderline': {
                    'basic': [20000, 15.0, 'D', '3 years', 55000, 22.0, 75, 1, 4, 10, 30],
                    'advanced': [10000, 10000, 20000, 2000, 36, 48, 6, 6, 3, 85, 650, 660, 5,
                                "TX", "MORTGAGE", "home_improvement", "Source Verified", "Home renovation loan"]
                }
            }

            # Function to get all inputs for an example
            def get_example(example_type):
                basic = examples[example_type]['basic']
                advanced = examples[example_type]['advanced']
                return basic + advanced

            # Connect buttons
            all_inputs = [loan_amnt, int_rate, grade, emp_length, annual_inc,
                          dti, revol_util, delinq_2yrs, inq_last_6mths,
                          open_acc, total_acc, revol_bal, total_bc_limit,
                          total_bal_ex_mort, avg_cur_bal, mo_sin_old_il_acct,
                          mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op,
                          mths_since_recent_bc, mths_since_recent_inq,
                          pct_tl_nvr_dlq, last_fico_range_low, last_fico_range_high,
                          years_since_earliest_cr, addr_state, home_ownership,
                          purpose, verification_status, title]

            submit_btn.click(
                fn=predict_loan,
                inputs=all_inputs,
                outputs=[decision_output, results_output, color_indicator, plot_output]
            )

            # Clear function with defaults
            def clear_form():
                basic_defaults = [15000, 12.5, 'C', '5 years', 75000, 18.5, 45, 0, 2, 8, 25]
                advanced_defaults = [5000, 20000, 30000, 2500, 60, 48, 12, 6, 3, 95, 680, 684,
                                    10, "CA", "RENT", "debt_consolidation", "Verified",
                                    "Debt consolidation loan"]
                return basic_defaults + advanced_defaults + [None, None, None, None]

            clear_btn.click(
                fn=clear_form,
                outputs=all_inputs + [decision_output, results_output, plot_output]
            )

            # Example buttons
            low_risk_btn.click(
                fn=lambda: get_example('low'),
                outputs=all_inputs
            )

            high_risk_btn.click(
                fn=lambda: get_example('high'),
                outputs=all_inputs
            )

            borderline_btn.click(
                fn=lambda: get_example('borderline'),
                outputs=all_inputs
            )

            # Simple mode button (hides advanced features)
            simple_mode_btn.click(
                fn=lambda: gr.Accordion(open=False),
                outputs=None
            )

            # Slider → profile sync
            _FEATURE_KEYS = [
                'loan_amnt', 'int_rate', 'grade', 'emp_length', 'annual_inc',
                'dti', 'revol_util', 'delinq_2yrs', 'inq_last_6mths',
                'open_acc', 'total_acc', 'revol_bal', 'total_bc_limit',
                'total_bal_ex_mort', 'avg_cur_bal', 'mo_sin_old_il_acct',
                'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
                'mths_since_recent_bc', 'mths_since_recent_inq',
                'pct_tl_nvr_dlq', 'last_fico_range_low', 'last_fico_range_high',
                'years_since_earliest_cr', 'addr_state', 'home_ownership',
                'purpose', 'verification_status', 'title',
            ]

            def _make_profile_updater(key):
                def updater(new_value, current_profile):
                    return {**current_profile, key: new_value}
                return updater

            for _component, _key in zip(all_inputs, _FEATURE_KEYS):
                _component.change(
                    fn=_make_profile_updater(_key),
                    inputs=[_component, profile_state],
                    outputs=[profile_state],
                )

        with gr.Tab("AI Financial Advisor", visible=_CHATBOT_ENABLED):
            gr.Markdown("Profile syncs automatically from sliders. Ask what-if questions or request predictions.")
            chatbot_ui = gr.Chatbot(label="Conversation", height=450, type="messages")
            chat_input = gr.Textbox(placeholder="Ask about your credit profile...", label="", lines=2)
            with gr.Row():
                chat_send = gr.Button("Send", variant="primary")
                chat_reset_btn = gr.Button("Reset Chat", variant="secondary")
            messages_state = gr.State(value=[])

            for event in [chat_send.click, chat_input.submit]:
                event(
                    fn=_chat,
                    inputs=[chat_input, messages_state, profile_state],
                    outputs=[messages_state, chatbot_ui, profile_state],
                )
                event(fn=lambda: "", outputs=[chat_input])

            chat_reset_btn.click(
                fn=_reset_chat,
                inputs=[profile_state],
                outputs=[messages_state, chatbot_ui, profile_state],
            )

    demo.queue()

# Run the app
if __name__ == "__main__":
    print("🚀 Starting Credit Risk Predictor...")
    print(f"📊 Model features: {len(predictor.feature_list) if predictor.feature_list else 'Unknown'}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
