from gradio_client import Client
import time

def predict_credit_risk():
    try:
        # Connect to the local Gradio app (Docker container)
        client = Client("http://localhost:7860")
        print("✅ Connected to Credit Risk Predictor")
        
        # 29 input features in the exact order expected by the Gradio app
        inputs = [
            15000,          # loan_amnt
            12.5,           # int_rate
            "C",            # grade
            "5 years",      # emp_length
            75000,          # annual_inc
            18.5,           # dti
            45,             # revol_util (%)
            0,              # delinq_2yrs
            2,              # inq_last_6mths
            8,              # open_acc
            25,             # total_acc
            5000,           # revol_bal
            20000,          # total_bc_limit
            30000,          # total_bal_ex_mort
            2500,           # avg_cur_bal
            60,             # mo_sin_old_il_acct
            48,             # mo_sin_old_rev_tl_op
            12,             # mo_sin_rcnt_rev_tl_op
            6,              # mths_since_recent_bc
            3,              # mths_since_recent_inq
            95,             # pct_tl_nvr_dlq
            680,            # last_fico_range_low
            684,            # last_fico_range_high
            10,             # years_since_earliest_cr
            "CA",           # addr_state
            "RENT",         # home_ownership
            "debt_consolidation", # purpose
            "Verified",     # verification_status
            "Debt consolidation loan" # title
        ]
        
        print("📊 Sending prediction request...")
        
        # Call the API using the correct api_name found in config
        # The result returns a tuple of outputs corresponding to the app's output components
        result = client.predict(
            *inputs,
            api_name="/predict_loan"
        )
        
        print("\n✅ Prediction Successful!")
        print("-" * 30)
        
        # Result tuple: [decision_html, results_md, color_indicator, plot_json]
        decision_html = result[0]
        results_md = result[1]
        
        # Parse decision from HTML
        if "LOAN APPROVED" in decision_html:
            print("Decision: APPROVE ✅")
        else:
            print("Decision: REJECT ❌")
            
        print("\nDetails:")
        # Simple extraction of key metrics from markdown
        for line in results_md.split('\n'):
            if '| **' in line:
                print(line.replace('|', '').strip())
                
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the Docker container is running on port 7860.")

if __name__ == "__main__":
    predict_credit_risk()
