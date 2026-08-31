# ============================================================
# CCAR RISK SCORING - CENTRAL CONFIGURATION
# ============================================================


# ============================================================
# 1. PLATFORM / DATA PATHS
# ============================================================

VOLUME_PATH = (     "/Volumes/workspace/risk_scoring/risk_scoring_poc" )

# INPUT_FILE_PATH = (
#     f"{VOLUME_PATH}/CCAR_Mortgage_data_for_model_DEV_20260319_01.csv"
# )

INPUT_FILE_PATH = "/Volumes/workspace/risk_scoring/risk_scoring_poc/CCAR_Mortgage_data_for_model_DEV_20260319_01.csv"

pd_input_file_v0 = "/Volumes/workspace/risk_scoring/risk_scoring_poc/CCAR_Mortgage_data_for_model_DEV_20260319_01.csv"

# pd_model_path = (
#     f"{VOLUME_PATH}/ccar_pd_model_2026-03-22.pkl"
# )

pd_model_path = "/Volumes/workspace/risk_scoring/risk_scoring_poc/ccar_pd_model_2026-03-22.pkl"

DELTA_TABLE = (    "workspace.risk_scoring.mortgage_model_dev" )


### from cell 21 ###
x_list=['original_balance', 'credit_score_orig', 'loan_to_value_orig', 'interest_rate', 'loan_term_months',
        'delta_Unemployment1',
       'delta_Mortgage_rate1', 'delta_House_Price_Index__Level1',
       'delta_Unemployment3', 'delta_Mortgage_rate3',
       'delta_House_Price_Index__Level3', 'delta_Unemployment6',
       'delta_Mortgage_rate6', 'delta_House_Price_Index__Level6',
       'delta_Unemployment12', 'delta_Mortgage_rate12',
       'delta_House_Price_Index__Level12', 'delta_Unemployment24',
       'delta_Mortgage_rate24', 'delta_House_Price_Index__Level24']

x_list_v2=['original_balance', 'credit_score_orig', 'loan_to_value_orig', 'interest_rate',
        'delta_Unemployment1',
       'delta_Unemployment3',
       'delta_Unemployment6',
       'delta_Mortgage_rate6', 'delta_House_Price_Index__Level6',
       'delta_Unemployment12',
       'delta_Unemployment24' ]

x_list_v3= ['original_balance', 'credit_score_orig', 'loan_to_value_orig', 'interest_rate']

# y_list= ['flag_default']

decile_list = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

pd_model = 'ccar_pd_model_2026-03-22.pkl'
# pd_model_path = "/content/drive/MyDrive/Colab Notebooks/ccar_pd_model_2026-03-22.pkl"
# pd_model_path = "/Volumes/workspace/risk_scoring/risk_scoring_poc/ccar_pd_model_2026-03-22.pkl"

llm_model_name = "gpt-4o-mini"

# Use the full Absolute Path:
# pd_input_file = "/content/drive/MyDrive/Colab Notebooks/CCAR_Mortgage_data_for_model_DEV_20260319_01.csv"
# pd_input_file_v0 = "/Volumes/workspace/risk_scoring/risk_scoring_poc/CCAR_Mortgage_data_for_model_DEV_20260319_01.csv"

# pd_vintage = 201604
pd_x_list = x_list_v2
pd_y = 'flag_default'
pd_threshold = 0.50      ### pd_threshold value will be used in the agent_a step.

my_scoring_vintage = '201604'

### from Cell 48 ###
sample_knowledge_base = [
    "If model recall drops below threshold, investigate data drift, feature distribution shift, and target definition changes.",
    "If AUC or KS deteriorates materially, perform model monitoring review and consider recalibration.",
    "If data validation fails, stop scoring and escalate to data owner.",
    "If predicted bad rate deviates materially from actual bad rate, review calibration and cutoff strategy."
]


### from cell 76 ###
### _11_agent_DeterministicSQLAgent

semantic_map = {
    "entities": {
        "portfolio": "data_1_uploaded_db"
    },
    "metrics": {
        "total_original_balance": "SUM(original_balance)",
        "total_current_balance": "SUM(current_balance)",
        "avg_credit_score": "AVG(credit_score_orig)"
    },
    "dimensions": {
        "type": "product_type",
        "vintage": "report_yrmo"
    }
}

# ============================================================
# 08. DATABRICKS SECRET CONFIGURATION
# ============================================================


# ============================================================
# 10. DATABRICKS SECRET CONFIGURATION
# ============================================================

SECRET_SCOPE = "_01_risk-scoring-poc"

OPENAI_SECRET_KEY = "OPENAI_API_KEY"
LANGSMITH_SECRET_KEY = "LANGSMITH_API_KEY"

LANGSMITH_PROJECT = "JCHEN-Databricks-credit-risk_scoring-validation-langgraph"

# project_name = "JCHEN-Databricks-credit-risck_scoring-validation-langgraph"

