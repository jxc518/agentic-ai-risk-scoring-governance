import os
import pandas as pd
import numpy as np
import joblib
import ccar_config
from scipy.stats import ks_2samp
import json

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

#####  Step 01: uploading class

def step_1_uploading(input_path, var_list ):

    validation_report = []

    def add_validation(check, status, details):
        validation_report.append({
            "Validation": check,
            "Status": status,
            "Details": details
        })

    # # 01. File exists
    # if os.path.exists(input_path):
    #     add_validation("File exists", "PASS", input_path)
    # else:
    #     add_validation("File exists", "FAIL", input_path)
    #     raise FileNotFoundError(f"{input_path} does not exist.")

    # # 02. Empty file
    # file_size = os.path.getsize(input_path)

    # if file_size > 0:
    #     add_validation("Empty file", "PASS", f"{file_size} bytes")
    # else:
    #     add_validation("Empty file", "FAIL", "0 bytes")
    #     raise ValueError("Input file is empty.")

    # # 03. Read file
    # df = pd.read_csv(input_path)

    if isinstance(input_path, pd.DataFrame):

        df = input_path.copy()

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        print("Input source: Pandas DataFrame")
        print("Shape:", df.shape)

    else:

        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Input file does not exist: {input_path}"
            )

        file_size = os.path.getsize(input_path)

        if file_size <= 0:
            raise ValueError("Input file is empty.")

        df = pd.read_csv(input_path)

        print("Input source: CSV file")
        print("Shape:", df.shape)

    upload_report = pd.DataFrame(validation_report)

    # Detailed exploratory statistics are intentionally suppressed in the engineering/demo notebook.
    # print("---------------- 1-A-01: Validation Summary Report ----------------")
    # print(df.yrmo.value_counts())

    # print("\n---------------- 1-A-02: Summary of Numerical Columns ----------------")
    # # print(df[ var_list ].describe(decile_list))
    # print(df[ var_list ].describe( percentiles=ccar_config.decile_list ))

    return df, upload_report


#####  Step 02: DataValidator class
class _02_DataValidator:

    def __init__(self, df, required_columns, numeric_columns, target_col="flag_default"):

        self.df = df
        self.required_columns = required_columns
        self.numeric_columns = numeric_columns
        self.target_col = target_col
        self.report = []

    def add_result(self, check, status, details):

        self.report.append({
            "Validation": check,
            "Status": status,
            "Details": details
        })

    def _02_A_validate_required_columns(self):

        missing_cols = list(set(self.required_columns) - set(self.df.columns))

        if len(missing_cols) == 0:
            self.add_result(
                "Required columns",
                "PASS",
                f"{len(self.required_columns)} required columns found"
            )
        else:
            self.add_result(
                "Required columns",
                "FAIL",
                f"Missing columns: {missing_cols}"
            )
            raise Exception(f"Missing columns: {missing_cols}")

    def _02_B_validate_duplicates(self):

        dup = self.df.duplicated().sum()

        if dup == 0:
            self.add_result(
                "Duplicate Rows",
                "PASS",
                "No duplicates"
            )
        else:
            self.add_result(
                "Duplicate Rows",
                "WARNING",
                f"{dup} duplicate rows"
            )

    def _02_C_check_missing(self, threshold=10):

        missing_pct = self.df.isna().mean() * 100
        high_missing = missing_pct[missing_pct > threshold].sort_values(ascending=False)

        if len(high_missing) == 0:
            self.add_result(
                "Missing values",
                "PASS",
                f"No variable >{threshold}% missing"
            )
        else:
            details = ", ".join([
                f"{col}: {val:.1f}%"
                for col, val in high_missing.items()
            ])

            self.add_result(
                "Missing values",
                "WARNING",
                details
            )

    def _02_D_check_yrmo(self):

        if "yrmo" in self.df.columns:
            yrmo_count = self.df["yrmo"].nunique()

            self.add_result(
                "YRMO distribution",
                "PASS",
                f"{yrmo_count} unique months"
            )
        else:
            self.add_result(
                "YRMO distribution",
                "WARNING",
                "yrmo column not found"
            )

    def _02_E_check_target(self):

        if self.target_col in self.df.columns:

            target_values = sorted(self.df[self.target_col].dropna().unique())

            if set(target_values).issubset({0, 1}):

                bad_rate = self.df[self.target_col].mean() * 100

                self.add_result(
                    "Target distribution",
                    "PASS",
                    f"Bad rate = {bad_rate:.2f}%"
                )

            else:
                self.add_result(
                    "Target distribution",
                    "FAIL",
                    f"Unexpected target values: {target_values}"
                )
        else:
            self.add_result(
                "Target distribution",
                "WARNING",
                f"{self.target_col} column not found"
            )

    def _02_F_check_outliers(self):

        outlier_count = 0

        for col in self.numeric_columns:

            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):

                q1 = self.df[col].quantile(0.01)
                q99 = self.df[col].quantile(0.99)

                outlier_count += (
                    (self.df[col] < q1) |
                    (self.df[col] > q99)
                ).sum()

        if outlier_count == 0:
            self.add_result(
                "Outliers",
                "PASS",
                "No extreme outliers detected"
            )
        else:
            self.add_result(
                "Outliers",
                "WARNING",
                f"{outlier_count} values outside 1%-99% range"
            )

    def run_all(self):

        self._02_A_validate_required_columns()
        self._02_B_validate_duplicates()
        self._02_C_check_missing()
        self._02_D_check_yrmo()
        self._02_E_check_target()
        self._02_F_check_outliers()

        return pd.DataFrame(self.report)
    

##### Step 03: Feature Engineer class
class _03_FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()

    def cap_outliers(self):
        # optional placeholder
        return self

    def handle_missing_values(self):
        # optional placeholder
        return self

    def create_model_features(self):
        # optional placeholder
        return self

    def risk_bucket_creation(self):
        # optional placeholder
        return self

    def categorical_encoding(self):
        # optional placeholder
        return self

    def categorical_encoding(self):
        # optional placeholder
        return self

    def date_feature_selection(self):
        # optional placeholder
        return self

    def run_all(self):
        self.cap_outliers()
        self.handle_missing_values()
        self.create_model_features()
        self.risk_bucket_creation()
        self.categorical_encoding()
        self.categorical_encoding()
        self.date_feature_selection()
        return self.df



##### Step 04: Model Scorer class
class _04_Model_Scorer:

    def __init__(self, df, model, x_list, y_target, pd_threshold):
        self.df = df.copy()
        self.model = model
        self.x_list = x_list
        self.y_target = y_target
        self.pd_threshold = pd_threshold

    def _scoring_action(self, vintage):

        try:
            df_v = self.df.loc[
                self.df["report_yrmo"].astype(str) == str(vintage)
            ].reset_index(drop=True)

            if df_v.empty:
                return None, {"Vintage": vintage, "Status": "Skipped (No Data)"}

            X = df_v[self.x_list]
            y = df_v[self.y_target]

            y_proba = self.model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= self.pd_threshold).astype(int)

            df_v["pd_score"] = y_proba
            df_v["pred_flag"] = y_pred

            tn, fp, fn, tp = confusion_matrix(
                y, y_pred, labels=[0, 1]
            ).ravel()

            report = {
                "Vintage": vintage,
                "Status": "Success",
                "Count": len(df_v),
                "Avg_PD": y_proba.mean(),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1": f1_score(y, y_pred, zero_division=0),
                "TP": tp,
                "FN": fn,
                "TN": tn,
                "FP": fp
            }

            return df_v, report

        except Exception as e:
            return None, {"Vintage": vintage, "Status": f"Error: {str(e)}"}

    def run_all(self, vintage):
        df_scored, report = self._scoring_action(vintage)
        report_df = pd.DataFrame([report])
        return df_scored, report_df
    

##### Step 05: Performance Evaluator class

class _05_PerformanceEvaluator:

    def __init__(self, df_scored, y_target, score_col="pd_score", pred_col="pred_flag"):

        self.df = df_scored.copy()
        self.y_target = y_target
        self.score_col = score_col
        self.pred_col = pred_col
        self.report = []

    def add_result(self, metric, status, value=None, details=""):

        self.report.append({
            "Metric": metric,
            "Status": status,
            "Value": value,
            "Details": details
        })

    def check_auc(self, threshold_warning=0.65, threshold_fail=0.60):

        y = self.df[self.y_target]
        score = self.df[self.score_col]

        if y.nunique() < 2:
            self.add_result(
                "AUC",
                "SKIPPED",
                None,
                "AUC requires both good and bad observations"
            )
            return

        auc = roc_auc_score(y, score)

        if auc < threshold_fail:
            status = "FAIL"
        elif auc < threshold_warning:
            status = "WARNING"
        else:
            status = "PASS"

        self.add_result("AUC", status, round(auc, 4))

    def check_ks(self, threshold_warning=0.25, threshold_fail=0.20):

        good = self.df.loc[self.df[self.y_target] == 0, self.score_col]
        bad = self.df.loc[self.df[self.y_target] == 1, self.score_col]

        if len(good) == 0 or len(bad) == 0:
            self.add_result(
                "KS",
                "SKIPPED",
                None,
                "KS requires both good and bad observations"
            )
            return

        ks = ks_2samp(bad, good).statistic

        if ks < threshold_fail:
            status = "FAIL"
        elif ks < threshold_warning:
            status = "WARNING"
        else:
            status = "PASS"

        self.add_result("KS", status, round(ks, 4))

    def check_classification_metrics(self):

        y = self.df[self.y_target]
        y_pred = self.df[self.pred_col]

        self.add_result(
            "Precision",
            "INFO",
            round(precision_score(y, y_pred, zero_division=0), 4)
        )

        self.add_result(
            "Recall",
            "INFO",
            round(recall_score(y, y_pred, zero_division=0), 4)
        )

        self.add_result(
            "F1",
            "INFO",
            round(f1_score(y, y_pred, zero_division=0), 4)
        )

    def check_bad_rate(self):

        actual_bad_rate = self.df[self.y_target].mean()
        predicted_bad_rate = self.df[self.pred_col].mean()
        avg_pd = self.df[self.score_col].mean()

        self.add_result(
            "Actual Bad Rate",
            "INFO",
            round(actual_bad_rate, 4)
        )

        self.add_result(
            "Predicted Bad Rate",
            "INFO",
            round(predicted_bad_rate, 4)
        )

        self.add_result(
            "Average PD",
            "INFO",
            round(avg_pd, 4)
        )

    def run_all(self):

        self.check_auc()
        self.check_ks()
        self.check_classification_metrics()
        self.check_bad_rate()

        return pd.DataFrame(self.report)
    

##### Step 06: Conditional Router class

class _06_ConditionalRouter:

    def __init__(self, validation_report, scoring_report, performance_report=None):
        self.validation_report = validation_report
        self.scoring_report = scoring_report
        self.performance_report = performance_report
        self.route_decision = None
        self.reason = None

    def _get_status_col(self, df):
        if df is None:
            return None
        if "Status" in df.columns:
            return "Status"
        if "status" in df.columns:
            return "status"
        return None

    def check_validation_failure(self):
        status_col = self._get_status_col(self.validation_report)

        if status_col and "FAIL" in self.validation_report[status_col].values:
            self.route_decision = "STOP"
            self.reason = "Data validation failed"
            return True

        return False

    def check_scoring_failure(self):
        status_col = self._get_status_col(self.scoring_report)

        if status_col is None:
            return False

        status_list = self.scoring_report[status_col].astype(str).tolist()

        if any("Error" in s for s in status_list):
            self.route_decision = "STOP"
            self.reason = "Model scoring error"
            return True

        if any("Skipped" in s for s in status_list):
            self.route_decision = "HUMAN_REVIEW"
            self.reason = "No data found for selected vintage"
            return True

        return False

    def check_performance_warning(self):
        status_col = self._get_status_col(self.performance_report)

        if status_col is None:
            return False

        if "FAIL" in self.performance_report[status_col].values:
            self.route_decision = "RECOMMENDATION_AGENT"
            self.reason = "Model performance failed"
            return True

        if "WARNING" in self.performance_report[status_col].values:
            self.route_decision = "RAG_AGENT"
            self.reason = "Model performance warning"
            return True

        return False

    def default_route(self):
        self.route_decision = "REPORTING_AGENT"
        self.reason = "All checks passed"

    def run_all(self):
        if self.check_validation_failure():
            return self.get_decision()

        if self.check_scoring_failure():
            return self.get_decision()

        if self.check_performance_warning():
            return self.get_decision()

        self.default_route()
        return self.get_decision()

    def get_decision(self):
        return {
            "route_decision": self.route_decision,
            "route_reason": self.reason,
            "route_result": self.route_decision
        }


##### Step 07: Reporting Agent

class _07_ReportingAgent:

    def __init__(self, validation_report, scoring_report, performance_report, route_result):
        self.validation_report = validation_report
        self.scoring_report = scoring_report
        self.performance_report = performance_report
        self.route_result = route_result

    def run_all(self):

        validation_summary = self.validation_report["Status"].value_counts().to_dict()

        if self.scoring_report is not None and len(self.scoring_report) > 0:
            scoring = self.scoring_report.iloc[0].to_dict()
        else:
            scoring = {}

        technical_report = pd.DataFrame([{
            "Validation PASS": validation_summary.get("PASS", 0),
            "Validation WARNING": validation_summary.get("WARNING", 0),
            "Validation FAIL": validation_summary.get("FAIL", 0),

            "Scoring Status": scoring.get("Status"),
            "Vintage": scoring.get("Vintage"),
            "Record Count": scoring.get("Count"),
            "Average PD": scoring.get("Avg_PD"),
            "Precision": scoring.get("Precision"),
            "Recall": scoring.get("Recall"),
            "F1": scoring.get("F1"),

            "Route Decision": self.route_result.get("route_decision"),
            "Route Reason": self.route_result.get("reason")
        }])

        return technical_report
    

##### Step 08: RAG Agent class

class _08_RAG_Agent:

    def __init__(self, route_result, performance_report, knowledge_base):
        self.route_result = route_result
        self.performance_report = performance_report
        self.knowledge_base = knowledge_base

    def build_query(self):

        reason = self.route_result.get("reason", "")

        issues = self.performance_report[
            self.performance_report["Status"].isin(["WARNING", "FAIL"])
        ]

        metrics = ", ".join(issues["Metric"].tolist())

        query = (
            f"Model monitoring issue: {reason}. "
            f"Problematic metrics: {metrics}. "
            f"Retrieve relevant model governance monitoring guidance."
        )

        return query

    def retrieve_documents(self, query):

        relevant_docs = []

        for doc in self.knowledge_base:
            if any(word.lower().strip(".,") in doc.lower() for word in query.split()):
                relevant_docs.append(doc)

        return relevant_docs[:5]

    def run_all(self):

        query = self.build_query()

        retrieved_docs = self.retrieve_documents(query)

        if retrieved_docs:
            rag_answer = " ".join(retrieved_docs)
        else:
            rag_answer = "No relevant governance monitoring guidance found."

        return {
            "rag_query": query,
            "retrieved_guidance": retrieved_docs,
            "rag_answer": rag_answer
        }


##### Step 09: Recommendation Agent class

class _09_RecommendationAgent:

    def __init__(self, route_result, performance_report, rag_answer=None):
        self.route_result = route_result
        self.performance_report = performance_report.copy()
        self.rag_answer = rag_answer

    def identify_issues(self):
        return self.performance_report[
            self.performance_report["Status"].isin(["WARNING", "FAIL"])
        ]

    def generate_recommendations(self):

        issues = self.identify_issues()
        recs = []

        if issues.empty:
            recs.append({
                "Issue": "No material issue",
                "Severity": "LOW",
                "Recommendation": "Proceed to final reporting.",
                "Human Approval Required": "No"
            })
            return pd.DataFrame(recs)

        for _, row in issues.iterrows():

            metric = row["Metric"]
            status = row["Status"]
            value = row["Value"]

            if metric in ["AUC", "KS"] and status == "FAIL":
                severity = "HIGH"
                recommendation = (
                    "Stop automated approval. Investigate model degradation, "
                    "feature drift, population shift, and consider recalibration."
                )

            elif metric in ["AUC", "KS"] and status == "WARNING":
                severity = "MEDIUM"
                recommendation = (
                    "Continue with caution. Review score distribution, PSI, "
                    "feature drift, and recent vintage behavior."
                )

            elif metric in ["Recall", "Precision", "F1"]:
                severity = "MEDIUM"
                recommendation = (
                    "Review cutoff threshold, confusion matrix, and business cost tradeoff."
                )

            elif metric in ["Actual Bad Rate", "Predicted Bad Rate", "Average PD"]:
                severity = "MEDIUM"
                recommendation = (
                    "Review calibration and predicted-vs-actual bad-rate alignment."
                )

            else:
                severity = "MEDIUM"
                recommendation = "Review this metric manually."

            recs.append({
                "Issue": metric,
                "Status": status,
                "Value": value,
                "Severity": severity,
                "Recommendation": recommendation,
                "Human Approval Required": "Yes"
            })

        return pd.DataFrame(recs)

    def run_all(self):
        return self.generate_recommendations()
    

##### Step 10: Human Approval agent

class _10_HumanApproval:

    def __init__(self, recommendation_report):
        self.recommendation_report = recommendation_report.copy()

    def approval_required(self):

        if "Human Approval Required" not in self.recommendation_report.columns:
            return True

        return (
            self.recommendation_report["Human Approval Required"]
            .astype(str)
            .str.upper()
            .eq("YES")
            .any()
        )

    def run_all(self):

        approval_needed = self.approval_required()

        if approval_needed:
            decision = "PENDING_HUMAN_APPROVAL"
            status = "WAIT"
            details = "Human approval is required before final reporting."
        else:
            decision = "AUTO_APPROVED"
            status = "CONTINUE"
            details = "No material issue. Continue to final reporting."

        approval_result = {
            "Approval Decision": decision,
            "Pipeline Status": status,
            "Details": details
        }

        return pd.DataFrame([approval_result])
    

##### Step 11: Deterministic SQL Agent class

class _11_DeterministicSQLAgent:

    def __init__(self, client, model_name, semantic_map, conn):
        self.client = client
        self.model_name = model_name
        self.semantic_map = semantic_map
        self.conn = conn

    def parse_user_question(self, question):

        ccar_tool = {
            "type": "function",
            "function": {
                "name": "generate_ccar_intent",
                "description": "Extracts CCAR query parameters from natural language.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "enum": ["portfolio"]
                        },
                        "metric": {
                            "type": "string",
                            "enum": [
                                "total_original_balance",
                                "total_current_balance",
                                "avg_credit_score"
                            ]
                        },
                        "group_by": {
                            "type": "string",
                            "enum": ["vintage"]
                        },
                        "filter_key": {
                            "type": "string",
                            "enum": ["type"]
                        },
                        "filter_val": {
                            "type": "string",
                            "description": "The product name, e.g., HELOC"
                        }
                    },
                    "required": [
                        "entity",
                        "metric",
                        "group_by",
                        "filter_key",
                        "filter_val"
                    ]
                }
            }
        }

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a specialized CCAR Risk data architect. "
                        "Map user questions to the provided tool only."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            tools=[ccar_tool],
            tool_choice={
                "type": "function",
                "function": {
                    "name": "generate_ccar_intent"
                }
            }
        )

        # --------------------------------------------------------
        # LLM Cost Calculation
        # GPT-4o-mini pricing
        # Input  : $0.15 / 1M tokens
        # Output : $0.60 / 1M tokens
        # --------------------------------------------------------

        print("\n" + "=" * 60)
        print("LLM TOKEN & COST SUMMARY")
        print("=" * 60)

        print(f"{'Model':<22}: {self.model_name}")
        print(f"{'Prompt Tokens':<22}: {response.usage.prompt_tokens:,}")
        print(f"{'Completion Tokens':<22}: {response.usage.completion_tokens:,}")
        print(f"{'Total Tokens':<22}: {response.usage.total_tokens:,}")

        print("-" * 60)

        input_cost = (  response.usage.prompt_tokens / 1000000 ) * 0.15
        output_cost = ( response.usage.completion_tokens / 1000000 ) * 0.60
        total_cost = input_cost + output_cost

        print(f"{'Input Cost':<22}: ${input_cost:.8f}")
        print(f"{'Output Cost':<22}: ${output_cost:.8f}")
        print(f"{'Total LLM Cost':<22}: ${total_cost:.8f}")

        print("=" * 60)

        tool_call = response.choices[0].message.tool_calls[0]

        intent_json = json.loads(tool_call.function.arguments)

        return intent_json


    def generate_sql(self, intent, scoring_vintage):

        entity = self.semantic_map["entities"][intent["entity"]]
        metric = self.semantic_map["metrics"][intent["metric"]]
        group_by = self.semantic_map["dimensions"][intent["group_by"]]
        filter_col = self.semantic_map["dimensions"][intent["filter_key"]]
        filter_val = intent["filter_val"]

        sql = f"""
        SELECT
            {group_by},
            {metric} AS metric_value
        FROM {entity}
        WHERE {filter_col} = '{filter_val}'
          AND date(substr({group_by}, 1, 4) || '-' || substr({group_by}, 5, 2) || '-01')
              BETWEEN date(substr('{scoring_vintage}', 1, 4) || '-' || substr('{scoring_vintage}', 5, 2) || '-01', '-5 months')
                  AND date(substr('{scoring_vintage}', 1, 4) || '-' || substr('{scoring_vintage}', 5, 2) || '-01')
        GROUP BY {group_by}
        ORDER BY {group_by};
        """

        return sql

    def execute_sql(self, sql):

        result = pd.read_sql_query(sql, self.conn)

        return result

    def run_all(self, question, scoring_vintage):

        intent = self.parse_user_question(question)

        sql = self.generate_sql(intent, scoring_vintage)

        result = self.execute_sql(sql)

        return intent, sql, result


##### Step 12: Executive Summary Agent class

class _12_ExecutiveSummaryAgent:

    def __init__(
        self,
        validation_report,
        scoring_report,
        performance_report,
        route_result,
        recommendation_report,
        approval_report=None,
        sql_results=None
    ):
        self.validation_report = validation_report
        self.scoring_report = scoring_report
        self.performance_report = performance_report
        self.route_result = route_result
        self.recommendation_report = recommendation_report
        self.approval_report = approval_report
        self.sql_results = sql_results

    def summarize_validation(self):
        return self.validation_report["Status"].value_counts().to_dict()

    def summarize_performance(self):
        issues = self.performance_report[
            self.performance_report["Status"].isin(["WARNING", "FAIL"])
        ]
        return issues


    def generate_summary(self):

        validation_counts = self.summarize_validation()
        issues = self.summarize_performance()

        scoring = self.scoring_report.iloc[0].to_dict()

        def get_value(*keys):
            for key in keys:
                if key in scoring and scoring[key] is not None:
                    return scoring[key]
            return None

        vintage = get_value("Vintage", "vintage")
        count = get_value("Count", "count")
        status = get_value("Status", "status")

        avg_pd = get_value("Avg_PD", "avg_pd")
        precision = get_value("Precision", "precision")
        recall = get_value("Recall", "recall")
        f1 = get_value("F1", "f1")

        summary = []

        summary.append(
            f"The scoring pipeline completed for vintage {vintage} "
            f"with status {status} and {count} records scored."
        )

        summary.append(
            f"Data validation produced {validation_counts.get('PASS', 0)} PASS, "
            f"{validation_counts.get('WARNING', 0)} WARNING, and "
            f"{validation_counts.get('FAIL', 0)} FAIL checks."
        )

        if avg_pd is not None:
            summary.append(
                f"Model performance showed average PD of {avg_pd:.4f}, "
                f"precision of {precision:.4f}, "
                f"recall of {recall:.4f}, and "
                f"F1 of {f1:.4f}."
            )
        else:
            summary.append(
                "Model scoring did not produce valid PD/performance metrics, likely because scoring was skipped or failed."
            )

        if issues.empty:
            summary.append(
                "No material model performance issue was detected."
            )
        else:
            issue_names = ", ".join(issues["Metric"].tolist())
            summary.append(
                f"Performance issues were detected in: {issue_names}."
            )

        summary.append(
            f"The conditional router decision was: {self.route_result.get('route_decision')} "
            f"because {self.route_result.get('route_reason')}."
        )

        return " ".join(summary)

    def run_all(self):

        executive_summary = self.generate_summary()

        return pd.DataFrame([{
            "Executive Summary": executive_summary
        }])





