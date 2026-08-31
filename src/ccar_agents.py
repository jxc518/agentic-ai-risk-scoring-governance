import pandas as pd
import joblib

import logging
logger = logging.getLogger("CCAR_RISK_SCORING")

from ccar_core import (
    step_1_uploading,
    _02_DataValidator,
    _03_FeatureEngineer,
    _04_Model_Scorer,
    _05_PerformanceEvaluator,
    _06_ConditionalRouter,
    _07_ReportingAgent,
    _08_RAG_Agent,
    _09_RecommendationAgent,
    _10_HumanApproval,
    _11_DeterministicSQLAgent,
    _12_ExecutiveSummaryAgent
)


##### _01_agent_upload

DATA_STORE = {}

# def _01_agent_upload(state: AgentState):
def _01_agent_upload(state):

    logger.info("-" * 60)
    logger.info("AGENT 01 - Agent Data Uploading")

    try:
        # ========================================================
        # Normal Processing
        # ========================================================

        df_uploaded, upload_report = step_1_uploading(
            input_path=state["input_file"],
            var_list=state["numeric_columns"]
        )

        dataset_id = "uploaded_df"
        DATA_STORE[dataset_id] = df_uploaded

        state["dataset_id"] = dataset_id
        state["upload_report"] = upload_report

        # Add this part
        df_uploaded.to_sql(
            "data_1_uploaded_db",
            state["sql_conn"],
            if_exists="replace",
            index=False
        )

        state["upload_summary"] = {
            "rows": len(df_uploaded),
            "columns": len(df_uploaded.columns),
            "memory_mb": round(df_uploaded.memory_usage(deep=True).sum() / 1024**2, 2)
        }

        # Successful pipeline status
        state["pipeline_status"] = "RUNNING"

        # ========================================================
        # Logging
        # ========================================================
        logger.info(f"Input type          : {type(state['input_file']).__name__}")
        logger.info(f"Records Uploaded    : {state['upload_summary']['rows']:,}")
        logger.info(f"Attribute Columns   : {state['upload_summary']['columns']}")
        logger.info(f"Memory MB           : {state['upload_summary']['memory_mb']}")
        logger.info("Status         : PASS")

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_01_DATA_UPLOADING"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)
        state["pipeline_status"] = "FAILED"

        logger.error(f"Error Type          : {type(e).__name__}")
        logger.error(f"Error Message       : {str(e)}")
        logger.error("Status              : FAIL")
   
        raise


##### _02_agent_DataValidator

# def _02_agent_DataValidator(state: AgentState):
# def _02_agent_DataValidator(state):

#     df_uploaded = DATA_STORE[state["dataset_id"]]

#     validator = _02_DataValidator(
#         df=df_uploaded,
#         required_columns=state["required_columns"],
#         numeric_columns=state["numeric_columns"],
#         target_col=state["target_col"]
#     )

#     state["validation_report"] = validator.run_all()

#     return state

def _02_agent_DataValidator(state):

    logger.info("-" * 60)
    logger.info("AGENT 02 - DATA VALIDATION")

    try:
        # ========================================================
        # Normal Processing
        # ========================================================

        # Get uploaded DataFrame from DATA_STORE
        df_uploaded = DATA_STORE[state["dataset_id"]]

        # Create DataValidator object
        validator = _02_DataValidator(
            df=df_uploaded,
            required_columns=state["required_columns"],
            numeric_columns=state["numeric_columns"],
            target_col=state["target_col"]
        )

        # Run all validation checks
        validation_report = validator.run_all()

        # Save validation report into LangGraph state
        state["validation_report"] = validation_report

        # Count validation results
        pass_count = (validation_report["Status"] == "PASS").sum()
        warning_count = (validation_report["Status"] == "WARNING").sum()
        fail_count = (validation_report["Status"] == "FAIL").sum()

        # Successful pipeline status
        state["pipeline_status"] = "RUNNING"

        # ========================================================
        # Logging
        # ========================================================
        logger.info(f"Records Validated   : {len(df_uploaded):,}")
        logger.info(f"Columns Validated   : {len(df_uploaded.columns)}")
        logger.info(f"Validation Checks   : {len(validation_report)}")
        logger.info(f"PASS                : {pass_count}")
        logger.info(f"WARNING             : {warning_count}")
        logger.info(f"FAIL                : {fail_count}")

        if fail_count > 0:
            logger.error("Status              : FAIL")
        elif warning_count > 0:
            logger.warning("Status              : WARNING")
        else:
            logger.info("Status              : PASS")

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_02_DATA_VALIDATION"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)
        state["pipeline_status"] = "FAILED"
   
        logger.error(f"Error Type          : {type(e).__name__}" )
        logger.error(f"Error Message      : {str(e)} " )
        logger.error("Status               : FAIL")

        raise 
         

##### _03_agent_FeatureEngineer

# def _03_agent_FeatureEngineer(state: AgentState):
def _03_agent_FeatureEngineer(state):

    logger.info("-" * 60)
    logger.info("AGENT 03 - Feature Engineering")

    try:
        # ========================================================
        # Normal Processing
        # ========================================================

        df_uploaded = DATA_STORE[state["dataset_id"]]

        feature_engineer = _03_FeatureEngineer(
            df=df_uploaded
        )

        df_features = feature_engineer.run_all()

        feature_dataset_id = "feature_df"
        DATA_STORE[feature_dataset_id] = df_features

        state["feature_dataset_id"] = feature_dataset_id

        state["feature_summary"] = {
            "rows": len(df_features),
            "columns": len(df_features.columns),
            "memory_mb": round(df_features.memory_usage(deep=True).sum() / 1024**2, 2)
        }

        # Logging
        logger.info(f"Input Rows          : {len(df_uploaded):,}")
        logger.info(f"Input Columns       : {len(df_uploaded.columns)}")
        logger.info(f"Output Rows         : {len(df_features):,}")
        logger.info(f"Output Columns      : {len(df_features.columns)}")
        logger.info(f"Features Added      : {len(df_features.columns) - len(df_uploaded.columns)}")
        logger.info(f"Output Memory MB    : {state['feature_summary']['memory_mb']}")
        logger.info("Status              : RUNNING")

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_03_FEATURE_ENGINEERING"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)
        state["pipeline_status"] = "FAILED"   

        logger.error(f"Error Type          : {type(e).__name__}")
        logger.error(f"Error Message       : {str(e)}")
        logger.error("Status              : FAIL")

        raise


##### _04_agent_Model_Scorer

# def _04_agent_Model_Scorer(state: AgentState):
def _04_agent_Model_Scorer(state):

    logger.info("-" * 60)
    logger.info("AGENT 04 - Model Scoring")

    # print("Agent 4 state keys:", state.keys())
    # print("feature_dataset_id:", state.get("feature_dataset_id"))

    try:
        # ========================================================
        # Normal Processing
        # ========================================================        

        df_features = DATA_STORE[state["feature_dataset_id"]]

        scoring_vintage = str(state["scoring_vintage"])

        available_vintages = (
            df_features["report_yrmo"]
            .astype(str)
            .unique()
        )

        if scoring_vintage not in available_vintages:
            raise ValueError(
                f"Scoring vintage {scoring_vintage} not found in input data."
            )

        model_scorer = _04_Model_Scorer(
            df=df_features,
            model=joblib.load(state["model_path"]),
            x_list=state["model_x_list"],
            y_target=state["target_col"],
            pd_threshold=state["pd_threshold"]
        )

        df_scored, scoring_report = model_scorer.run_all(
            vintage=state["scoring_vintage"]
        )

        # print("Agent 4 scoring_report:")
        # display(scoring_report)

        state["scoring_report"] = scoring_report

        # ========================================================
        # Critical scoring failure
        # ========================================================
        if df_scored is None:
            state["scored_dataset_id"] = None
            state["scoring_summary"] = {
                "rows": 0,
                "columns": 0,
                "memory_mb": 0,
                "status": scoring_report.iloc[0]["Status"]
            }
            # return state

            raise RuntimeError(
                f"Model scoring failed: {scoring_report.iloc[0]['Status']}"
            )

        # DATA_STORE["df_scored"] = df_scored
        # state["scored_dataset_id"] = "df_scored"

        # ========================================================
        # Save scored dataset
        # ========================================================
        scored_dataset_id = f"df_scored_{state['scoring_vintage']}"
        DATA_STORE[scored_dataset_id] = df_scored
        state["scored_dataset_id"] = scored_dataset_id

        state["scoring_summary"] = {
            "rows": len(df_scored),
            "columns": len(df_scored.columns),
            "memory_mb": round(df_scored.memory_usage(deep=True).sum() / 1024**2, 2),
            "status": "Success"
        }

        state["pipeline_status"] = "RUNNING"

        # print("Feature data shape:", df_features.shape)
        # print("Available vintages:")
        # print(df_features["report_yrmo"].astype(str).value_counts().sort_index().tail(20))

        # Logging
        logger.info(f"Feature data shape  : {df_features.shape}")
        logger.info("Available vintages:")
        logger.info(df_features["report_yrmo"].astype(str).value_counts().sort_index().tail(20))
        logger.info(f"Scored Records      : {len(df_scored):,}")
        logger.info(f"Output Columns      : {len(df_scored.columns)}")
        logger.info(f"Scored Dataset ID   : {state.get('scored_dataset_id')}")
        logger.info(f"Agent 4 DATA_STORE keys: { list(DATA_STORE.keys()) }")    
        logger.info("Status              : PASS")

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_04_MODEL_SCORING"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)
        state["pipeline_status"] = "FAILED"   

        logger.error(f"Error Type          : {type(e).__name__}")
        logger.error(f"Error Message       : {str(e)}")
        logger.error("Status               : FAIL")

        raise


##### _05_agent_PerformanceEvaluator

# Use state[...] for values that may change between runs or environments.
# Use literal strings like "pd_score" for fixed column names that are part of my pipeline design.

# def _05_agent_PerformanceEvaluator(state: AgentState):
def _05_agent_PerformanceEvaluator(state):

    logger.info("-" * 60)
    logger.info("AGENT 05 - Model Performance Evaluation")

    scored_dataset_id = state.get("scored_dataset_id")
    logger.info(f"Scored Dataset ID   : {scored_dataset_id}")


    try:
        # ========================================================
        # Normal Processing
        # ========================================================        

        if scored_dataset_id is None or scored_dataset_id not in DATA_STORE:

            state["performance_report"] = pd.DataFrame([{
                "Metric": "Performance Evaluation",
                "Status": "SKIPPED",
                "Value": None,
                "Details": (
                    "No scored dataframe available from Agent 4. "
                    f"scored_dataset_id={scored_dataset_id}; "
                    f"available DATA_STORE keys={list(DATA_STORE.keys())}"
                )
            }])

            logger.warning("Performance Evaluation: SKIPPED")
            logger.warning("Reason              : No scored dataframe available")
            logger.warning("Status              : SKIPPED")

            return state

        df_scored = DATA_STORE[scored_dataset_id]

        state["pipeline_status"] = "RUNNING"

        logger.info(f"Records Evaluated   : {len(df_scored):,}")
        logger.info(f"Input Columns       : {len(df_scored.columns)}")

        required_cols = [
            state["target_col"],
            "pd_score",
            "pred_flag"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df_scored.columns
        ]

        if missing_cols:

            state["performance_report"] = pd.DataFrame([{
                "Metric": "Performance Evaluation",
                "Status": "SKIPPED",
                "Value": None,
                "Details": f"Scored dataframe missing required columns: {missing_cols}"
            }])

            logger.warning(f"Missing Columns     : {missing_cols}")
            logger.warning("Performance Evaluation: SKIPPED")
            logger.warning("Status              : SKIPPED")

            return state

        perf_evaluator = _05_PerformanceEvaluator(
            df_scored,
            state["target_col"],
            "pd_score",
            "pred_flag"
        )

        performance_report = perf_evaluator.run_all()

        state["performance_report"] = performance_report

        # print("Agent 5 received scored_dataset_id:", state.get("scored_dataset_id"))
        # print("Agent 5 DATA_STORE keys:", list(DATA_STORE.keys()))

        # --------------------------------------------------------
        # Summarize performance results
        # --------------------------------------------------------
        logger.info(f"Performance Metrics : {len(performance_report)}")

        pass_count = (    performance_report["Status"] == "PASS" ).sum()
        warning_count = (     performance_report["Status"] == "WARNING"  ).sum()
        fail_count = (      performance_report["Status"] == "FAIL"   ).sum()

        logger.info(f"PASS                : {pass_count}")
        logger.info(f"WARNING             : {warning_count}")
        logger.info(f"FAIL                : {fail_count}")

        # --------------------------------------------------------
        # Determine overall Agent 5 status
        # --------------------------------------------------------
        if fail_count      > 0:   logger.error("Status              : FAIL")
        elif warning_count > 0:   logger.warning("Status              : WARNING")
        else:                     logger.info("Status              : PASS")

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_05_MODEL_PERFORMANCE"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)
        state["pipeline_status"] = "FAILED"
   
        logger.error(f"Error Type          : {type(e).__name__}" )
        logger.error(f"Error Message      : {str(e)} " )
        logger.error("Status               : FAIL")

        raise 


##### _06_agent_router

# def _06_agent_conditional_router(state: AgentState):
def _06_agent_conditional_router(state):

    logger.info("-" * 60)
    logger.info("AGENT 06 - Conditional Router")    

    try:
        # ========================================================
        # Normal Processing
        # ========================================================   

        # print("validation_report columns:", state.get("validation_report").columns.tolist())
        # print("scoring_report columns:", state.get("scoring_report").columns.tolist())
        # print("performance_report columns:", state.get("performance_report").columns.tolist())
        print("validation_report:")
        display(state.get("validation_report"))
        print("scoring_report:")
        display(state.get("scoring_report"))
        print("performance_report:")
        display(state.get("performance_report"))

        router = _06_ConditionalRouter(
            validation_report=state["validation_report"],
            scoring_report=state["scoring_report"],
            performance_report=state["performance_report"]
        )

        route_result = router.run_all()
        state["route_result"] = route_result
        # state["route_result"] = router.run_all()

        state["pipeline_status"] = "RUNNING"

        logger.info( f"Route Decision      : {route_result.get('route_decision')}"   )
        logger.info( f"Route Reason        : {route_result.get('route_reason')}"     )
        logger.info( "Status              : PASS")

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_06_CONDITIONAL_ROUTING"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)
        state["pipeline_status"] = "FAILED"
   
        logger.error(f"Error Type          : {type(e).__name__}" )
        logger.error(f"Error Message      : {str(e)} " )
        logger.error("Status               : FAIL")

        raise 


##### _07_agent_reporting

# def debug_before_reporter(state: AgentState):
def debug_before_reporter(state):

    print("\n===== DEBUG BEFORE AGENT 7 =====")
    print("route_result:", state.get("route_result"))
    print("router branch should be:", router_decision(state))
    print("available keys:", list(state.keys()))
    print("has validation_report:", "validation_report" in state)
    print("has scoring_report:", "scoring_report" in state)
    print("has performance_report:", "performance_report" in state)

    return state


# def _07_agent_reporting(state: AgentState):
def _07_agent_reporting(state):

    # print("\n===== AGENT 7 STARTED =====")
    # print("state keys:", list(state.keys()))

    logger.info("-" * 60)
    logger.info("AGENT 07 - Agent Reporting")    

    try:
        reporting_agent = _07_ReportingAgent(
            validation_report=state["validation_report"],
            scoring_report=state["scoring_report"],
            performance_report=state["performance_report"],
            route_result=state["route_result"]
        )

        technical_report = reporting_agent.run_all()

        state["technical_report"] = technical_report
        state["agent7_debug"] = "Agent 7 completed successfully."

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        # state["pipeline_status"] = "RUNNING_WITH_WARNING"
        if technical_report is not None:
            state["pipeline_status"] = "RUNNING"

            logger.info(
                f"Report Length       : {len(str(technical_report)):,} characters"
            )
            logger.info("Status              : PASS")

        else:
            state["pipeline_status"] = "RUNNING_WITH_WARNING"
            logger.warning("Status              : WARNING")

        # print("Agent 7 completed.")
        # print(technical_report)

        return state

    except Exception as e:
        state["error_agent"] = "AGENT_07_REPORTING"
        state["error_type"] = type(e).__name__
        state["error_message"] = str(e)

        state["technical_report"] = None
        state["pipeline_status"] = "RUNNING_WITH_WARNING"

        # --------------------------------------------------------
        # Error Logging
        # --------------------------------------------------------
        logger.error(f"Error               : {str(e)}")
        logger.error("Technical Report    : Not Generated")
        logger.error("Status              : FAIL")

        return state


##### _08_agent_RAG_goverance_monitoring

sample_knowledge_base = [
    "If model recall drops below threshold, investigate data drift, feature distribution shift, and target definition changes.",
    "If AUC or KS deteriorates materially, perform model monitoring review and consider recalibration.",
    "If data validation fails, stop scoring and escalate to data owner.",
    "If predicted bad rate deviates materially from actual bad rate, review calibration and cutoff strategy."
]

# def _08_agent_RAG_governance_monitoring(state: AgentState):
def _08_agent_RAG_governance_monitoring(state):

    # print("\n===== AGENT 8 STARTED =====")
    # print("state keys:", list(state.keys()))

    logger.info("-" * 60)
    logger.info("AGENT 08 -RAG Governance Monitoring")    

    knowledge_base = state.get(  "knowledge_base",   sample_knowledge_base )

    try:
        rag_agent = _08_RAG_Agent(
            route_result=state["route_result"],
            performance_report=state["performance_report"],
            # knowledge_base=state.get("knowledge_base", sample_knowledge_base)
            knowledge_base = knowledge_base
        )

        rag_result = rag_agent.run_all()

        state["rag_result"] = rag_result
        state["_rag_query"] = rag_result.get("rag_query")
        state["_rag_documents"] = rag_result.get("retrieved_guidance")
        state["rag_answer"] = rag_result.get("rag_answer")
        state["agent8_debug"] = "Agent 8 completed successfully."

        # print("Agent 8 completed.")
        # print("rag_result:", rag_result)

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------
        logger.info(f"Knowledge Documents : {len(knowledge_base)}")
        logger.info(
            f"RAG Report    : "
            f"{'Generated' if rag_result is not None else 'Not Generated'}" )

        logger.info(f"RAG Query           : {state.get('_rag_query')}")
        logger.info(
            f"Retrieved Guidance  : "
            f"{'Available' if state.get('_rag_documents') is not None else 'Not Available'}"
        )
        logger.info(
            f"RAG Answer          : "
            f"{'Generated' if state.get('rag_answer') else 'Not Generated'}"
        )
        logger.info("Status              : PASS")

    except Exception as e:
        state["rag_result"] = None
        state["rag_answer"] = None
        state["agent8_debug"] = f"Agent 8 failed: {str(e)}"
        print("Agent 8 failed:", str(e))

        # --------------------------------------------------------
        # Error Logging
        # --------------------------------------------------------
        logger.error(f"Error               : {str(e)}")
        logger.error("RAG Report    : Not Generated")
        logger.error("Status              : FAIL")

    return state


##### _09_agent_recommendation

# def _09_agent_recommendation(state: AgentState):
def _09_agent_recommendation(state):

    logger.info("-" * 60)
    logger.info("AGENT 09 -Recommendation")  

    recommender = _09_RecommendationAgent(
        route_result=state["route_result"],
        performance_report=state["performance_report"],
        rag_answer=state.get("rag_answer")
    )

    recommendation_report = recommender.run_all()

    state["recommendation_report"] = recommendation_report

    logger.info(
        f"Recommendation Report Status : "
        f"{'Available' if recommendation_report is not None else 'Not Available'}"
    )
    
    if recommendation_report is not None:
        logger.info("Status                : PASS")
    else:
        logger.warning("Status                : WARNING")

    return state


##### _10_agent_human_approval

# def _10_agent_human_approval(state: AgentState):
def _10_agent_human_approval(state):

    logger.info("-" * 60)
    logger.info("AGENT 10 - Human Approval")  

    approval_agent = _10_HumanApproval(
        recommendation_report=state["recommendation_report"]
    )

    approval_report = approval_agent.run_all()

    state["approval_status"] = approval_report.loc[0, "Approval Decision"]
    state["approval_reason"] = approval_report.loc[0, "Details"]

    logger.info( f"Approval Report Status:"
                f"{'Available' if approval_report is not None else 'Not Available' } "  )
    
    logger.info( f"Approval Status : {state.get( 'approval_status') }")
    logger.info( f"Approval Reason : {state.get( 'approval_reason') }")
    
    if approval_report is not None:
        logger.info( f"Status             : PASS " )
    else:
        logger.error("Status             : FAIL")

    return state


##### _11_agent_DeterministicSQLAgent

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


# def _11_agent_DeterministicSQLAgent(state: AgentState):
def _11_agent_DeterministicSQLAgent(state):

    logger.info("-" * 60)
    logger.info("AGENT 11 - Deterministic SQL")  

    sql_agent = _11_DeterministicSQLAgent(
        client=state["openai_client"],
        model_name=state["llm_model_name"],
        semantic_map=state.get("semantic_map", semantic_map ),
        conn=state["sql_conn"]
    )

    sql_intent, sql_query, sql_result = sql_agent.run_all(
        question=state["sql_user_query"],
        scoring_vintage=state["scoring_vintage"]
    )

    state["sql_intent"] = sql_intent
    state["sql_query"] = sql_query
    state["sql_result"] = sql_result

    logger.info( f"SQL Intent :               { state.get('sql_intent') } ")
    logger.info( f"SQL Query  :               { state.get('sql_query') } ")
    logger.info( f"SQL Result :               { state.get('sql_result') } ") 
                
    # logger.info( f"{ 'PASS' if state.get('sql_result') is not None else 'FAIL'}")

    if state.get("sql_result") is not None:
        logger.info("Status     : PASS")
    else:
        logger.error("Status     : FAIL")

    return state


##### _12_agent_ExecutiveSummaryAgent

# def _12_agent_ExecutiveSummaryAgent(state: AgentState):
def _12_agent_ExecutiveSummaryAgent(state):

    logger.info("-" * 60)
    logger.info("AGENT 12 - Executive Summary")  

    exec_agent = _12_ExecutiveSummaryAgent(
        validation_report = state["validation_report"],
        scoring_report    = state["scoring_report"],
        performance_report= state["performance_report"],
        route_result      = state["route_result"],
        recommendation_report=state["recommendation_report"],
        approval_report=state.get("approval_report", pd.DataFrame([{
            "Status": state.get("approval_status", "UNKNOWN"),
            "Reason": state.get("approval_reason", "")
        }])) ,
        sql_results       =state["sql_result"]
    )

    executive_summary = exec_agent.run_all()

    state["executive_summary"] = executive_summary

    logger.info( f"Validation Report      :   { state.get('validation_report') } " )
    logger.info( f"Scoring Report         :   { state.get('scoring_report')  } " )
    logger.info( f"Performance Report     :   { state.get('performance_report') }" )
    logger.info( f"Route Result           :   {state.get('route_result') } " )
    logger.info( f"Recommendation Report  :  {state.get('recommendation_report') }" )

    if state.get("executive_summary")  is not None:
        logger.info( "Sttus    ：   PASS")
    else:
        logger.error("Status    ：   FAIL")

    return state

