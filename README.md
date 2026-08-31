
# Agentic AI Risk Scoring & Model Governance

An engineering-grade Agentic AI framework for credit risk scoring, model governance, monitoring, and decision support using LangGraph and Databricks.

## Project Overview

This project demonstrates an end-to-end 12-agent workflow for credit risk scoring and model governance.

The original prototype was developed in Google Colab and subsequently migrated and engineered on Databricks using Unity Catalog, Delta Lake, PySpark, modular Python components, Databricks Secrets, structured logging, exception handling, failure-path validation, token/cost tracking, and LangSmith observability.

The project is designed as a proof of concept (POC) demonstrating how Agentic AI can orchestrate machine learning, governance, deterministic analytics, human review, and executive reporting within a unified workflow.

## Architecture

The Databricks implementation follows this high-level architecture:

**Data → Unity Catalog / Delta Lake → PySpark → LangGraph 12-Agent Workflow → Risk & Governance Outputs**

Key platform and engineering components include:

- Databricks
- Unity Catalog
- Delta Lake
- PySpark
- LangGraph
- LangSmith
- OpenAI
- Scikit-learn
- Pandas

## 12-Agent Workflow

The workflow includes:

1. Data Upload
2. Data Validation
3. Feature Engineering
4. Model Scoring
5. Performance Evaluation
6. Conditional Routing
7. Technical Reporting
8. Governance RAG
9. AI Recommendation
10. Human-in-the-Loop Approval
11. Deterministic SQL Analytics
12. Executive Summary

The LangGraph workflow supports conditional routing based on model validation and performance results.

## Engineering Enhancements on Databricks

The Databricks Engineering POC extends the original functional prototype with:

- Colab-to-Databricks migration
- Unity Catalog and Delta Lake integration
- PySpark data integration
- Modular Python architecture
- Centralized configuration
- Databricks Secrets
- Structured logging
- Exception handling
- Failure-path validation
- Agent-level execution metrics
- LLM token and cost tracking
- LangSmith end-to-end observability
- Clean Engineering Full Run validation

## Project Structure

```text
agentic-ai-risk-scoring-governance/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── ccar_config.py
│   ├── ccar_core.py
│   ├── ccar_agents.py
│   └── ccar_graph.py
│
├── notebooks/
│   └── CCAR_risk_scoring_Main_Databricks_for_multi_agent.ipynb
│
└── docs/
    ├── architecture/
    ├── observability/
    └── presentation/



## Engineering Validation

The final Databricks Engineering POC successfully completed an end-to-end clean engineering run.

Validation included:

12-agent workflow execution
Data validation
ML model scoring
Performance monitoring
Conditional workflow routing
Governance RAG
Human-in-the-loop control
Deterministic SQL analytics
Executive reporting
LangSmith runtime tracing
Token and cost tracking
Failure-path testing

Final Status: Clean Engineering Full Run — PASS

## Observability

LangSmith is used for end-to-end workflow observability, including:

Full LangGraph execution tracing
Agent/node execution paths
Agent latency
Error tracing
Token usage
LLM cost tracking

## Documentation

Architecture diagrams, workflow diagrams, runtime observability evidence, and the project presentation are available under the docs/ directory.

## Security

API keys and credentials are not stored in source code.

The Databricks implementation uses Databricks Secrets for protected credentials. Sensitive datasets, model artifacts, secret values, and local environment files are excluded from this public repository.

## Disclaimer

This repository is a personal engineering proof of concept for demonstration and educational purposes.

The data and examples are used to demonstrate technical architecture and workflow design. This repository is not a production banking system and should not be interpreted as providing financial, credit, or regulatory decisions.
