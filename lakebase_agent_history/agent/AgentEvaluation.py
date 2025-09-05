# Databricks notebook source

# COMMAND ----------

##################################################################################
# Agent Evaluation
# 
# Notebook that downloads an evaluation dataset and evaluates the model using
# llm-as-a-judge with the Databricks agent framework.
#
# Parameters:
# * uc_catalog (required)           - Name of the Unity Catalog 
# * schema (required)               - Name of the schema inside Unity Catalog 
# * experiment (required)           - Name of the experiment to register the run under
# * registered_model (required)     - Name of the model registered in mlflow
# * model_alias (required)          - Model alias to deploy
# * bundle_root (required)          - Root of the bundle
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Experiment: Text widget to input the name of the experiment to register the run under
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Model Alias: Text widget to input the model alias to deploy
# * Bundle root: Text widget to input the root of the bundle
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run to evaluate your agent.
#
##################################################################################

# COMMAND ----------

# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.

# A Unity Catalog containing the model
dbutils.widgets.text(
    "uc_catalog",
    "ai_agent_stacks",
    label="Unity Catalog",
)
# Name of schema
dbutils.widgets.text(
    "schema",
    "ai_agent_ops",
    label="Schema",
)
# Name of experiment to register under in mlflow
dbutils.widgets.text(
    "experiment",
    "agent_function_chatbot",
    label="Experiment name",
)
# Name of model registered in mlflow
dbutils.widgets.text(
    "registered_model",
    "agent_function_chatbot",
    label="Registered model name",
)
# Model alias
dbutils.widgets.text(
    "model_alias",
    "agent_latest",
    label="Model Alias",
)

# Bundle root
dbutils.widgets.text(
    "bundle_root",
    "/",
    label="Root of bundle",
)

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 databricks-agents pydantic psycopg2 databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
experiment = dbutils.widgets.get("experiment")
registered_model = dbutils.widgets.get("registered_model")
model_alias = dbutils.widgets.get("model_alias")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert experiment != "", "experiment notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert model_alias != "", "model_alias notebook parameter must be specified"

# Updating to bundle root
import sys 

root = dbutils.widgets.get("bundle_root")
sys.path.append(root)

# COMMAND ----------

# DBTITLE 1,Create Evaluation Dataset
# import mlflow.genai.datasets

# try: 
#     eval_dataset = mlflow.genai.datasets.create_dataset(
#         uc_table_name=f"{uc_catalog}.{schema}.{eval_table}",
#     )
# except: 
#     # Eval table already exists 
#     eval_dataset = mlflow.genai.datasets.get_dataset(
#         uc_table_name=f"{uc_catalog}.{schema}.{eval_table}",
#     )

# print(f"Evaluation dataset: {uc_catalog}.{schema}.{eval_table}")

# COMMAND ----------

# DBTITLE 1,Run Evaluation
# import mlflow 
# from mlflow.genai.scorers import scorer
# from mlflow.genai.scorers import RetrievalRelevance, RetrievalGroundedness
# import re

# # Workaround for serverless compatibility
# mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = lambda: "databricks-uc"

# model = mlflow.pyfunc.load_model(f"models:/{uc_catalog}.{schema}.{registered_model}@{model_alias}")
# def evaluate_model(question):
#     return model.predict({"messages": [{"role": "user", "content": question}]})

# mlflow.set_experiment(experiment)

# with mlflow.start_run():
#     # Evaluate the logged model
#     eval_results = mlflow.genai.evaluate(
#         data=eval_dataset,
#         predict_fn=evaluate_model,
#         scorers=[RetrievalRelevance(), RetrievalGroundedness()],
#     )