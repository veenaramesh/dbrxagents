# Databricks notebook source

# COMMAND ----------

################################################################################### 
# Agent Chain Creation
#
# This notebook shows an example of a Agent that uses Lakebase to store conversation history.
#
# Parameters:
# * uc_catalog (required)                     - Name of the Unity Catalog 
# * schema (required)                         - Name of the schema inside Unity Catalog 
# * experiment (required)                     - Name of the experiment to register the run under
# * registered_model (required)               - Name of the model to register in mlflow
# * model_alias (required)                    - Model alias for latest model version
# * bundle_root (required)                    - Root of the bundle
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Experiment: Text widget to input the name of the experiment to register the run under
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Model Alias: Text widget to input the alias of the model to register in mlflow
# * Bundle root: Text widget to input the root of the bundle
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Run the pipeline to create and register an agent with tool calling.
#
##################################################################################

# COMMAND ----------

# List of input args needed to run this notebook as a job
# Provide them via DB widgets or notebook arguments

# A Unity Catalog containing the preprocessed data
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
    "agent_lakebase",
    label="Experiment name",
)
# Name of model to register in mlflow
dbutils.widgets.text(
    "registered_model",
    "agent_lakebase",
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
bundle_root = dbutils.widgets.get("bundle_root")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert experiment != "", "experiment notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert model_alias != "", "model_alias notebook parameter must be specified"
assert bundle_root != "", "bundle_root notebook parameter must be specified"

# Updating to bundle root
import sys 
sys.path.append(bundle_root)

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from pkg_resources import get_distribution

mlflow.set_experiment(experiment)

resources = [
    DatabricksServingEndpoint(endpoint_name="databricks-meta-llama-3-3-70b-instruct"), 
    DatabricksFunction("system.ai.python_exec")
]

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model="app.py", # Pass the path to the saved model file
        name="model",
        resources=resources, 
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"unitycatalog-langchain[databricks]=={get_distribution('unitycatalog-langchain[databricks]').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"psycopg2-binary=={get_distribution('psycopg2-binary').version}", 
            f"databricks-sdk=={get_distribution('databricks-sdk').version}"
        ],
    )

# COMMAND ----------

from mlflow import MlflowClient

client_mlflow = MlflowClient()

registered_model_name = f"{uc_catalog}.{schema}.{registered_model}"
uc_registered_model_info = mlflow.register_model(
  model_info.model_uri, 
  name=registered_model_name
  )

