# Databricks notebook source

# COMMAND ----------

##################################################################################
# Agent Deployment
# 
# Deploy the agent via agents.deploy()
#
# Parameters:
# * uc_catalog (required)                       - Name of the Unity Catalog 
# * schema (required)                           - Name of the schema inside Unity Catalog 
# * registered_model (required)                 - Name of the model registered in mlflow
# * model_alias (required)                      - Model alias to deploy
# * bundle_root (required)                      - Root of the bundle
# * databricks_host (required)                  - Databricks host
# * databricks_secret_scope (required)          - Databricks secret scope
# * databricks_client_id_key (required)         - Databricks client id key
# * databricks_client_secret_key (required)     - Databricks client secret key
# * databricks_sp_application_id_key (required) - Databricks sp application id key
#
# Widgets:
# * Unity Catalog: Text widget to input the name of the Unity Catalog
# * Schema: Text widget to input the name of the database inside the Unity Catalog
# * Registered model name: Text widget to input the name of the model to register in mlflow
# * Model Alias: Text widget to input the model alias to deploy
# * Scale to zero: Whether the clusters should scale to zero (requiring more time at startup after inactivity)
# * Workload Size: Compute that matches estimated number of requests for the endpoint
# * Bundle root: Text widget to input the root of the bundle
#
# Usage:
# 1. Set the appropriate values for the widgets.
# 2. Add members that you want to grant access to for the review app to the user_list.
# 3. Run to deploy endpoint.
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

# Databricks host
dbutils.widgets.text(
    "databricks_host",
    "https://e2-demo-field-eng.cloud.databricks.com",
    label="Databricks host",
)
# Databricks secret scope
dbutils.widgets.text(
    "databricks_secret_scope",
    "veena-ramesh",
    label="Databricks secret scope",
)
# Databricks client id key
dbutils.widgets.text(
    "databricks_client_id_key",
    "sp_client_id",
    label="Databricks client id key",
)
# Databricks client secret key
dbutils.widgets.text(
    "databricks_client_secret_key",
    "sp_secret",
    label="Databricks client secret key",
)
# Databricks sp application id key
dbutils.widgets.text(
    "databricks_sp_application_id_key",
    "sp_application_id",
    label="Databricks sp application id key",
)   
# Lakebase instance
dbutils.widgets.text(
    "lakebase_instance",
    "veena-test",
    label="Lakebase instance",
)

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 databricks-agents pydantic psycopg2 databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

uc_catalog = dbutils.widgets.get("uc_catalog")
schema = dbutils.widgets.get("schema")
registered_model = dbutils.widgets.get("registered_model")
model_alias = dbutils.widgets.get("model_alias")
databricks_host = dbutils.widgets.get("databricks_host")
databricks_secret_scope = dbutils.widgets.get("databricks_secret_scope")
databricks_client_id_key = dbutils.widgets.get("databricks_client_id_key")
databricks_client_secret_key = dbutils.widgets.get("databricks_client_secret_key")
databricks_sp_application_id_key = dbutils.widgets.get("databricks_sp_application_id_key")
lakebase_instance = dbutils.widgets.get("lakebase_instance")

assert uc_catalog != "", "uc_catalog notebook parameter must be specified"
assert schema != "", "schema notebook parameter must be specified"
assert registered_model != "", "registered_model notebook parameter must be specified"
assert model_alias != "", "model_alias notebook parameter must be specified"
assert databricks_host != "", "databricks_host notebook parameter must be specified"
assert databricks_secret_scope != "", "databricks_secret_scope notebook parameter must be specified"
assert databricks_client_id_key != "", "databricks_client_id_key notebook parameter must be specified"
assert databricks_client_secret_key != "", "databricks_client_secret_key notebook parameter must be specified"
assert databricks_sp_application_id_key != "", "databricks_sp_application_id_key notebook parameter must be specified"
assert lakebase_instance != "", "lakebase_instance notebook parameter must be specified"


# COMMAND ----------
# DBTITLE 1,Create agent deployment

from databricks import agents
from mlflow import MlflowClient

client = MlflowClient()

model_name = f"{uc_catalog}.{schema}.{registered_model}"
model_version = client.get_model_version_by_alias(model_name, model_alias).version

deployment_info = agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    scale_to_zero=True, 
    workload_size="Small", 
    environment_vars={
        "LAKEBASE_INSTANCE": lakebase_instance,
        "DATABRICKS_HOST": databricks_host, 
        "DATABRICKS_CLIENT_ID": dbutils.secrets.get(databricks_secret_scope, databricks_client_id_key), 
        "DATABRICKS_CLIENT_SECRET": dbutils.secrets.get(databricks_secret_scope, databricks_client_secret_key)
        "DATABRICKS_SP_APPLICATION_ID": dbutils.secrets.get(databricks_secret_scope, databricks_sp_application_id_key)
    }
)

# COMMAND ----------
# DBTITLE 1, Wait for model serving endpoint to be ready

# DBTITLE 1,Test Endpoint
from mlflow.deployments import get_deploy_client
from agent_deployment.model_serving.serving import wait_for_model_serving_endpoint_to_be_ready

wait_for_model_serving_endpoint_to_be_ready(deployment_info.endpoint_name)

client = get_deploy_client()
input_example = {
    "messages": [{"role": "user", "content": "What is MLflow?"}],
    "databricks_options": {"return_trace": True},
}

response = client.predict(endpoint=deployment_info.endpoint_name, inputs=input_example)

print(response['messages'][-1]['content'])

