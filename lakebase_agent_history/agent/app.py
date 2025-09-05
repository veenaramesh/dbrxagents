import json
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict, Union, List, Dict
from uuid import uuid4
from datetime import datetime
import os 

import psycopg2
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from databricks.sdk import WorkspaceClient

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

############################################
# Define your LLM endpoint and system prompt
############################################

LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

system_prompt = "You are a helpful assistant that can run Python code."

ONLINE_STORE = os.environ["LAKEBASE_INSTANCE"]
CATALOG_NAME = "databricks_postgres" # "memory"  

############################################
# Database configuration
############################################

class ConversationDB:
    def __init__(self, online_store: str, catalog_name: str):
        self.online_store = online_store
        self.catalog_name = catalog_name
        self.conn = None
        self._connect()
        self._create_table_if_not_exists()
    
    def _connect(self):
        """Establish database connection using Databricks workspace client"""

        workspace = WorkspaceClient(
            host=os.environ["DATABRICKS_HOST"],
            client_id=os.environ["DATABRICKS_CLIENT_ID"],
            client_secret=os.environ["DATABRICKS_CLIENT_SECRET"]
            )
            
        instance = workspace.database.get_database_instance(name=self.online_store)
        cred = workspace.database.generate_database_credential(
            request_id=str(uuid4()), 
            instance_names=[self.online_store]
        )

        application_id = os.environ["APPLICATION_ID"] 

        self.conn = psycopg2.connect(
            host=instance.read_write_dns,
            dbname=self.catalog_name,
            user=application_id,
            password=cred.token,
            sslmode="require"
        )
        
        with self.conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"[SUCCESS] Connected to: {version}")
    
    def _create_table_if_not_exists(self):
        """Create conversation_history table if it doesn't exist"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id VARCHAR(255) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            user_message TEXT,
            assistant_message TEXT,
            tool_calls JSONB,
            custom_inputs JSONB,
            custom_outputs JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.conn.cursor() as cur:
            cur.execute(create_table_sql)
            self.conn.commit()
            print("[SUCCESS] Conversation history table ready")
    
    def save_conversation(
        self, 
        session_id: str,
        user_message: Optional[str] = None,
        assistant_message: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
        custom_outputs: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a conversation exchange to the database"""
        insert_sql = """
        INSERT INTO conversation_history 
        (session_id, user_message, assistant_message, tool_calls, custom_inputs, custom_outputs)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        with self.conn.cursor() as cur:
            cur.execute(insert_sql, (
                session_id,
                user_message,
                assistant_message,
                json.dumps(tool_calls) if tool_calls else None,
                json.dumps(custom_inputs) if custom_inputs else None,
                json.dumps(custom_outputs) if custom_outputs else None
            ))
            conversation_id = cur.fetchone()[0]
            self.conn.commit()
            return str(conversation_id)
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session"""
        select_sql = """
        SELECT id, session_id, timestamp, user_message, assistant_message, 
               tool_calls, custom_inputs, custom_outputs, created_at
        FROM conversation_history 
        WHERE session_id = %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        
        with self.conn.cursor() as cur:
            cur.execute(select_sql, (session_id, limit))
            rows = cur.fetchall()
            
            conversations = []
            for row in rows:
                conversations.append({
                    'id': str(row[0]),
                    'session_id': row[1],
                    'timestamp': row[2],
                    'user_message': row[3],
                    'assistant_message': row[4],
                    'tool_calls': row[5][0] if row[5] else None,
                    'custom_inputs': json.loads(row[6]) if row[6] else None,
                    'custom_outputs': json.loads(row[7]) if row[7] else None,
                    'created_at': row[8]
                })
            
            return conversations
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    

conversation_db = ConversationDB(ONLINE_STORE, CATALOG_NAME)

#####################
## Define tools
#####################

tools = []

UC_TOOL_NAMES = ["system.ai.python_exec"]
uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
tools.extend(uc_toolkit.tools)

#####################
## Define agent logic
#####################


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    session_id: Optional[str]


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    system_prompt: Optional[str] = None,
):
    model = model.bind_tools(tools)

    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]

        # If there are function calls, continue. else, end
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        if session_id := state.get("session_id"):

            history = conversation_db.get_conversation_history(session_id, limit=10)
            for conv in reversed(history): 
                    if conv['user_message']:
                        state["messages"].insert(-1, HumanMessage(content=conv['user_message']))
                    if conv['assistant_message']:
                        state["messages"].insert(-1, AIMessage(content=conv['assistant_message']))

            response = model_runnable.invoke(state, config)

            user_message = None
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'type') and msg.type == "human":
                    user_message = msg.content
                    break
            
            assistant_message = response.content if hasattr(response, 'content') else str(response)
            
            tool_calls = None
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = [
                    {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "args": tool_call["args"]
                    }
                    for tool_call in response.tool_calls
                ]
            
            conversation_db.save_conversation(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                tool_calls=tool_calls,
                custom_inputs=state.get("custom_inputs"),
                custom_outputs=state.get("custom_outputs")
            )
        
        else: 
            response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class LangGraphResponsesAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert from a Responses API output item to ChatCompletion messages."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def _prep_msgs_for_cc_llm(self, responses_input) -> list[dict[str, Any]]:
        "Convert from Responses input items to ChatCompletion dictionaries"
        cc_msgs = []
        for msg in responses_input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

    def _langchain_to_responses(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        "Convert from ChatCompletion dict to Responses output item dictionaries"
        for message in messages:
            message = message.model_dump()
            role = message["type"]
            if role == "ai":
                if tool_calls := message.get("tool_calls"):
                    return [
                        self.create_function_call_item(
                            id=message.get("id") or str(uuid4()),
                            call_id=tool_call["id"],
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["args"]),
                        )
                        for tool_call in tool_calls
                    ]
                else:
                    return [
                        self.create_text_output_item(
                            text=message["content"],
                            id=message.get("id") or str(uuid4()),
                        )
                    ]
            elif role == "tool":
                return [
                    self.create_function_call_output_item(
                        call_id=message["tool_call_id"],
                        output=message["content"],
                    )
                ]
            elif role == "user":
                return [message]

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = []
        for msg in request.input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))

        # Extract session_id from custom_inputs if available
        session_id = request.custom_inputs.get("session_id") if request.custom_inputs else None
        
        state = {"messages": cc_msgs}
        if session_id:
            state["session_id"] = session_id
            
        for event in self.agent.stream(state, stream_mode=["updates", "messages"]):
            if event[0] == "updates":
                for node_data in event[1].values():
                    for item in self._langchain_to_responses(node_data["messages"]):
                        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception as e:
                    print(e)


# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
agent = create_tool_calling_agent(llm, tools, system_prompt)
AGENT = LangGraphResponsesAgent(agent)
mlflow.models.set_model(AGENT)
