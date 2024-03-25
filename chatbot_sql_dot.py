import streamlit as st
import os
import openai
from langchain.pydantic_v1 import  Field, ValidationError, BaseModel, validator
#from pydantic import validator,BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Optional
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain.utilities.openapi import OpenAPISpec

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
import json
import requests
import base64


###  New Classes to support intermediate steps
from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.pydantic_v1 import Field

from langchain.memory.utils import get_prompt_input_key


class SandeepBaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            #if len(outputs) != 1:
            #    raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()



from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.pydantic_v1 import root_validator

from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key


class SandeepConversationBufferMemory(SandeepBaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:

    @property
    def buffer(self) -> Any:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        return get_buffer_string(
            self.chat_memory.messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}

###

openai.api_key = os.environ['OPENAI_API_KEY']


### Define a figure just as a placeholder.

import plotly.express as px
data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data_canada, x='year', y='pop')


### end of figure

#from pydantic import BaseModel, Field
class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")

@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"

class OrderInput(BaseModel):
    salesordernumber: Optional[str] = Field(description="sales order number for which the user wants information")
    webordernumber: Optional[str] = Field(description="Web order number for which the user wants information")

@tool(args_schema=OrderInput)
def get_order_type(**kwargs) -> str:
    """gets the order type given either an order number or a web order number"""
    keys_list = list(kwargs.keys())
    values_list = list(kwargs.values())
    if values_list[0]:
        if int(values_list[0]) <= 999:
            return f'{keys_list[0]} {values_list[0]} is not shipped'
        else:
            return f'{keys_list[0]} {values_list[0]} is shipped'
    elif values_list[1]:
        if int(values_list[1]) <= 999:
            return f'{keys_list[1]} {values_list[1]} is not shipped'
        else:
            return f'{keys_list[1]} {values_list[1]} is shipped'  
    else:
        return "Please provide either a web order number or a sales order number"  
    
import requests
#from pydantic import BaseModel, Field
import datetime

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'

import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

from typing import Literal

# class pizza(BaseModel):
#     size: Literal['small', 'medium', 'large'] = Field(description="size of the pizza that is being ordered")
#     toppings: Literal['cheese', 'pepperoni', 'olives'] = Field(description="toppings that are requested")

class pizza(BaseModel):
    size: str = Field(description="size of the pizza that is being ordered")
    toppings: List[str] = Field(description="toppings that are requested")

    @validator("size", allow_reuse=True)
    def check_size(cls, v):
        acceptable_items = {"small", "medium", "large"}
        if v not in acceptable_items:
            return (f"{v} is not an acceptable item for 'size'")
        return v

    @validator("toppings", each_item=True, allow_reuse=True)
    def check_toppings(cls, v):
        acceptable_items = {"cheese", "olives", "pepperoni"}
        if v not in acceptable_items:
            return (f"{v} is not an acceptable item for 'toppings'")
        return v

@tool(args_schema=pizza)
def pizza_order(**kwargs) -> dict:
    """gets deatils of the pizza being ordered"""
    keys_list = list(kwargs.keys())
    values_list = list(kwargs.values())
    """Gets details of the pizza being ordered."""
    try:
        ordered_pizza = pizza(**kwargs)
    except ValidationError as e:
        return "we only have olives, cheese and pepperoni.  Please try again"
    return json.dumps(keys_list) + json.dumps(values_list) + " Order has been placed"

class source_destination(BaseModel):
    source: str = Field(description="name of the source table we wish to join")
    destination: str = Field(description="name of the destination table we wish to join")
    
@tool(args_schema=source_destination)
def salesorder_to_sales_order_line(source: str, destination: str) -> dict:
    """connects the sales order table to the sales order line table"""
    return {"SQL":"select * from bv_sales_order left join bv_sales_order_line on bv_sales_order.salesorderkey = bv.sales_order_line.salesorderkey", "Biz Rules":"Key other conditions include Corp flag = Y"}

@tool(args_schema=source_destination)
def sales_order_line_to_end_customer(source: str, destination: str) -> str:
    """connects the sales order line table to the customer table"""
    return "select * from bv_sales_order_line left join bv_end_customer on bv_sales_order_line.end_customer_key = bv_end_customer.end_customer_key"

@tool(args_schema=source_destination)
def end_customer_to_market_segment(source: str, destination: str) -> str:
    """connects the end customer table to the market segment"""
    return "select * from bv_end_customer left join bv_customer_registry on bv_end_customer.end_customer_key = bv_customer_registry.end_customer_key"

functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia,salesorder_to_sales_order_line,sales_order_line_to_end_customer,end_customer_to_market_segment,pizza_order
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)



system_instructions = """
You are a assistant that has the following tools.  Please read the rules of each tool and proceed

1. Piza orderinng tool.
       a. size.  Needs to mbe small, medium or large \
       b. Toppings - needs to be any option of cheese, pepperoni or olives.  
       c. share the list of valid toppings if they choose a value which is not in the list \
2. SQL generation tool.  follow the graphviz dot instructions  enclosed below in ``` and uses the tools provided to construct SQL queries that adhere to the graphviz relationships and give the user the  query they need.  
    Please also adhere to the following rules:
    a. Share biz rules returned by the tools as notes to the user
    b. Do not make up any  additional commentary other than SQL and the biz rules returned by the tools.
```
    // Nodes representing tables
    tableA [label="Sales Order" shape=plaintext width=.15 height=.15];
    tableB [label="Sales Order Line" shape=plaintext width=.15 height=.15];
    tableC [label="end customer" shape=plaintext width=.15 height=.15];
    tableD [label="market segment" shape=plaintext width=.15 height=.15];
    // Nodes representing columns
    A_sales_order_id [label="sales order id"];
    A_sales_order_currency [label="sales order currency"];
    B_sales_order_id [label="sales order id"];
    B_sales_order_line_id [label="sales order line id"];
    B_sales_order_amt_usd [label="sales order amt USD"];
    B_end_customer_key [label="end customer key"];
    C_end_customer_key [label="end customer key"];
    C_end_customer_name [label="end customer name"];
    D_end_customer_key [label="end customer key"];
    D_market_segment_name[label="market segment name"];


    // Edges connecting common key
    A_sales_order_id -> B_sales_order_id [label="sales order id"];
    B_end_customer_key -> C_end_customer_key;
    C_end_customer_key -> D_end_customer_key

    // Subgraphs representing tables
    subgraph cluster_tableA ()
        label = "Sales Order";
        style = filled;
        color = lightgrey;
        A_sales_order_id;
        A_sales_order_currency;
    )

    subgraph cluster_tableB (
        label = "Sales Order Line";
        style = filled;
        color = lightgrey;
        B_sales_order_id;
        B_sales_order_line_id;
        B_sales_order_amt_usd;
        B_end_customer_key;
    )
    subgraph cluster_tableC (
        label = "End Customer";
        style = filled;
        color = lightgrey;
        C_end_customer_key
        C_end_customer_name;
    )
    subgraph cluster_tableD (
        label = "Market Segment";
        style = filled;
        color = lightgrey;
        D_end_customer_key
        D_market_segment_name;
    )   

```

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instructions),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

if "memory" not in st.session_state:
    memory = SandeepConversationBufferMemory(return_messages=True,memory_key="chat_history")
else:
    memory=st.session_state['memory']


tools = [search_wikipedia,salesorder_to_sales_order_line,sales_order_line_to_end_customer,end_customer_to_market_segment,pizza_order] 

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True,
                               memory=memory,
                               return_intermediate_steps=True,
                               )



with st.sidebar:
    new_con = st.button("New Conversation")

if new_con:
    memory.chat_memory.messages=[]

prompt = st.chat_input("ask me something")
if prompt:
    result = agent_executor.invoke({"input": prompt})
    for message in result['chat_history']:
        if "HumanMessage" in str(type(message)):
            with st.chat_message('user', avatar='🧑‍💻'):
                st.write(message.content)
        else:
            #st.write(type(message))
            with st.chat_message('assistant', avatar='🤖'):
                st.write(message.content)
    st.session_state['memory']=memory