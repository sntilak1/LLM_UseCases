import streamlit as st
import os
import openai
from langchain.pydantic_v1 import BaseModel, Field,ValidationError
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

class OrderInput(BaseModel):
    salesordernumber: Optional[str] = Field(description="sales order number for which the user wants information")
    webordernumber: Optional[str] = Field(description="Web order number for which the user wants information")

@tool(args_schema=OrderInput)
def get_order_status(**kwargs) -> dict:
    """gets the order type given either a sales order number or a web order number"""
    keys_list = list(kwargs.keys())
    values_list = list(kwargs.values())
    if len(keys_list)==0:
        {"Status":"Please provide either a web order number or a sales order number" }
    elif values_list[0]:
        if int(values_list[0]) <= 999:
            return_dict = {"Status":keys_list[0] + ' ' + values_list[0]+ " is not shipped","Plotly": fig.to_json() }
            return return_dict
        else:
            
            return_dict = {"Status":keys_list[0] + ' ' + values_list[0]+ " is shipped","Plotly": fig.to_json() }
            return return_dict
    elif values_list[1]:
        if int(values_list[1]) <= 999:
            return_dict = {"Status":keys_list[1] + ' ' + values_list[1]+ " is not shipped","Plotly": fig.to_json() }
            return return_dict
        else:
            
            return_dict = {"Status":keys_list[1] + ' ' + values_list[1]+ " is shipped","Plotly": fig.to_json() }
            return return_dict  
    else:
        return {"Status":"Please provide either a web order number or a sales order number" }
    
import requests

import datetime

# Define the input schema

# @tool(args_schema=pizza)
# def pizza_order(**kwargs) -> dict:
#     """gets deatils of the pizza being ordered"""
#     keys_list = list(kwargs.keys())
#     values_list = list(kwargs.values())
#     """Gets details of the pizza being ordered."""
#     try:
#         ordered_pizza = pizza(**kwargs)
#     except ValidationError as e:
#         return "You have choosen an size or topping  that is not avaialble.  Please try again"
#     return json.dumps(keys_list) + json.dumps(values_list) + " Order has been placed"

functions = [
    format_tool_to_openai_function(f) for f in [
        get_order_status,
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)


system_instructions = """
You are helpful but sassy assistant that needs certain information before using a specific tool. \
    1. pizza order tool sample: \
       a. size.  Needs to mbe small, medium or large \
       b. Toppings - needs to be any option of cheese, pepperoni or olives. Do not accept any other use options other than these \
    2. order status tool sample: \
       a. Either a Sales Order Number or a Web Order Number
"""
system_instructions = """
You are helpful but sassy assistant that needs certain information before using a specific tool. \

    1. order status tool sample: \
       a. Either a Sales Order Number or a Web Order Number
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


tools = [get_order_status] #,pizza_order

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True,
                               memory=memory,
                               return_intermediate_steps=True,
                               )




prompt = st.chat_input("ask me something")
if prompt:
    result = agent_executor.invoke({"input": prompt})
    for i,message in enumerate(result['chat_history']):
        if "HumanMessage" in str(type(message)):
            with st.chat_message('user', avatar='🧑‍💻'):
                st.write(message.content)
        else:
            #st.write(type(message))
            with st.chat_message('assistant', avatar='🤖'):
                st.write(message.content)
                if len(result['chat_history'])-1 == i:
                    # last message from assistant.  Check if there reponse has a figure.
                    try:
                        if 'Plotly' in result['intermediate_steps'][0][1]:
                            fig=json.loads(result['intermediate_steps'][0][1]['Plotly'])
                            st.plotly_chart(fig)
                    except:
                        pass
    st.session_state['memory']=memory