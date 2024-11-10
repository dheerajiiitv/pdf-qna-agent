import json
from typing import Sequence, List

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from openai.types.chat import ChatCompletionMessageToolCall

from app.utils.logger import get_logger
from app.config import settings
logger = get_logger(__name__)



class PDFQAAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]
        
        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)
        logger.info(f"Agent response: {ai_message}")
        tool_calls = additional_kwargs.get("tool_calls", None)
        logger.info(f"Tool calls: {tool_calls}")
   

        while tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history, tools=tools).message
                chat_history.append(ai_message)
                tool_calls = ai_message.additional_kwargs.get("tool_calls", None)
                logger.info(f"Agent response: {ai_message}")
        


                

        return ai_message.content

    def _call_function(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> ChatMessage:
        id_ = tool_call.id # type: ignore
        function_call = tool_call.function # function_call contains the name and arguments
        tool = self._tools[function_call.name]
        schema = tool.metadata.fn_schema

        output = tool(**json.loads(function_call.arguments))
        return ChatMessage(
            name=function_call.name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_call.name,
            },
        )