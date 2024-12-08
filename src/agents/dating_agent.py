from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage  # 导入人类消息类
from utils.logger import LOG  # 导入日志工具

from langchain_core.chat_history import (
    BaseChatMessageHistory,  # 基础聊天消息历史类
    InMemoryChatMessageHistory,  # 内存中的聊天消息历史类
)
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类

from .base_scenario_agent import ScenarioAgent

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

class DatingAgent(ScenarioAgent):
    def __init__(self):
        super().__init__()
        self.name = "Dating Agent"

        prompt_path = "prompts/dating_prompt.txt"
        with open(prompt_path) as f:
            self.system_prompt = f.read().strip()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        self.chatbot = self.prompt | ChatOllama(
            model="llama3.1:8b-instruct-q8_0",
            max_token=8192,
            temperature=0.8,
        )

        self.chatbot_with_history = RunnableWithMessageHistory(
            self.chatbot,
            get_session_history,
        )

        self.config = {
            "configurable": {
                "session_id": "dating-chat"
            }
        }

    def chat(self, user_input):
        response = self.chatbot.invoke(
            [HumanMessage(content=user_input)],
        )
        return response.content
    
    def chat_with_history(self, user_input):
        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],
            self.config,
        )
        return response.content

    def respond(self, user_input):
        return f"Dating echo: {user_input}"
