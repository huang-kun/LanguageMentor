import unittest
from unittest.mock import patch, mock_open, MagicMock
import json, os, sys
from io import StringIO

# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.agent_base import AgentBase  # 引入被测试的类


class TestAgentBase(unittest.TestCase):

    @patch("builtins.open", mock_open(read_data="test prompt"))
    def test_load_prompt(self):
        agent = AgentBase(name="TestAgent", prompt_file="prompts/conversation_prompt.txt")
        prompt = agent.load_prompt()
        self.assertEqual(prompt, "test prompt")
    
    @patch("builtins.open", mock_open(read_data='{"message": "Hello"}'))
    def test_load_intro(self):
        agent = AgentBase(name="TestAgent", prompt_file="prompts/conversation_prompt.txt", intro_file="content/intro/hotel_checkin.json")
        intro = agent.load_intro()
        self.assertEqual(intro, {"message": "Hello"})

    def test_load_intro_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            agent = AgentBase(name="TestAgent", prompt_file="prompts/fake_prompt.txt", intro_file="content/intro/fake.json")
            agent.load_intro()

    @patch("builtins.open", mock_open(read_data="invalid json"))
    def test_load_intro_invalid_json(self):
        with self.assertRaises(ValueError):
            agent = AgentBase(name="TestAgent", prompt_file="prompts/conversation_prompt.txt", intro_file="content/intro/hotel_checkin.json")
            agent.load_intro()

    # 这里的不知道哪里没有写好，总是mock不到调用
    # @patch("langchain_ollama.chat_models.ChatOllama")
    # @patch("langchain_core.prompts.ChatPromptTemplate.from_messages")
    # def test_create_chatbot(self, mock_prompt_template, mock_chat_ollama):
    #     mock_prompt_template.return_value = MagicMock()
    #     mock_chat_ollama.return_value = MagicMock()

    #     agent = AgentBase(name="Conversation", prompt_file="prompts/conversation_prompt.txt")
    #     agent.create_chatbot()

    #     with open('prompts/conversation_prompt.txt') as f:
    #         system_prompt = f.read().strip()

    #     # Check if the system prompt and chatbot were created correctly
    #     mock_prompt_template.assert_called_once_with([("system", system_prompt), ("messages", "")])
    #     mock_chat_ollama.assert_called_once_with(
    #         model="llama3.1:8b-instruct-q8_0",
    #         max_tokens=8192,
    #         temperature=0.8,
    #     )

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    @patch("utils.logger.LOG.debug")
    def test_chat_with_history(self, mock_debug, mock_invoke):
        mock_invoke.return_value = MagicMock(content="Hello, this is a response.")

        agent = AgentBase(name="Conversation", prompt_file="prompts/conversation_prompt.txt")
        user_input = "Hi"
        session_id = "test_session"
        
        response = agent.chat_with_history(user_input, session_id)

        # Check that the response is correct
        self.assertEqual(response, "Hello, this is a response.")
        mock_invoke.assert_called_once()
        mock_debug.assert_called_once_with("[ChatBot][Conversation] Hello, this is a response.")

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    def test_chat_with_history_default_session_id(self, mock_invoke):
        mock_invoke.return_value = MagicMock(content="Hello, default session response.")

        agent = AgentBase(name="Conversation", prompt_file="prompts/conversation_prompt.txt")
        user_input = "Hello"

        # No session ID provided, should use the default one
        response = agent.chat_with_history(user_input)

        self.assertEqual(response, "Hello, default session response.")
        mock_invoke.assert_called_once()

if __name__ == "__main__":
    unittest.main()
