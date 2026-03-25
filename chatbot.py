from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain
import json


class ChatBot:
    """
    LangChain-based chatbot using GPT-4o-mini.
    """

    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=api_key
        )

    def ask(self, user_question: str, context_logs: list[tuple]) -> str:
        """
        Ask a question with context from scene logs.
        """
        formatted_context = []
        for row in context_logs:
            _, timestamp, clip_top3, yolo_detections = row
            formatted_context.append({
                "timestamp": timestamp,
                "clip_top3": json.loads(clip_top3),
                "yolo_detections": json.loads(yolo_detections)
            })

        context_str = json.dumps(formatted_context, indent=2)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant analyzing scene logs."),
            ("user", "Context: {context}\nQuestion: {question}")
        ])

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(context=context_str, question=user_question)
