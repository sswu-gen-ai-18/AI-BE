from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from agents.policy_rag import POLICY_RETRIEVER
import os

class GuideAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        # LangChain의 ChatOpenAI 모델
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            openai_api_key=api_key
        )

        # PromptTemplate (system + user 합쳐서 관리)
        self.template = PromptTemplate(
            input_variables=["system_prompt", "user_prompt", "policy_context"],
            template="""
{system_prompt}

[정책 정보 참고]
{policy_context}

[고객 입력]
{user_prompt}

위 모든 정보를 참고하여 고객에게 제공할 대응 문장을 2~3문장으로 작성하세요.
너무 딱딱하지 않고 친절하게 설명하세요.
"""
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.template
        )

    def generate(self, system_prompt: str, user_prompt: str):
        # RAG: 정책 문서에서 고객 발화 기반 검색 수행
        related_docs = POLICY_RETRIEVER.get_relevant_documents(user_prompt)
        policy_context = "\n\n".join(doc.page_content for doc in related_docs)

        # LangChain LLMChain 실행
        response = self.chain.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            policy_context=policy_context
        )

        return response.strip()
