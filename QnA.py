from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)


class QnAWithLLM:
    def __init__(self, model_name: str = "o1-mini", temperature: float = 0.0):
        """
        :param model_name: The LLM model name for Q/A.
        :param temperature: The temperature for answer generation.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        # Define a prompt template that references two variables: context_text, question
        self.answer_prompt_template = PromptTemplate(
            input_variables=["context_text", "question"],
            template="""
            You are a helpful assistant.
            Use the following context to answer the user's question.

            CONTEXT:
            {context_text}

            QUESTION: {question}

            Please provide a concise answer:
            """,
        )

    def generate_answer(self, question: str, relevant_chunks: List[Document]) -> str:
        """
        Combine the question + relevant chunks into a single prompt,
        and generate an answer from the LLM using the chain approach.
        """
        # 1) Prepare the context from chunks
        context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])

        # 2) Build a mini-chain by piping the template into the LLM
        sequence = self.answer_prompt_template | self.llm

        # 3) Invoke the chain with the required inputs
        response = sequence.invoke({"context_text": context_text, "question": question})

        # 4) response is typically an AIMessage with .content
        answer = response.content.strip()
        return answer
