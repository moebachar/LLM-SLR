import openpyxl
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from typing import List
from dotenv import load_dotenv


load_dotenv(override=True)


class ScreeningRetriever:
    def __init__(
        self,
        vectorstore: Chroma,
        llm_model_name: str = "gpt-4o",
        top_k: int = 3,
        num_subqueries: int = 2,
    ):
        """
        :param vectorstore: An instance of the Chroma VectorStore (already populated).
        :param llm_model_name: LLM name for sub-query generation.
        :param top_k: Number of chunks to retrieve per query.
        :param num_subqueries: Number of sub-queries to generate per user question (for multi-query retrieval).
        """
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.num_subqueries = num_subqueries
        # Using ChatOpenAI or OpenAI from langchain is up to you; here's a generic LLM usage
        self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0.0)

        # Prompt template for generating multi sub-questions
        self.sub_queries_prompt_template = PromptTemplate(
            input_variables=["question", "num_subqueries"],
            template="""
        You are an assistant that rephrases user questions. 
        Given the user question, produce different relevant ways of asking it, 
        focusing on synonyms or different angles.

        Question: {question}
        Generate {num_subqueries} sub-questions:
        """,
        )

    def read_questions_from_excel(self, excel_path: str) -> List[str]:
        """
        Reads all questions from the 'questions' sheet of the Excel file.
        Assumes no header row, just one column of questions.
        """
        wb = openpyxl.load_workbook(excel_path)
        sheet = wb["questions"]
        questions = []
        for row in sheet.iter_rows(values_only=True):
            if row and row[0]:
                questions.append(str(row[0]))
        return questions

    def generate_subqueries(self, question: str) -> list:
        """
        Generate multiple paraphrased queries (multi-query retrieval)
        using the updated chain approach.
        """
        # Chain the template with the LLM
        sequence = self.sub_queries_prompt_template | self.llm

        # Invoke the chain with the required inputs
        response = sequence.invoke(
            {"question": question, "num_subqueries": self.num_subqueries}
        )

        # response is typically an AIMessage (or ChainValues) with .content
        content = response.content.strip()

        # For simplicity, assume each line is one sub-question
        lines = [
            line.strip("- ").strip() for line in content.split("\n") if line.strip()
        ]

        # If the LLM doesn't produce enough lines, just pad with the original question
        while len(lines) < self.num_subqueries:
            lines.append(question)

        return lines

    def retrieve_relevant_chunks(
        self, question: str, article_source: str = None
    ) -> List[Document]:
        """
        Retrieves relevant chunks from Chroma for the given question.
        Optionally, filter by an article's source if you only want chunks from that article.
        """
        # 1) Generate sub-queries
        subqueries = self.generate_subqueries(question)

        # 2) For each sub-query, retrieve top_k chunks
        all_docs = []
        for q in subqueries:
            if article_source:
                # Filter to only retrieve from a specific article's chunks
                retrieved_docs = self.vectorstore.similarity_search(
                    query=q, k=self.top_k, filter={"source": article_source}
                )
            else:
                # No filter, search across entire collection
                retrieved_docs = self.vectorstore.similarity_search(q, k=self.top_k)

            all_docs.extend(retrieved_docs)

        # 3) De-duplicate by page_content (simple approach)
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        return list(unique_docs)
