from Retriever import ScreeningRetriever
from QnA import QnAWithLLM
from Decision import ScreeningDecision
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import os
from typing import List
from dotenv import load_dotenv


import argparse


load_dotenv(override=True)


def run_screening_pipeline(
    excel_questions_path: str,
    articles: List[str],  # e.g. from your PDF metadata or filenames
    vectorstore,  # your Chroma vector store
    output_excel_path: str,
):
    """
    Orchestrates the pipeline:
      - For each of n articles,
      - For each of m questions,
      - retrieve -> Q/A -> decide -> record in new article's sheet
    """
    # Instantiate the needed classes
    screening_retriever = ScreeningRetriever(vectorstore=vectorstore)
    qna_model = QnAWithLLM(model_name="o1-mini", temperature=1)
    decision_model = ScreeningDecision(model_name="o1-mini", temperature=1)

    print(f"Running screening pipeline for {len(articles)} articles...")

    print(f"Reading questions from {excel_questions_path}...")
    # Read all questions from the Excel (m questions)
    questions = screening_retriever.read_questions_from_excel(excel_questions_path)

    print(f"Processing {len(questions)} questions for each article...")
    # For each article => new sheet
    for article_name in articles:
        print(f"Processing article: {article_name}...")
        decisions_for_this_article = []

        for question in questions:
            # 1) Retrieve
            print(f"Retrieving relevant chunks for question: {question}...", end="")
            relevant_chunks = screening_retriever.retrieve_relevant_chunks(
                question=question, article_source=article_name
            )
            print(f"✅ Retrieved {len(relevant_chunks)} chunks.")
            # 2) Q/A
            print(f"Generating answer for question: {question}...", end="")
            answer = qna_model.generate_answer(question, relevant_chunks)
            print(f"✅ Generated answer: {answer}")
            # 3) Decide => (YES/NO, explanation=answer)
            print(f"Deciding inclusion for question: {question}...", end="")
            decision, explanation = decision_model.decide_inclusion(question, answer)
            print(f"✅ Decision: {decision}")
            decisions_for_this_article.append((decision, explanation))

        print(
            f"Finished processing {len(questions)} questions for article: {article_name}."
        )
        # 4) Write all answers to new sheet for article
        decision_model.write_decisions_for_article(
            excel_path=output_excel_path,
            article_name=article_name,
            decisions=decisions_for_this_article,
        )

    print(
        f"Done! Created {len(articles)} sheets in {output_excel_path}, each with {len(questions)} rows."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract text from PDF files and construct a vector database."
    )

    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="articles-lite",
        help="Directory containing PDF files",
    )

    parser.add_argument(
        "--output_excel",
        type=str,
        default="output.xlsx",
        help="Output Excel file path",
    )

    parser.add_argument(
        "--questions_excel",
        type=str,
        default="input.xlsx",
        help="Excel file path containing questions",
    )

    args = parser.parse_args()


    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    directory_path = args.pdf_dir

    all_files = os.listdir(directory_path)

    # Filter for .pdf files (case-insensitive)
    pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

    # Strip off the .pdf extension for each file
    article_names = [os.path.splitext(pdf_file)[0] for pdf_file in pdf_files]

    # Example usage
    run_screening_pipeline(
        excel_questions_path=args.questions_excel,
        articles=article_names,
        vectorstore=vectorstore,
        output_excel_path=args.output_excel,
    )
