import os
import shutil
import json

from dotenv import load_dotenv

import nltk
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT

from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate

load_dotenv(override=True)


class MultiPDFExtractor:
    def __init__(
        self,
        pdf_directory: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-ada-002",
        persist_directory: str = "chroma_db",
    ):
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

        # Initialize embeddings and vectorstore
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore = Chroma(
            collection_name="pdf_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        self.keybert = KeyBERT()

        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

        # Prompt for extracting metadata as JSON from the structured text
        self.metadata_prompt_template = PromptTemplate(
            input_variables=["structured_text"],
            template="""    
            You are a helpful assistant. I have a structured article as follows:

            {structured_text}

            From this structured text, extract the following metadata and return it as a strict JSON object with keys:
            - title (string)
            - authors (list of strings)
            - keywords (list of strings)

            Only return the JSON object and nothing else.


            the first character of your output should be an opening bracket and the last character should be a closing bracket.

            the output is bad if it is not clean, meaning that it contains anything else than the JSON object.
            """,
        )

    def load_pdfs(self):
        """
        Load all PDFs from the directory as a list of LangChain Documents.
        Each PDF will return a list of Documents (usually one Document per PDF).
        """
        loader = DirectoryLoader(
            self.pdf_directory, glob="*.pdf", loader_cls=PyPDFLoader
        )
        docs = loader.load()

        all_files = os.listdir(self.pdf_directory)

        # Filter for .pdf files (case-insensitive)
        pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

        article_names = [os.path.splitext(pdf_file)[0] for pdf_file in pdf_files]

        # Combine pages of the same PDF into one Document
        documents_by_source = {}
        for d in docs:
            src = d.metadata.get("source", "")
            if src not in documents_by_source:
                documents_by_source[src] = []
            documents_by_source[src].append(d)

        combined_docs = []
        for src, doc_list in documents_by_source.items():
            # Combine text from all pages into one Document
            full_text = "\n".join([doc.page_content for doc in doc_list])
            print(os.path.splitext(src.split("\\")[-1])[0])
            combined_docs.append(
                Document(
                    page_content=full_text,
                    metadata={"source": os.path.splitext(src.split("\\")[-1])[0]},
                )
            )

        return combined_docs

    def extract_metadata_with_llm(self, structured_text: str) -> dict:
        """
        Use a second LLM call to extract metadata as JSON from the structured text.
        """
        sequence = self.metadata_prompt_template | self.llm

        response = sequence.invoke({"structured_text": structured_text[:5000]})

        response = response.content

        # remove the first line of the response
        response = response[response.find("\n") + 1 :]

        # remove the last line of the response
        response = response[: response.rfind("\n")]

        # Parse the JSON from the response
        metadata = {"title": "", "authors": [], "keywords": []}
        try:
            metadata = json.loads(response)
        except json.JSONDecodeError:
            pass
        return metadata

    def chunk_text_RecursiveCharacterTextSplitter(self, text: str) -> list:
        """
        Use a RecursiveCharacterTextSplitter to chunk the structured text.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def chunk_text_NLTK(self, text: str) -> list:
        """
        Use NLTK sentence tokenization to split the text into sentences,
        then group sentences into chunks up to the specified chunk_size.
        """
        # Split the text into sentences
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence stays below chunk_size, keep appending
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Otherwise, push the current chunk to the list, and reset
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def embed_chunks(self, chunks: list) -> list:
        """
        Embed the chunks of text using the OpenAI text-embedding-ada-002 model.
        """
        embeddings = []
        for chunk in chunks:
            embeddings.append(self.embeddings.embed_query(chunk))
        return embeddings

    def extract_keyphrases(self, text: str, top_n: int = 5) -> list:
        """
        Use KeyBERT to extract top_n keyphrases from the text.
        Returns a list of keyphrase strings.
        """
        # keyBERT returns a list of (keyphrase, score) tuples
        keywords_with_scores = self.keybert.extract_keywords(text, top_n=top_n)
        keyphrases = [kw[0] for kw in keywords_with_scores]
        return keyphrases

    def save_to_chroma(self, chunks: list, embeddings: list, metadata: dict):
        """
        Save the chunks and metadata to the Chroma vectorstore.
        """

        metadatas = [metadata] * len(chunks)
        documents = []
        for i, chunk in enumerate(chunks):
            # Optionally customize metadata per chunk if needed
            chunk_metadata = metadatas[i]
            # You may include additional per-chunk metadata here
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        self.vectorstore.add_documents(documents)

    def process_pdf(self, doc: Document):
        """
        Process a single PDF document:
        - Parse structure with LLM into structured text
        - Extract metadata with another LLM call
        - Chunk text
        - Embed chunks
        - Store in Chroma
        """

        print("getting metadata.........", end="")
        metadata = self.extract_metadata_with_llm(doc.page_content)
        print("✅")
        # Convert list fields to strings
        if isinstance(metadata.get("authors"), list):
            metadata["authors"] = ", ".join(metadata["authors"])
        if isinstance(metadata.get("keywords"), list):
            metadata["keywords"] = ", ".join(metadata["keywords"])

        # (Optional) Also add KeyBERT keyphrases to metadata
        print("extracting keyphrases....", end="")
        keyphrases = self.extract_keyphrases(doc.page_content, top_n=5)
        metadata["keybert_keyphrases"] = ", ".join(keyphrases)

        print("✅")

        # Add extra metadata: source
        metadata["source"] = doc.metadata.get("source", "")

        print("chunking text............", end="")
        # Chunk the text
        chunks = self.chunk_text_RecursiveCharacterTextSplitter(doc.page_content)
        print("✅")

        print("embedding chunks.........", end="")
        # Embed the chunks
        embeddings = self.embed_chunks(chunks)
        print("✅")

        print("saving to Chroma.........", end="")
        # Save to Chroma
        self.save_to_chroma(chunks=chunks, embeddings=embeddings, metadata=metadata)
        print("✅")

    def set_vectorstore(self):
        """Remove the Chroma DB folder so that a new DB is created from scratch."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        # Re-instantiate Chroma to ensure it's a fresh instance
        self.vectorstore = Chroma(
            collection_name="pdf_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def run_pipeline(self):

        print(
            f"-----------------------------  Processing PDFs...  -----------------------------"
        )
        docs = self.load_pdfs()

        print(f"loaded {len(docs)} PDFs ✅")

        for d in docs:
            print(
                f"-----------------------------  Processing PDF: {d.metadata.get('source', '')}  -----------------------------"
            )
            self.process_pdf(d)


if __name__ == "__main__":
    pdf_dir = "articles-lite"  # Directory containing PDF files
    extractor = MultiPDFExtractor(pdf_directory=pdf_dir)
    extractor.run_pipeline()
    extractor.vectorstore._client.persist()
    print("Processing complete. Vector database is constructed and persisted.")
