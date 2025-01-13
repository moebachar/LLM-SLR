## Overview
This project automates the systematic review process by leveraging Large Language Models (LLMs) and advanced NLP techniques. It provides a modular pipeline to handle text extraction, vector embeddings, and decision-making for relevance determination. The automation significantly reduces the manual workload while maintaining scalability and accuracy.

Key features include:
- **PDF Text Extraction**: Extract structured text from articles.
- **Vector Embedding Storage**: Store embeddings for semantic similarity search.
- **Automated Screening**: Use LLMs to screen abstracts and make inclusion decisions.
- **Decision Logging**: Export results to a structured Excel file for review.

---

## Walkthrough of the Pipeline
1. **PDF Text Extraction**
   - The pipeline begins by loading and parsing PDFs into structured text using `MultiPDFExtractor`.
   - Metadata such as title, authors, and keywords are extracted via LLMs.
   - Extracted text is split into manageable chunks and embedded using OpenAI embeddings.
   - The embeddings and metadata are stored in a Chroma vector database for efficient retrieval.

2. **Question Retrieval and Answer Generation**
   - Questions are loaded from an Excel file.
   - Relevant chunks are retrieved from the vector database using multi-query retrievers.
   - Answers to the questions are generated using LLMs with context-aware prompt templates.

3. **Decision-Making**
   - Based on the generated answers, inclusion decisions are made for each article.
   - Decisions and explanations are logged in a structured Excel sheet with color-coded rows for easy interpretation.

---

## Installation Guide
### Prerequisites
- Python 3.8+
- OpenAI API Key (stored in a `.env` file)
- Required Python packages (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/automated-review.git
   cd automated-review
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: `env\\Scripts\\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK resources:
   ```bash
   python install.py
   ```

5. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY="your-openai-api-key"
   ```

6. Run the extraction pipeline:
   ```bash
   python extract.py --pdf_dir path/to/your/pdf-directory
   ```

7. Execute the main pipeline:
   ```bash
   python main.py
   ```

---

## References
1. Akinseloyin, O., Jiang, X., & Palade, V. (2024). A Novel Question-Answering Framework for Automated Abstract Screening Using Large Language Models. *medRxiv*. https://doi.org/10.1101/2023.12.17.23300102
2. Syriani, E., David, I., & Kumar, G. (2023). Assessing the Ability of ChatGPT to Screen Articles for Systematic Reviews. *arXiv*. https://doi.org/10.48550/arXiv.2307.06464
3. Khraisha, Q., et al. (2023). Can large language models replace humans in the systematic review process? *arXiv*. https://doi.org/10.48550/arXiv.2310.17526
4. Mitrov, G., et al. (2024). Combining Semantic Matching, Word Embeddings, Transformers, and LLMs for Enhanced Document Ranking: Application in Systematic Reviews. *Big Data and Cognitive Computing*. https://doi.org/10.3390/bdcc8090110
5. Landschaft, A., et al. (2024). Implementation and evaluation of an additional GPT-4-based reviewer in PRISMA-based medical systematic literature reviews. *International Journal of Medical Informatics*. https://doi.org/10.1016/j.ijmedinf.2024.105531
6. Scherbakov, D., et al. (2024). The emergence of Large Language Models (LLM) as a tool in literature reviews. *arXiv*. https://doi.org/10.48550/arXiv.2409.04600

---

Feel free to contribute to the repository or raise issues for improvements!
