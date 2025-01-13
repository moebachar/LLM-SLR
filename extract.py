from Extractor import MultiPDFExtractor


import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Extract text from PDF files and construct a vector database."
)
parser.add_argument(
    "--pdf_dir",
    type=str,
    default="articles-lite",
    help="Directory containing PDF files",
)

pdf_dir = parser.parse_args().pdf_dir
extractor = MultiPDFExtractor(pdf_directory=pdf_dir)
extractor.run_pipeline()
print("Processing complete. Vector database is constructed and persisted.")
