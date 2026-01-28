from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
from dllmforge.rag_search_and_response import AzureOpenAIEmbeddingModel, IndexManager, Retriever


def set_up_RAG_AZURE(
    data_dir,
    index_name,
    index_exists=False,
    chunk_size=1000,
    overlap_size=200,
    embedding_model="text-embedding-3-large",
):
    # initialize the embedding model
    model = AzureOpenAIEmbeddingModel(model=embedding_model)  # here there is a default model set, you can customize it if needed
    if not (index_exists):
        pdfs = list(data_dir.glob("*.pdf"))  # find all PDF files in the directory
        loader = PDFLoader()  # Load the PDF document
        chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)  # Create chunks with custom settings
        # embed the chunks
        global_embeddings = []
        for pdf_path in pdfs:
            pages_with_text, file_name, metadata = loader.load(pdf_path)
            # Create chunks with custom settings
            chunks = chunker.chunk_text(pages_with_text, file_name)
            # Embed the document chunks
            chunk_embeddings = model.embed(chunks)
            global_embeddings.extend(chunk_embeddings)
            print(f"Embedded {len(chunk_embeddings)} chunks from {file_name}.")
        print(f"Total embeddings generated: {len(global_embeddings)}")
        ## Index and upload phase
        embedding_dim = 3072  # Adjust if your embedding model uses a different dimension
        index_manager = IndexManager(index_name=index_name, embedding_dim=embedding_dim)
        index_manager.create_index()
        index_manager.upload_documents(global_embeddings)
    # Retrieval phase
    retriever = Retriever(embedding_model=model, index_name=index_name)
    return retriever
