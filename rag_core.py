import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

class RAGSystem:
    def __init__(self, db_path="./chroma_db"):
        self.db_path = db_path
        self._setup_environment()
        
        # Initialize chroma db
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.chroma_collection = self.chroma_client.get_or_create_collection("rag_experiment")
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
        )

    def _setup_environment(self):
        gemini_api_key = os.getenv("GOOGLE_API_KEY", "")
        if gemini_api_key:
            from llama_index.llms.gemini import Gemini
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            llm = Gemini(model="models/gemini-2.5-flash", api_key=gemini_api_key)
            Settings.llm = llm
        
        embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    def ingest_documents(self, file_paths):
        docs = SimpleDirectoryReader(input_files=file_paths).load_data()
        for doc in docs:
            doc.metadata["file_name"] = os.path.basename(doc.metadata.get("file_name", "unknown"))
            doc.metadata["doc_id"] = doc.metadata["file_name"]
            
        self.index.insert_nodes(Settings.node_parser.get_nodes_from_documents(docs))
        return [doc.id_ for doc in docs]

    def retrieve(self, query, k=5):
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(query)
        return nodes

    def generate_response(self, query, retrieved_nodes):
        context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
        prompt = f"Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer: "
        
        response = Settings.llm.complete(prompt)
        return str(response)
        
    def adaptive_retrieve(self, query):
        nodes = self.retrieve(query, k=5)
        if not nodes:
            return nodes
        top_score = getattr(nodes[0], 'score', 0)
        if top_score > 0.85:
            return nodes[:1]
        elif top_score > 0.70:
            return nodes[:3]
        else:
            return nodes[:5]
