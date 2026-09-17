from langchain_pinecone import PineconeVectorStore


class Query_Agent:
    def __init__(self, pinecone_index, embeddings, namespace: str):
        self.namespace = namespace
        self.vectorstore = PineconeVectorStore(
            index=pinecone_index,
            embedding=embeddings,
        )

    def query_vector_store(self, query: str, k: int = 6):
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            namespace=self.namespace,
        )
