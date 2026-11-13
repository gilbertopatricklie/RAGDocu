from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from ragdocu.langchain_rag.lang_vector_store import get_retriever
from ragdocu.langchain_rag.lang_llm import llm, rag_prompt


def format_documents(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )
    
def create_rag_chain(top_k=3):
    retriever = get_retriever(k=top_k)
    
    retriever_chain = (
        RunnableParallel(
            {
                "question": RunnablePassthrough(),
                "documents": retriever
            }
        )
        | RunnablePassthrough.assign(context = lambda x: format_documents(x["documents"]))
    )
    
    
    rag_chain = (
        retriever_chain
        | RunnableParallel(
            {
                "answer": (
                    {
                        "context": lambda x: x["context"],
                        "question": lambda x: x["question"]
                    }
                    | rag_prompt
                    | llm
                    | StrOutputParser()
                ),
                "documents": lambda x: x["documents"]
            }
        )
    )
    
    return rag_chain

def ask_question(question, top_k=3):
    rag_chain = create_rag_chain(top_k=top_k)
    
    result = rag_chain.invoke(question)
    
    sources = []
    
    for document in result["documents"]:
        sources.append(
            {
                "text": document.page_content,
                "metadata": document.metadata
            }
        )
        
    return {
        "question": question,
        "answer": result["answer"],
        "sources": sources
    }