from ragdocu.embedding import generate_embeddings as manual_embed
from ragdocu.vector_store import search_documents as manual_search
from ragdocu.langchain_rag.lang_vector_store import lang_search_documents as langchain_search



TEST_CASES = [
    {
        "question": "Apa contoh algoritma supervised learning?",
        "manual_relevant_chunks": [(1, 1)],
        "langchain_relevant_chunks": [(1, 1)],
    },
    {
        "question": "Sebutkan algoritma yang termasuk supervised learning.",
        "manual_relevant_chunks": [(1, 1)],
        "langchain_relevant_chunks": [(1, 1)],
    },
    {
        "question": "Apa contoh algoritma unsupervised learning?",
        "manual_relevant_chunks": [(1, 1)],
        "langchain_relevant_chunks": [(1, 1)],
    },
    {
        "question": "Jelaskan perbedaan latency dan throughput.",
        "manual_relevant_chunks": [(2, 4)],
        "langchain_relevant_chunks": [(2, 3)],
    },
    {
        "question": "Apa pengertian throughput?",
        "manual_relevant_chunks": [(2, 4)],
        "langchain_relevant_chunks": [(2, 3)],
    },
    {
        "question": "Apa pengertian latency?",
        "manual_relevant_chunks": [(2, 4)],
        "langchain_relevant_chunks": [(2, 3)],
    },
    {
        "question": "Apa yang dimaksud dengan learning rate?",
        "manual_relevant_chunks": [],
        "langchain_relevant_chunks": [],
    },
    {
        "question": "Bagaimana cara kerja convolutional neural network?",
        "manual_relevant_chunks": [],
        "langchain_relevant_chunks": [],
    },
    {
        "question": "Siapa presiden Indonesia saat ini?",
        "manual_relevant_chunks": [],
        "langchain_relevant_chunks": [],
    },
    {
        "question": "Bagaimana cara memasak nasi goreng?",
        "manual_relevant_chunks": [],
        "langchain_relevant_chunks": [],
    },
]


def retrieve_manual(question, k):
    query_embedding = manual_embed([question])[0]
    
    results = manual_search(
        query_embedding, n_results=k
    )
    
    retrieved = []
    
    for i in range (len(results["ids"][0])):
        retrieved.append(
            {
                "page": results["metadatas"][0][i]["page"],
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "distance": results["distances"][0][i]
            }
        )
        
    return retrieved

def retrieve_langchain(question, k):
    results = langchain_search(question, k=k)
    
    retrieved = []
    
    for document, distance in results:
        retrieved.append(
            {
                "page": document.metadata["page"],
                "chunk_id": document.metadata["chunk_id"],
                "distance": distance
            }
        )
        
    return retrieved


def evaluate_system(system_name, retrieve_function, k=3):
    total_hits = 0
    total_in_scope = 0
    reciprocal_ranks = []

    if system_name == "Manual RAG":
        relevance_key = "manual_relevant_chunks"
    else:
        relevance_key = "langchain_relevant_chunks"

    print(f"\n{'=' * 60}")
    print(f"{system_name} - Chunk-Level Retrieval Evaluation")
    print(f"{'=' * 60}")

    for test_case in TEST_CASES:
        question = test_case["question"]
        relevant_chunks = test_case[relevance_key]

        retrieved = retrieve_function(question, k)

        print(f"\nQuestion: {question}")

        for rank, document in enumerate(retrieved, start=1):
            print(
                f"  Rank {rank}: "
                f"Page {document['page']}, "
                f"Chunk {document['chunk_id']}, "
                f"Distance {document['distance']:.4f}"
            )

        if not relevant_chunks:
            print("  Evaluation: OUT OF SCOPE")
            continue

        total_in_scope += 1

        relevant_rank = None

        for rank, document in enumerate(retrieved, start=1):
            chunk_identity = (
                document["page"],
                document["chunk_id"]
            )

            if chunk_identity in relevant_chunks:
                relevant_rank = rank
                break

        if relevant_rank is not None:
            total_hits += 1
            reciprocal_rank = 1 / relevant_rank
            reciprocal_ranks.append(reciprocal_rank)

            print(f"  Hit: YES (Rank {relevant_rank})")
            print(f"  Reciprocal Rank: {reciprocal_rank:.4f}")
        else:
            reciprocal_ranks.append(0)
            print("  Hit: NO")
            print("  Reciprocal Rank: 0")

    hit_rate = total_hits / total_in_scope
    mrr = sum(reciprocal_ranks) / total_in_scope

    print(f"\nHit Rate@{k}: {hit_rate:.2%}")
    print(f"MRR@{k}: {mrr:.4f}")
    print(f"Hits: {total_hits}/{total_in_scope}")


def evaluate_threshold(
    system_name,
    retrieve_function,
    thresholds,
):
    if system_name == "Manual RAG":
        relevance_key = "manual_relevant_chunks"
    else:
        relevance_key = "langchain_relevant_chunks"

    print(f"\n{'=' * 60}")
    print(f"{system_name} - Threshold Evaluation")
    print(f"{'=' * 60}")

    for threshold in thresholds:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for test_case in TEST_CASES:
            question = test_case["question"]
            is_in_scope = bool(test_case[relevance_key])

            retrieved = retrieve_function(question, 1)

            if not retrieved:
                distance = float("inf")
            else:
                distance = retrieved[0]["distance"]

            predicted_accept = distance <= threshold

            if is_in_scope and predicted_accept:
                tp += 1
            elif not is_in_scope and predicted_accept:
                fp += 1
            elif not is_in_scope and not predicted_accept:
                tn += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        specificity = (
            tn / (tn + fp)
            if tn + fp
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0
        )

        balanced_accuracy = (recall + specificity) / 2
        accuracy = (tp + tn) / (tp + fp + tn + fn)

        print(f"\nThreshold: {threshold:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-score: {f1:.2%}")
        print(f"Balanced Accuracy: {balanced_accuracy:.2%}")

        
if __name__ == "__main__":
    thresholds = [0.20, 0.24, 0.25, 0.30, 0.35, 0.40]
    
    evaluate_system(
        "Manual RAG",
        retrieve_manual,
        k=3
    )
    
    evaluate_system(
        "Langchain RAG",
        retrieve_langchain,
        k=3
    )
    
    evaluate_threshold(
        "Manual RAG",
        retrieve_manual,
        thresholds,
    )

    evaluate_threshold(
        "LangChain RAG",
        retrieve_langchain,
        thresholds,
    )