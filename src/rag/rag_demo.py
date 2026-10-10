from src.rag.rag_engine import retrieve_similar_tickets


def main():
    ticket = """
    The data analytics platform crashed because of insufficient RAM.
    We restarted the server but the problem persists.
    The platform is currently unavailable and we need it operational
    as soon as possible.
    """

    results = retrieve_similar_tickets(ticket, top_k=5)

    print("\n" + "=" * 80)
    print("RAG TEST")
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        print(f"\nRESULT #{i}")
        print("-" * 80)
        print(f"Similarity: {result['score']:.4f}")
        print(result["document"][:1500])

if __name__ == "__main__":
    main()