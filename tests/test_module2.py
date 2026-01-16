from modules.module2_semantic_indexing.semantic_indexing import SemanticIndexer

def test_semantic_search():
    # Load existing index
    indexer = SemanticIndexer()
    indexer.load_index()
    
    # Test queries
    test_queries = [
        "TCP protocol connection establishment",
        "OSI model layers",
        "IP addressing and subnetting",
        "Ethernet frame format",
        "routing algorithms"
    ]
    
    print("=== SEMANTIC SEARCH TEST ===\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        results = indexer.search(query, k=3)
        
        for result in results:
            print(f"  {result['rank']}. Score: {result['similarity']}")
            print(f"     Source: {result['source']}")
            print(f"     Text: {result['text']}")
            print()
        print("-" * 80)

if __name__ == "__main__":
    test_semantic_search()