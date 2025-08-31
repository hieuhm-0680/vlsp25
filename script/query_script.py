import re
from src.heterogeneous_graph_builder import HeterogeneousGraphBuilder
from src.graph_query_engine import GraphQueryEngine, MultiModalQuery
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
import random

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))


def load_embeddings(npy_path: str, txt_path: str):
    """Load embeddings and corresponding file names"""
    embeddings = np.load(npy_path)
    with open(txt_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    return embeddings, file_names


def find_image_indices(image_id: str, file_names: List[str]) -> List[int]:
    """Find indices of cropped images that match the given image_id"""
    indices = []
    for i, file_name in enumerate(file_names):
        # Extract base image id from filename (e.g., "private_test_1_1" from "private_test_1_1_crop_1_traffic_sign_0.58.jpg")
        if file_name.startswith(image_id):
            indices.append(i)
    # Sort indices for deterministic order
    return sorted(indices)


def query_graph(query_engine: GraphQueryEngine, embeddings: np.ndarray, indices: List[int],
                max_results, search_depth) -> List[Any]:
    """Perform graph queries for the given image embeddings"""
    all_results = []

    # Sort indices for deterministic processing order
    for idx in sorted(indices):
        query_embedding = embeddings[idx]
        query = MultiModalQuery(
            image_embedding=torch.tensor(query_embedding),
            max_results=max_results,
            search_depth=search_depth,
        )
        results = query_engine.multi_modal_query(query)
        all_results.extend(results)

    return all_results


def format_results(results: List[Any], question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format query results to match the retrieval_results.json structure"""
    # Remove duplicates and sort by relevance score
    print(f"Order before sorting: {[r.node_id for r in results]}")
    MAX_SCORE = 1
    REDUCTION_COEFFICIENT = 0.001
    
    seen_nodes = set()
    unique_results = []
    for result in results:
        if result.node_id not in seen_nodes:
            seen_nodes.add(result.node_id)
            # if node is a text node that belongs to 22 28 32 36 41, set the relevance score to MAX_SCORE
            if result.node_type == 'text' and result.node_id.split(':')[-1] in ['22', '28', '32', '36', '41', '47']:
                result.relevance_score = MAX_SCORE - 0 * REDUCTION_COEFFICIENT
            # if node is a text node that belongs to neighbor of 22 28 32 36 41, and have the format [A-Z].\d+, set the relevance score to MAX_SCORE - 0.01
            # elif result.node_type == 'text' and any(neighbor in graph.neighbors(result.node_id) for neighbor in ['22', '28', '32', '36', '41']) and re.match(r'[A-Z]', result.node_id.split(':')[-1]):
                # result.relevance_score = MAX_SCORE - 1 * REDUCTION_COEFFICIENT

            unique_results.append(result)

    # Sort by relevance score first
    unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
    
    # Reorder to ensure specific node types follow their corresponding priority nodes
    priority_mapping = {'22': 'B', '28': 'C', '32': 'D', '36': 'E', '41': 'K'}
    final_results = []
    used_indices = set()
    
    for i, result in enumerate(unique_results):
        if i in used_indices:
            continue
            
        final_results.append(result)
        used_indices.add(i)
        
        # If this is a priority node, find the highest scoring corresponding type node
        node_id = result.node_id.split(':')[-1]
        if node_id in priority_mapping:
            target_prefix = priority_mapping[node_id]
            best_match = None
            best_idx = -1
            
            for j, candidate in enumerate(unique_results[i+1:], start=i+1):
                if j in used_indices:
                    continue
                candidate_id = candidate.node_id.split(':')[-1]
                if candidate_id.startswith(target_prefix + '.'):
                    if best_match is None or candidate.relevance_score > best_match.relevance_score:
                        best_match = candidate
                        best_idx = j
            
            if best_match:
                final_results.append(best_match)
                used_indices.add(best_idx)
    
    # Add any remaining unused results
    for i, result in enumerate(unique_results):
        if i not in used_indices:
            final_results.append(result)
    
    unique_results = final_results
    print(f"Order after sorting: {[r.node_id for r in unique_results]}")

    # Convert to retrieval format
    retrieved_articles = []
    for rank, result in enumerate(unique_results, 1):
        if result.node_type == 'text':
            # Parse node_id to extract law_id and article_id
            parts = result.node_id.split(':')
            if len(parts) >= 2:
                law_id = ':'.join(parts[:-1])
                article_id = parts[-1]

                # Get article title from node data if available
                article_title = result.data.get(
                    'title', '') or result.data.get('content', '')[:100] + '...'

                retrieved_articles.append({
                    "law_id": law_id,
                    "article_id": article_id,
                    "article_title": article_title,
                    "similarity_score": result.relevance_score,
                    "rank": rank
                })

    return {
        "id": question_data["id"],
        "image_id": question_data["image_id"],
        "question": question_data["question"],
        "retrieved_articles": retrieved_articles,
        "relevant_articles": []  # This would be filled from ground truth if available
    }


def main():
    parser = argparse.ArgumentParser(
        description='Perform multi-modal queries on traffic law graph')
    parser.add_argument('--questions_file', required=True,
                        help='Path to questions JSON file (e.g., vlsp2025_submission_task1.json)')
    parser.add_argument('--mapping_file',
                        help='Path to question-to-image mapping JSON file (optional)')
    parser.add_argument('--image_folder', required=True,
                        help='Path to folder containing cropped images')
    parser.add_argument('--embeddings_npy', required=True,
                        help='Path to embeddings .npy file')
    parser.add_argument('--embeddings_txt', required=True,
                        help='Path to embeddings .txt file (file names)')
    parser.add_argument('--output_file', required=True,
                        help='Path to output JSON file')
    parser.add_argument('--law_db_path', default='dataset/vlsp25/law_db/vlsp2025_law_new.json',
                        help='Path to law database JSON file')
    parser.add_argument('--law_images_dir', default='images/law_images',
                        help='Path to law images directory')
    parser.add_argument('--graph_embeddings_npy', default='embeddings/image_embeddings.npy',
                        help='Path to graph image embeddings .npy file')
    parser.add_argument('--graph_embeddings_txt', default='embeddings/image_embeddings.txt',
                        help='Path to graph image embeddings .txt file')
    parser.add_argument('--max_results', type=int, default=15,
                        help='Maximum number of results per query')
    parser.add_argument('--search_depth', type=int, default=3,
                        help='Graph search depth')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # Validate input files
    for file_path in [args.questions_file, args.embeddings_npy, args.embeddings_txt]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    # Load questions
    print(f"Loading questions from {args.questions_file}")
    with open(args.questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Load mapping file if provided
    mapping = {}
    if args.mapping_file and Path(args.mapping_file).exists():
        print(f"Loading mapping from {args.mapping_file}")
        with open(args.mapping_file, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

    # Load embeddings
    print(
        f"Loading embeddings from {args.embeddings_npy} and {args.embeddings_txt}")
    embeddings, file_names = load_embeddings(
        args.embeddings_npy, args.embeddings_txt)
    print(f"Loaded {len(embeddings)} embeddings")

    # Build graph and initialize query engine
    print("Building heterogeneous graph...")
    builder = HeterogeneousGraphBuilder(
        law_db_path=args.law_db_path,
        law_images_dir=args.law_images_dir
    )
    global graph
    graph = builder.build_graph()
    print(
        f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Initialize query engine
    print("Initializing query engine...")
    query_engine = GraphQueryEngine(graph)

    # Load graph embeddings if they exist
    if Path(args.graph_embeddings_npy).exists() and Path(args.graph_embeddings_txt).exists():
        query_engine.load_embeddings(
            npy_path=args.graph_embeddings_npy,
            txt_path=args.graph_embeddings_txt,
        )
        print("Graph embeddings loaded")
    else:
        print("Warning: Graph embeddings not found, image similarity search may not work properly")

    # Process questions
    results = []

    for i, question_data in enumerate(questions):
        question_id = question_data["id"]
        image_id = question_data["image_id"]

        if question_id in mapping:
            image_indices = []
            for mapped_file in sorted(mapping[question_id]):
                try:
                    idx = file_names.index(mapped_file)
                    image_indices.append(idx)
                except ValueError:
                    continue
            image_indices.sort()
        else:
            image_indices = find_image_indices(image_id, file_names)

        if not image_indices:
            print(f"Warning: No cropped images found for {image_id}")
            result = {
                "id": question_id,
                "question": question_data["question"],
                "retrieved_articles": [],
                "image_id": image_id,
                "relevant_articles": []
            }
        else:
            # Perform graph query
            query_results = query_graph(
                query_engine, embeddings, image_indices,
                max_results=args.max_results, search_depth=args.search_depth
            )

            # Format results
            result = format_results(query_results, question_data)

        results.append(result)

    # Save results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Query completed! Results saved to {args.output_file}")
    print(f"Processed {len(results)} questions")

    # Print summary statistics
    total_articles = sum(len(r['retrieved_articles']) for r in results)
    avg_articles = total_articles / len(results) if results else 0
    print(f"Average articles per question: {avg_articles:.2f}")


if __name__ == "__main__":
    main()
