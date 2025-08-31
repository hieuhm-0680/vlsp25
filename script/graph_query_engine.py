import re
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import json
from pathlib import Path

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Represents a query result with relevance score"""
    node_id: str
    node_type: str
    relevance_score: float
    data: Dict[str, Any]
    path_from_query: List[str]


@dataclass
class MultiModalQuery:
    """Represents a multi-modal query with text and image components"""
    text_query: Optional[str] = None
    image_query: Optional[str] = None
    image_embedding: Optional[np.ndarray] = None
    max_results: int = 10
    search_depth: int = 3


class GraphQueryEngine:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._image_embeddings: np.ndarray = np.empty((0, 0))
        self._image_files: List[str] = []
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load_embeddings(self,
                        npy_path: Optional[str] = None,
                        txt_path: Optional[str] = None):
        if npy_path and txt_path and Path(npy_path).exists() and Path(txt_path).exists():
            logger.info(
                f"Loading image embeddings from {npy_path} and image paths from {txt_path}")
            try:
                self._image_embeddings = np.load(npy_path)
                with open(txt_path, 'r') as f:
                    self._image_files = [line.strip()
                                         for line in f if line.strip()]
                if len(self._image_embeddings) != len(self._image_files):
                    logger.warning(
                        "Number of embeddings does not match number of image paths")
                    return
                logger.info(
                    f"Loaded {len(self._image_embeddings)} image embeddings")
            except Exception as e:
                logger.warning(f"Failed to load image embeddings: {e}")

    def _get_image_id_from_path(self, img_path: str) -> str:
        pattern = r'image(\d+)_crop_(\d+)'
        match = re.match(pattern, img_path)
        if match:
            id, seq = match.groups()
            return f"image:{id}:{seq}"
        return img_path

    def find_similar_images(self,
                            query_embedding: torch.Tensor,
                            top_k: int = 3,
                            min_similarity: float = 0.7) -> List[Tuple[str, float]]:
        similarities = F.cosine_similarity(
            query_embedding, torch.from_numpy(self._image_embeddings).to(self.device))

        top_k_indices = torch.topk(similarities, top_k).indices.cpu().numpy()

        results = [
            (self._get_image_id_from_path(
                self._image_files[i]), similarities[i].item())
            for i in top_k_indices if similarities[i].item() >= min_similarity
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def dfs_traversal(self,
                      start_nodes: List[str],
                      max_depth: int = 3,
                      node_types: Optional[Set[str]] = None,
                      edge_types: Optional[Set[str]] = None) -> Dict[str, Tuple[int, List[str]]]:
        visited = {}
        stack = []

        # Initialize with start nodes
        for start_node in start_nodes:
            if start_node in self.graph:
                visited[start_node] = (0, [start_node])
                stack.append((start_node, 0, [start_node]))

        while stack:
            current_node, depth, path = stack.pop()

            if depth >= max_depth:
                continue

            # Get all neighbors (undirected graph) - sort for deterministic order
            neighbors = set()

            for neighbor in sorted(self.graph.neighbors(current_node)):
                if edge_types:
                    # Check if edge has allowed type
                    edge_data = self.graph.get_edge_data(
                        current_node, neighbor)
                    if edge_data and edge_data.get('edge_type') in edge_types:
                        neighbors.add(neighbor)
                else:
                    neighbors.add(neighbor)

            # Process neighbors in sorted order for deterministic results
            for neighbor in sorted(neighbors):
                if neighbor not in visited:
                    # Check node type filter
                    if node_types:
                        neighbor_data = self.graph.nodes.get(neighbor, {})
                        neighbor_type = neighbor_data.get('node_type')
                        if neighbor_type not in node_types:
                            continue

                    new_path = path + [neighbor]
                    visited[neighbor] = (depth + 1, new_path)
                    stack.append((neighbor, depth + 1, new_path))

        return visited

    def bfs_traversal(self,
                      start_nodes: List[str],
                      max_depth: int = 3,
                      node_types: Optional[Set[str]] = None,
                      edge_types: Optional[Set[str]] = None) -> Dict[str, Tuple[int, List[str]]]:
        visited = {}
        queue = deque()

        # Initialize with start nodes
        for start_node in start_nodes:
            if start_node in self.graph:
                visited[start_node] = (0, [start_node])
                queue.append((start_node, 0, [start_node]))

        while queue:
            current_node, depth, path = queue.popleft()

            if depth >= max_depth:
                continue

            # Get all neighbors (undirected graph) - sort for deterministic order
            neighbors = set()

            for neighbor in sorted(self.graph.neighbors(current_node)):
                if depth > 1:
                    # skip neighbors in appendix (e.g 'QCVN 41:2024/BGTVT:D.12', 'QCVN 41:2024/BGTVT:F.9')
                    pattern = r'QCVN 41:2024/BGTVT:[A-Z]\.\d+'
                    if re.match(pattern, neighbor):
                        continue
                if edge_types:
                    # Check if edge has allowed type
                    edge_data = self.graph.get_edge_data(
                        current_node, neighbor)
                    if edge_data and edge_data.get('edge_type') in edge_types:
                        neighbors.add(neighbor)
                else:
                    neighbors.add(neighbor)

            # Process neighbors in sorted order for deterministic results
            for neighbor in sorted(neighbors):
                if neighbor not in visited:
                    # Check node type filter
                    if node_types:
                        neighbor_data = self.graph.nodes.get(neighbor, {})
                        neighbor_type = neighbor_data.get('node_type')
                        if neighbor_type not in node_types:
                            continue

                    new_path = path + [neighbor]
                    visited[neighbor] = (depth + 1, new_path)
                    queue.append((neighbor, depth + 1, new_path))

        return visited

    def multi_modal_query(self, query: MultiModalQuery) -> List[QueryResult]:
        all_results = []
        seed_nodes = set()

        # Image-based search
        if query.image_embedding is not None:
            query.image_embedding = torch.tensor(
                query.image_embedding, device=self.device)
            image_matches = self.find_similar_images(
                query.image_embedding,
                top_k=1,
                min_similarity=0.65
            )

            for node_id, score in image_matches:
                seed_nodes.add(node_id)
                node_data = self.graph.nodes.get(node_id, {})
                all_results.append(QueryResult(
                    node_id=node_id,
                    node_type='image',
                    relevance_score=score,
                    data=node_data,
                    path_from_query=[node_id]
                ))

        # Graph traversal to find related nodes
        if seed_nodes:
            # Perform BFS from seed nodes - sort seed nodes for deterministic order
            traversal_results = self.bfs_traversal(
                start_nodes=sorted(list(seed_nodes)),
                # max_depth=query.search_depth,
                max_depth=3,
                node_types={'text'}
            )

            # Perform DFS from seed nodes
            # traversal_results = self.dfs_traversal(
            #     start_nodes=sorted(list(seed_nodes)),
            #     max_depth=3,
            #     node_types={'text'},

            # )

            # Add related nodes found through traversal
            for node_id, (distance, path) in traversal_results.items():
                if node_id not in seed_nodes:  # Skip seed nodes already added
                    node_data = self.graph.nodes.get(node_id, {})
                    node_type = node_data.get('node_type', 'unknown')

                    # Calculate relevance score based on distance and path
                    # Closer nodes get higher scores
                    traversal_score = 1.0 / (distance + 1)

                    all_results.append(QueryResult(
                        node_id=node_id,
                        node_type=node_type,
                        relevance_score=traversal_score,
                        data=node_data,
                        path_from_query=path
                    ))

        seen_nodes = set()
        unique_results = []
        for result in all_results:
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        # Sort by relevance score descending
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique_results[:query.max_results]

    def find_articles_related_to_image(self, image_node_id: str) -> List[QueryResult]:
        """Find all articles related to a specific image"""
        if image_node_id not in self.graph:
            logger.warning(f"Image node {image_node_id} not found in graph")
            return []

        # Find all paths from image to text nodes
        traversal_results = self.bfs_traversal(
            start_nodes=[image_node_id],
            max_depth=3,
            node_types={'text'}
        )

        results = []
        for node_id, (distance, path) in traversal_results.items():
            if self.graph.nodes[node_id].get('node_type') == 'text':
                node_data = self.graph.nodes[node_id]
                score = 1.0 / (distance + 1)  # Closer = higher score

                results.append(QueryResult(
                    node_id=node_id,
                    node_type='text',
                    relevance_score=score,
                    data=node_data,
                    path_from_query=path
                ))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def find_tables_related_to_article(self, article_node_id: str) -> List[QueryResult]:
        """Find all tables related to a specific article"""
        if article_node_id not in self.graph:
            logger.warning(
                f"Article node {article_node_id} not found in graph")
            return []

        # Find all paths from article to table nodes
        traversal_results = self.bfs_traversal(
            start_nodes=[article_node_id],
            max_depth=2,
            node_types={'table'}
        )

        results = []
        for node_id, (distance, path) in traversal_results.items():
            if self.graph.nodes[node_id].get('node_type') == 'table':
                node_data = self.graph.nodes[node_id]
                score = 1.0 / (distance + 1)

                results.append(QueryResult(
                    node_id=node_id,
                    node_type='table',
                    relevance_score=score,
                    data=node_data,
                    path_from_query=path
                ))

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def get_node_neighborhood(self,
                              node_id: str,
                              radius: int = 1) -> Dict[str, Any]:
        """Get comprehensive information about a node's neighborhood"""
        if node_id not in self.graph:
            return {}

        traversal_results = self.bfs_traversal(
            start_nodes=[node_id],
            max_depth=radius
        )

        neighborhood = {
            'center_node': {
                'id': node_id,
                'data': self.graph.nodes[node_id]
            },
            'neighbors_by_distance': defaultdict(list),
            'neighbors_by_type': defaultdict(list),
            'edge_summary': defaultdict(int)
        }

        for neighbor_id, (distance, path) in traversal_results.items():
            if neighbor_id != node_id:  # Skip center node
                neighbor_data = self.graph.nodes[neighbor_id]
                neighbor_type = neighbor_data.get('node_type', 'unknown')

                neighborhood['neighbors_by_distance'][distance].append({
                    'id': neighbor_id,
                    'type': neighbor_type,
                    'data': neighbor_data,
                    'path': path
                })

                neighborhood['neighbors_by_type'][neighbor_type].append({
                    'id': neighbor_id,
                    'distance': distance,
                    'data': neighbor_data
                })

        # Count edge types
        for u, v, data in self.graph.edges(data=True):
            if u == node_id or v == node_id:
                edge_type = data.get('edge_type', 'unknown')
                neighborhood['edge_summary'][edge_type] += 1

        return dict(neighborhood)

    def export_query_results(self,
                             results: List[QueryResult],
                             output_path: str,
                             format: str = 'json') -> None:
        """Export query results to file"""
        if format == 'json':
            # Convert to JSON-serializable format
            export_data = []
            for result in results:
                export_data.append({
                    'node_id': result.node_id,
                    'node_type': result.node_type,
                    'relevance_score': result.relevance_score,
                    'path_length': len(result.path_from_query),
                    'path': result.path_from_query,
                    'data': {k: v for k, v in result.data.items()
                             if k not in ['text']}  # Exclude full text for brevity
                })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(results)} query results to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.append('/home/mhieu/git/vlsp25/src')
    from heterogeneous_graph_builder import HeterogeneousGraphBuilder

    # Load graph
    workspace_dir = "/home/mhieu/git/vlsp25"
    builder = HeterogeneousGraphBuilder(
        law_db_path="dataset/vlsp25/law_db/vlsp2025_law_new.json",
        law_images_dir="images/law_images"
    )

    graph = builder.build_graph()

    # Create query engine
    query_engine = GraphQueryEngine(graph)
    query_engine.load_embeddings(
        npy_path="embeddings/image_embeddings.npy",
        txt_path="embeddings/image_embeddings.txt",
    )

    # Example image query
    query = MultiModalQuery(
        image_embedding=torch.rand(1152),
        max_results=10,
        search_depth=2
    )

    results = query_engine.multi_modal_query(query)

    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(
            f"{i+1}. {result.node_id} ({result.node_type}) - Score: {result.relevance_score:.3f}")

    # Export results
    query_engine.export_query_results(
        results,
        f"{workspace_dir}/query_results.json"
    )
