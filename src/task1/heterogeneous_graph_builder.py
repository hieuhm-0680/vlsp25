import json
import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import networkx as nx
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TextNode:
    node_id: str  # "law_id:article_id"
    law_id: str
    article_id: str
    title: str
    text: str


@dataclass
class ImageNode:
    node_id: str  # "law_image_id:sequence"
    filename: str
    filepath: str


@dataclass
class TableNode:
    node_id: str  # "table:law_id:article_id:index"
    law_id: str
    article_id: str
    table_index: int
    html_table: str
    parsed_table: Optional[pd.DataFrame] = None


class HeterogeneousGraphBuilder:
    def __init__(self,
                 law_db_path: str,
                 law_images_dir: str):
        self.law_db_path = law_db_path
        self.law_images_dir = law_images_dir

        # Initialize graph
        self.graph = nx.Graph()

        # Storage for nodes
        self.text_nodes: Dict[str, TextNode] = {}
        self.image_nodes: Dict[str, ImageNode] = {}
        self.table_nodes: Dict[str, TableNode] = {}

        # Pattern matchers
        self.image_pattern = re.compile(r'<<IMAGE:\s*([^>]+)\s*/IMAGE>>')
        self.table_pattern = re.compile(
            r'<<TABLE:\s*(.*?)\s*/TABLE>>', re.DOTALL)
        # khoản 55.2
        # Phụ lục C. # Phu luc C
        # Phu luc B (muc B.9)
        # Capture: Điều|điều|khoản|Khoản <number>, Khoản <number>.<number>, Phụ lục|Phu luc <A-Z>
        self.article_ref_pattern = re.compile(
            r'(?:Điều|Khoản)\s+(\d+(?:\.\d+)?)'
            r'|Phụ lục\s*(?:[A-Z]\s*\(mục\s*([A-Z0-9.]+)\)|([A-Z]))',
            re.IGNORECASE
        )

    def load_law_database(self) -> Dict[str, Any]:
        """Load law database from JSON file"""
        logger.info(f"Loading law database from {self.law_db_path}")
        with open(self.law_db_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_images_from_text(self, text: str) -> List[str]:
        """Extract image filenames from text using pattern matching"""
        matches = self.image_pattern.findall(text)
        return [match.strip() for match in matches]

    def extract_tables_from_text(self, text: str) -> List[str]:
        """Extract table HTML from text using pattern matching"""
        matches = self.table_pattern.findall(text)
        return [match.strip() for match in matches]

    def extract_article_references(self, text: str) -> Set[str]:
        """Extract article references from text"""
        matches = self.article_ref_pattern.findall(text)
        references = set(
            next(g for g in m if g) for m in matches
        )
        logger.debug(f"Extracted references: {references} from text")
        return references

    def parse_html_table(self, html_table: str) -> Optional[pd.DataFrame]:
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            if table:
                rows = []
                for tr in table.find_all('tr'):
                    row = []
                    for td in tr.find_all(['td', 'th']):
                        text = td.get_text(strip=True)
                        row.append(text)
                    if row:
                        rows.append(row)

                if rows:
                    if len(rows) > 1:
                        df = pd.DataFrame(rows[1:], columns=rows[0])
                    else:
                        df = pd.DataFrame(rows)
                    return df
        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
        return None

    def create_text_nodes(self, law_db: List[Dict[str, Any]]):
        logger.info("Creating text nodes from law database")

        for law_doc in law_db:
            law_id = law_doc['id']
            articles = law_doc['articles']

            for article in articles:
                article_id = article['id']
                title = article['title']
                text = article['text']

                if not article_id:
                    continue

                node_id = f"{law_id}:{article_id}"

                # Create text node
                text_node = TextNode(
                    node_id=node_id,
                    law_id=law_id,
                    article_id=article_id,
                    title=title,
                    text=text
                )

                self.text_nodes[node_id] = text_node

                # Add to graph
                self.graph.add_node(
                    node_id,
                    node_type='text',
                    law_id=law_id,
                    article_id=article_id,
                    title=title,
                    text=text[:500] + '...' if len(text) > 500 else text
                )

        logger.info(f"Created {len(self.text_nodes)} text nodes")

    def create_image_nodes(self, image_dir: str):
        logger.info("Creating image nodes")
        # Sort for deterministic order
        image_files = sorted(os.listdir(image_dir))
        count = 0
        for img_file in image_files:
            parts = img_file.split('_')
            image_law_id = parts[0][-3:]
            if int(image_law_id) % 2 == 1:  # rule-base to avoid duplicate images
                continue
            sequence = parts[2]
            image_id = f"image:{image_law_id}:{sequence}"
            image_node = ImageNode(
                node_id=image_id,
                filename=img_file,
                filepath=os.path.join(image_dir, img_file)
            )
            self.image_nodes[image_id] = image_node

            self.graph.add_node(
                image_id,
                node_type='image',
                filename=img_file,
                filepath=os.path.join(image_dir, img_file),
                source='law_database'
            )
            count += 1
        logger.info(
            f"Created {count} image nodes from {len(image_files)} files in {image_dir}")

    def create_table_nodes(self, law_db: List[Dict[str, Any]]):
        """Create table nodes from tables in law text"""
        logger.info("Creating table nodes")

        for law_doc in law_db:
            law_id = law_doc.get('id', '')
            articles = law_doc.get('articles', [])

            for article in articles:
                article_id = article.get('id', '')
                text = article.get('text', '')

                # Extract tables from text
                table_htmls = self.extract_tables_from_text(text)

                for i, table_html in enumerate(table_htmls):
                    table_node_id = f"table:{law_id}:{article_id}:{i}"

                    # Parse HTML table
                    parsed_table = self.parse_html_table(table_html)

                    # Create table node
                    table_node = TableNode(
                        node_id=table_node_id,
                        law_id=law_id,
                        article_id=article_id,
                        table_index=i,
                        html_table=table_html,
                        parsed_table=parsed_table
                    )

                    self.table_nodes[table_node_id] = table_node

                    # Add to graph
                    self.graph.add_node(
                        table_node_id,
                        node_type='table',
                        law_id=law_id,
                        article_id=article_id,
                        table_index=i,
                        html_table=table_html[:200] +
                        '...' if len(table_html) > 200 else table_html,
                        has_parsed_data=parsed_table is not None
                    )

        logger.info(f"Created {len(self.table_nodes)} table nodes")

    def create_image_article_edges(self, law_db: List[Dict[str, Any]]):
        logger.info("Creating image_article edges")

        edge_count = 0
        skip_count = 0
        for law_doc in law_db:
            law_id = law_doc.get('id', '')
            articles = law_doc.get('articles', [])

            for article in articles:
                article_id = article.get('id', '')
                text = article.get('text', '')

                article_node_id = f"{law_id}:{article_id}"

                # Find images in this article
                image_files = self.extract_images_from_text(text)

                for img_file in image_files:
                    id_pattern = re.compile('^image(\d+)')
                    img_id = id_pattern.match(img_file).group(1)
                    if int(img_id) > 500:
                        skip_count += 1
                        continue
                    count_img_ref_to_img_id = 0
                    for image_node_id in sorted(self.image_nodes.keys()):
                        if image_node_id.startswith(f"image:{img_id}"):
                            self.graph.add_edge(
                                article_node_id,
                                image_node_id,
                                edge_type='image_article'
                            )
                            edge_count += 1
                            count_img_ref_to_img_id += 1
        '''
        rule base edge
        # 21 -> 26: prohibited
        # 27 -> 30: warning
        # 31 -> 33: hieu lenh
        # 35 -> 40: bien chi dan
        # 41 -> 44: bien phu
        # 45 -> 47: bien chi dan tren duong cao toc
        # 48 -> 51: vach ke duong
        # 52 -> 55: coc tieu
        '''
        rule_base_edges = [
            list(range(21, 27)),
            list(range(27, 31)),
            list(range(31, 34)),
            list(range(35, 41)),
            list(range(41, 45)),
            list(range(45, 48)),
            list(range(48, 52)),
            list(range(52, 56))
        ]
        for rule_base_edge in rule_base_edges:
            for u in rule_base_edge:
                for v in rule_base_edge:
                    if u != v:
                        u_node_id = f"QCVN 41:2024/BGTVT:{u}"
                        v_node_id = f"QCVN 41:2024/BGTVT:{v}"
                        if u_node_id in self.text_nodes and v_node_id in self.text_nodes:
                            self.graph.add_edge(
                                u_node_id,
                                v_node_id,
                                edge_type='rule_base_edge',
                            )
                            edge_count += 1
                        else:
                            logger.warning(
                                f"Rule base edge {u_node_id} -> {v_node_id} not found in text nodes")

        logger.info(f"Created {edge_count} image_article edges")
        logger.info(f"Skipped {skip_count} images with IDs > 500")

    def create_article_table_edges(self, law_db: List[Dict[str, Any]]):
        logger.info("Creating article_table edges")

        edge_count = 0
        for law_doc in law_db:
            law_id = law_doc.get('id', '')
            articles = law_doc.get('articles', [])

            for article in articles:
                article_id = article.get('id', '')
                text = article.get('text', '')

                article_node_id = f"{law_id}:{article_id}"

                # Find tables in this article
                table_htmls = self.extract_tables_from_text(text)

                for i, _ in enumerate(table_htmls):
                    table_node_id = f"table:{law_id}:{article_id}:{i}"

                    if table_node_id in self.table_nodes:
                        self.graph.add_edge(
                            article_node_id,
                            table_node_id,
                            edge_type='article_table'
                        )
                        edge_count += 1

        logger.info(f"Created {edge_count} article_table edges")

    def create_articles_articles_edges(self, law_db: List[Dict[str, Any]]):
        logger.info("Creating articles_articles edges")

        edge_count = 0
        for law_doc in law_db:
            law_id = law_doc.get('id', '')
            articles = law_doc.get('articles', [])

            for article in articles:
                article_id = article.get('id', '')
                text = article.get('text', '')

                if not article_id:
                    continue

                article_node_id = f"{law_id}:{article_id}"

                # Extract article references
                referenced_article_ids = self.extract_article_references(text)

                for ref_article_id in referenced_article_ids:
                    if ref_article_id == article_id:
                        continue
                    if ref_article_id.isalpha():
                        for node_id in self.text_nodes:
                            if node_id.startswith(f"{law_id}:{ref_article_id}"):
                                self.graph.add_edge(
                                    article_node_id,
                                    node_id,
                                    edge_type='articles_articles',
                                    reference_type='implicit_mention'
                                )
                                edge_count += 1

                    else:
                        if ref_article_id.split('.')[0].isdigit():
                            ref_node_id = f"{law_id}:{ref_article_id.split('.')[0]}"
                        else:
                            ref_node_id = f"{law_id}:{ref_article_id}"
                        if ref_node_id in self.text_nodes:
                            self.graph.add_edge(
                                article_node_id,
                                ref_node_id,
                                edge_type='articles_articles',
                                reference_type='explicit_mention'
                            )
                            edge_count += 1
                        else:
                            logger.warning(
                                f"Referenced article node {ref_node_id} not found for article {article_node_id}")

        logger.info(f"Created {edge_count} articles_articles edges")

    def remove_self_loops(self):
        """Remove self-loops from the graph"""
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))

    def build_graph(self) -> nx.Graph:
        """Build the complete heterogeneous graph"""
        logger.info("Building heterogeneous graph")

        law_db = self.load_law_database()

        self.create_text_nodes(law_db)
        self.create_image_nodes(self.law_images_dir)
        self.create_table_nodes(law_db)

        self.create_image_article_edges(law_db)
        self.create_article_table_edges(law_db)
        self.create_articles_articles_edges(law_db)

        self.remove_self_loops()

        logger.info("Graph construction completed")
        logger.info(
            f"Graph summary: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

        return self.graph

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {}
        }

        # Count nodes by type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(
                node_type, 0) + 1

        # Count edges by type
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(
                edge_type, 0) + 1

        return stats

    def save_graph(self, output_path: str, format: str = 'graphml'):
        """Save graph to file"""
        logger.info(f"Saving graph to {output_path} in {format} format")

        if format == 'gpickle':
            # Use pickle directly since networkx.write_gpickle might not be available
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(self.graph, f)
        elif format == 'gml':
            nx.write_gml(self.graph, output_path)
        elif format == 'graphml':
            nx.write_graphml(self.graph, output_path)
        elif format == 'json':
            # Save as JSON for better compatibility
            from networkx.readwrite import json_graph
            data = json_graph.node_link_data(self.graph)
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Graph saved successfully")

    def load_graph(self, input_path: str, format: str = 'graphml') -> nx.Graph:
        """Load graph from file"""
        logger.info(f"Loading graph from {input_path}")

        if format == 'gpickle':
            import pickle
            with open(input_path, 'rb') as f:
                self.graph = pickle.load(f)
        elif format == 'gml':
            self.graph = nx.read_gml(input_path)
        elif format == 'graphml':
            self.graph = nx.read_graphml(input_path)
        elif format == 'json':
            from networkx.readwrite import json_graph
            import json
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.graph = json_graph.node_link_graph(
                data, directed=True, multigraph=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Graph loaded successfully")
        return self.graph


if __name__ == "__main__":
    builder = HeterogeneousGraphBuilder(
        law_db_path="dataset/vlsp25/law_db/vlsp2025_law_new.json",
        law_images_dir="images/law_images"
    )

    # Build graph
    graph = builder.build_graph()

    # Print statistics
    stats = builder.get_graph_statistics()
    print("Graph Statistics:")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Node types: {stats['node_types']}")
    print(f"Edge types: {stats['edge_types']}")

    # Save graph
    builder.save_graph(
        f"./heterogeneous_graph.json", format='json')
