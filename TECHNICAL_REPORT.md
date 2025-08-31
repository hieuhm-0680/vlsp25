# Technical Report
## Subtask 1 - Information Retrieval

### Overview

Our approach to subtask 1 implements a graph-based information retrieval system that leverages multi-modal representations of traffic law documents. The system combines visual traffic sign detection, heterogeneous graph construction, and advanced reranking techniques to retrieve relevant law articles based on traffic sign images and questions.

### System Architecture

The information retrieval pipeline consists of three main components:
1. **Data Processing & Feature Extraction**
2. **Heterogeneous Graph Construction**  
3. **Multi-Modal Query Processing**

## 1. Data Processing

### 1.1 Traffic Sign Detection and Cropping

We employ **Grounding DINO** (`IDEA-Research/grounding-dino-base`) for automated traffic sign detection in both law database images and query images.

**For Law Database Images:**
- Detect traffic signs using text prompt "traffic sign."
- Apply confidence thresholds (box_threshold=0.2, text_threshold=0.2)
- Crop and save only detected traffic signs, discarding other content
- Generate cropped images with naming pattern: `{image_name}_crop_{index}_{label}.jpg`

**For Train/Test Images:**
- Apply same detection pipeline but retain all detected crops
- Preserve original image structure for later reference

```bash
# Run traffic sign cropping
python crop_traffic_signs.py /path/to/law_images /path/to/cropped_output
python crop_traffic_signs.py /path/to/train_images /path/to/train_cropped
```

### 1.2 Visual Embedding Generation

All cropped traffic signs are encoded using **SigLIP** (`ViT-SO400M-14-SigLIP-384`) to generate high-dimensional visual embeddings for similarity search.

## 2. Heterogeneous Graph Construction

Our system builds a knowledge graph with three node types representing different modalities of legal information:

### 2.1 Node Types

**TextNode:** Represents individual law articles
- Node ID format: `{law_id}:{article_id}`
- Contains article title and clean text (IMAGE/TABLE tags removed)
- Extracted from law database JSON

**ImageNode:** Represents cropped traffic sign images
- Node ID format: `image:{law_image_id}:{sequence}`
- Links to physical image files from detection pipeline
- Contains metadata: filename, filepath

**TableNode:** Represents structured tabular data
- Node ID format: `table:{law_id}:{article_id}:{index}`
- Extracted from `<<TABLE:...>>` tags in article text
- Parsed into structured format when possible

### 2.2 Edge Types

**Text-Image Edges:** Connect articles to referenced images
- Established when article text contains `<<IMAGE:...>>` tags
- Links TextNode to corresponding ImageNode(s)

**Text-Table Edges:** Connect articles to embedded tables
- Created for tables extracted from article text
- Direct relationship between TextNode and TableNode

**Text-Text Edges:** Inter-article references
- Pattern-based detection: `Điều \d+`, `Khoản \d+`, `Phụ lục [A-Z]`
- Rule-based connections for related articles
- Enables traversal between conceptually related content

```python
# Build heterogeneous graph
from src.heterogeneous_graph_builder import HeterogeneousGraphBuilder

builder = HeterogeneousGraphBuilder(
    law_db_path="dataset/vlsp25/law_db/vlsp2025_law_new.json",
    law_images_dir="images/law_images"
)
graph = builder.build_graph()
builder.save_graph("heterogeneous_graph.json", format='json')
```

---

## 3. Query Processing Pipeline

### 3.1 Visual Document Reranking

For each input question, we first determine traffic sign relevance using **Jina Reranker** (`jinaai/jina-reranker-m0`):

1. Get all cropped images from the question image (processed by Grounding DINO in step 1.1)
2. Apply multimodal reranking with image-text pairs: `[image, "Câu hỏi: {question}\n\nTài liệu liên quan: {context}"]`
3. Sort by score and filter crops based on relevance scores, keeping those that have a score difference of no more than `threshold=0.05`
4. Identify `n_cropped` relevant traffic signs

### 3.2 Similarity Search & Graph Traversal

**Image Matching:**
- Encode relevant cropped images using SigLIP
- Perform cosine similarity search against law database embeddings
- Select top-1 most similar law image (min_similarity=0.65)
- Map to corresponding ImageNode in graph

**Graph-Based Retrieval:**
- Initialize BFS from matched ImageNode(s)
- Traverse graph with max_depth=3, targeting TextNode types
- Skip appendix articles (pattern: `QCVN 41:2024/BGTVT:[A-Z]\.\d+`) at depth > 1

### 3.3 Dynamic Top-K Selection

Apply adaptive result filtering based on visual relevance:
- **Top-K Strategy:** `k = 2 * n_cropped`
- Where `n_cropped` is the count of traffic signs deemed relevant by Jina reranker
- Ensures result set scales with question complexity

```python
# Query processing example
from src.graph_query_engine import GraphQueryEngine, MultiModalQuery

engine = GraphQueryEngine(graph)
engine.load_embeddings("embeddings/image_embeddings.npy", "embeddings/image_embeddings.txt")

query = MultiModalQuery(
    image_embedding=query_embedding,
    max_results=2 * n_cropped,
    search_depth=3
)
results = engine.multi_modal_query(query)
```

## 4. Implementation and Execution

### 4.1 Dependencies
- PyTorch, transformers, PIL for deep learning
- NetworkX for graph operations
- BeautifulSoup for HTML table parsing
- NumPy for embedding operations

### 4.2 Execution Workflow

```bash
# 1. Crop traffic signs from law and query images
python crop_traffic_signs.py images/law_images images/law_cropped
python crop_traffic_signs.py train_images train_cropped

# 2. Build heterogeneous graph
python src/heterogeneous_graph_builder.py

# 3. Generate visual embeddings (using SigLIP)
# Check the CLIP_Extract.ipynb notebook for details

# 4. Process queries with reranking
python visual_reranking.py

# 5. Execute graph-based retrieval
python query_script.py
```

## Subtask 2 - Visual Question Answering (VQA)

### Overview

Our approach to Subtask 2 adopts a three-stage pipeline—**Image Conditioning**, **Law Context Extraction**, and **Reasoning & Answer Selection**—to answer traffic-law questions given three inputs: a **question**, a set of **answer choices** (multiple-choice or True/False), and the **image attached to the question**. The pipeline (i) prepares visual inputs for clearer sign understanding, (ii) assembles a focused legal context from retrieved articles, and (iii) applies a vision-language reasoner to select the final answer with step-by-step prompting.

### System Architecture

The VQA pipeline consists of three main components:

1. **Image Conditioning**
2. **Law Context Extraction**
3. **Reasoning & Answer Selection**

---

## 1. Image Conditioning

### 1.1 Law Image Consolidation

For each relevant article (from Subtask 1 retrieval), all associated law images are collected and merged into a single composite image per article. This reduces multi-image prompt complexity and shortens context length, which helps mitigate hallucination and repetitive outputs during inference.

### 1.2 Question Image Preprocessing

For each question image (the image attached to the question), **Grounding DINO** is used to detect traffic-sign regions. Detected bounding boxes are drawn on the original image to visually highlight target signs in complex scenes, providing clearer attention anchors for downstream models.

---

## 2. Law Context Extraction

### 2.1 Inputs

- **Question text**
- **Annotated question image** (from §1.2)
- **Law text** (relevant article(s) from Subtask 1)
- **Merged law image** (from §1.1)

### 2.2 Inference Configuration

- **Model:** `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit` (vision-language)
- **Prompting:** Chain-of-thought–style prompts to encourage structured extraction of only information relevant to the given image and question.
- **Heuristic:** Questions whose relevant articles contain **no** law image are skipped at this stage (empirically reduced noise and improved downstream reasoning).

### 2.3 Procedure

The model receives the inputs above and is instructed to extract a minimal, faithful legal context tied to the visual evidence. The goal is not to answer the question here, but to produce a compact knowledge slice for the reasoning stage.

### 2.4 Output Schema

- **Detected sign names**
- **Law details per sign**
- **Concise conclusion summary**

### 2.5 Design Choices & Assumptions

- Answer choices are **not** provided at this stage to avoid bias; the model focuses on extraction rather than early answer selection.
- Articles can contain multiple signs and extraneous text; targeted extraction reduces irrelevant content before reasoning.

---

## 3. Reasoning & Answer Selection

### 3.1 Inputs

- **Question text**
- **Answer choices** (multiple-choice or True/False)
- **Annotated question image**
- **Extracted law context** (from §2)

### 3.2 Inference Configuration

- **Primary models:**

  - `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`
  - `OpenGVLab/InternVL3-8B`
- **Prompting:** Chain-of-thought–style (“think step by step”) to reason strictly over the extracted law context.
- **Efficiency note:** Law images are **not** re-injected here to keep context concise and reduce hallucinations; the curated law text serves as the authoritative source.

### 3.3 Outputs

- **Reasoning trace** (structured rationale grounded in extracted law)
- **Final selected choice** (from the provided answer choices)

---

## 4. Submissions

- **Submission 1:** Qwen2.5-VL-7B-Instruct (unsloth, 4-bit) as the reasoner
- **Submission 2:** InternVL3-8B as the reasoner
- **Submission 3:** Stacked ensemble of QwenVL and InternVL final choices
