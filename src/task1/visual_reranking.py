import json
import os
import argparse
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import AutoModel
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrafficSignReranker:
    """Traffic sign visual reranking using jinaai/jina-reranker-m0 model."""

    def __init__(self, device: str = "auto", use_flash_attention: bool = True):
        """
        Initialize the reranker with the jina model.

        Args:
            device: Device to run the model on ('auto', 'cuda', 'cpu')
            use_flash_attention: Whether to use flash attention (requires compatible GPU)
        """
        self.device = self._get_device(device)
        self.use_flash_attention = use_flash_attention and torch.cuda.is_available()
        self.model = self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self) -> AutoModel:
        """Load the jinaai/jina-reranker-m0 model."""
        logger.info(f"Loading jinaai/jina-reranker-m0 model on {self.device}")

        model_kwargs = {
            'torch_dtype': "auto",
            'trust_remote_code': True
        }

        if self.use_flash_attention:
            model_kwargs['attn_implementation'] = "flash_attention_2"
            logger.info("Using flash attention 2")

        try:
            model = AutoModel.from_pretrained(
                'jinaai/jina-reranker-m0', **model_kwargs)
            model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            if self.use_flash_attention:
                logger.warning(f"Failed to load with flash attention: {e}")
                logger.info("Retrying without flash attention...")
                model_kwargs.pop('attn_implementation', None)
                model = AutoModel.from_pretrained(
                    'jinaai/jina-reranker-m0', **model_kwargs)
                model.to(self.device)
                model.eval()
                return model
            else:
                raise e

    def find_cropped_images(self, image_id: str, crop_folder: str) -> List[str]:
        """
        Find all cropped images for a given image_id.

        Args:
            image_id: Base image identifier (e.g., "train_1_3")
            crop_folder: Path to the folder containing cropped images

        Returns:
            List of absolute paths to cropped images
        """
        # Find all matching cropped images
        pattern = os.path.join(
            crop_folder, f"{image_id}_crop_*_traffic_sign*")
        cropped_images = glob.glob(pattern)

        if not cropped_images:
            logger.warning(
                f"No cropped images found for {image_id} (pattern: {pattern})")

        return sorted(cropped_images)

    def compute_relevance_scores(self, question: str, image_paths: List[str],
                                 batch_size: int = 8) -> List[Tuple[str, float]]:
        """
        Compute relevance scores between a question and multiple images.

        Args:
            question: The traffic law question
            image_paths: List of paths to cropped images
            batch_size: Number of images to process in each batch

        Returns:
            List of tuples (image_path, score) sorted by score descending
        """
        if not image_paths:
            return []

        results = []

        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Construct pairs for this batch
            image_pairs = [[question, img_path] for img_path in batch_paths]

            try:
                # Compute scores
                scores = self.model.compute_score(
                    image_pairs, max_length=2048, doc_type="image")

                # Ensure scores is a list
                if not isinstance(scores, list):
                    scores = [scores]

                # Store results
                for img_path, score in zip(batch_paths, scores):
                    results.append((os.path.basename(img_path), float(score)))

            except Exception as e:
                logger.error(
                    f"Error processing batch {i//batch_size + 1}: {e}")
                # Add zero scores for failed batch
                for img_path in batch_paths:
                    results.append((os.path.basename(img_path), 0.0))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def filter_relevant_images(self, scored_images: List[Tuple[str, float]],
                               threshold: float = 0.5) -> List[str]:
        """
        Filter images based on relevance threshold.

        Args:
            scored_images: List of (image_name, score) tuples
            threshold: Minimum score threshold for relevance

        Returns:
            List of relevant image names
        """
        relevant = [(img_name, score) for img_name,
                    score in scored_images if score >= threshold]
        return relevant

    def process_questions(self, questions_file: str, crop_folder: str,
                          threshold: float = 0.5, batch_size: int = 8) -> Dict[str, List[str]]:
        """
        Process all questions and find relevant cropped images.

        Args:
            questions_file: Path to JSON file containing questions
            crop_folder: Path to folder containing cropped images
            threshold: Relevance score threshold
            batch_size: Batch size for processing images

        Returns:
            Dictionary mapping question_id to list of relevant image names
        """
        # Load questions
        logger.info(f"Loading questions from {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        results = {}

        logger.info(f"Processing {len(questions)} questions...")
        for question_data in tqdm(questions, desc="Processing questions"):
            question_id = question_data['id']
            question_text = question_data['question']
            image_id = question_data['image_id']

            # Find cropped images for this question
            cropped_images = self.find_cropped_images(image_id, crop_folder)

            if not cropped_images:
                logger.warning(
                    f"No cropped images found for question {question_id}")
                results[question_id] = []
                continue

            # Compute relevance scores
            scored_images = self.compute_relevance_scores(
                question_text, cropped_images, batch_size
            )

            # Filter relevant images
            relevant_images = self.filter_relevant_images(
                scored_images, threshold)

            results[question_id] = relevant_images

            logger.debug(f"Question {question_id}: {len(relevant_images)} relevant images "
                         f"out of {len(cropped_images)} total")

        return results


def main():
    """Main function to run the visual reranking script."""
    parser = argparse.ArgumentParser(
        description="Visual Documents Reranking for Traffic Signs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'questions_file', help='Path to JSON file containing questions')
    parser.add_argument(
        'crop_folder', help='Path to folder containing cropped images')
    parser.add_argument('output_file', help='Path to output JSON file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Relevance score threshold (default: 0.5)')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Device to run the model on (default: auto)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing images (default: 8)')
    parser.add_argument('--no-flash-attention', action='store_true',
                        help='Disable flash attention 2 (use if you have GPU compatibility issues)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input files
    if not os.path.exists(args.questions_file):
        logger.error(f"Questions file not found: {args.questions_file}")
        return 1

    if not os.path.exists(args.crop_folder):
        logger.error(f"Crop folder not found: {args.crop_folder}")
        return 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Initialize reranker
        reranker = TrafficSignReranker(
            device=args.device,
            use_flash_attention=not args.no_flash_attention
        )

        # Process questions
        results = reranker.process_questions(
            args.questions_file,
            args.crop_folder,
            threshold=args.threshold,
            batch_size=args.batch_size
        )

        # Save results
        logger.info(f"Saving results to {args.output_file}")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        total_questions = len(results)
        total_relevant = sum(len(images) for images in results.values())
        avg_relevant = total_relevant / total_questions if total_questions > 0 else 0

        logger.info(f"Processing complete!")
        logger.info(f"Total questions: {total_questions}")
        logger.info(f"Total relevant images found: {total_relevant}")
        logger.info(
            f"Average relevant images per question: {avg_relevant:.2f}")

        return 0

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
