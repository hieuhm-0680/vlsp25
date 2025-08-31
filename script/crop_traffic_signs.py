import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm


class TrafficSignCropper:
    """Class to handle traffic sign detection and cropping using Grounding DINO."""

    def __init__(self, model_id="IDEA-Research/grounding-dino-base"):
        """
        Initialize the traffic sign cropper.

        Args:
            model_id: HuggingFace model ID for Grounding DINO
        """
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_prompt = "traffic sign."
        self.box_threshold = 0.2
        self.text_threshold = 0.2

        print(f"Using device: {self.device}")
        print("Loading DINO Grounding model...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device)
        print("Model loaded successfully!")

    def detect_traffic_signs(self, image_path):
        """
        Detect traffic signs in an image.

        Args:
            image_path: Path to the input image

        Returns:
            dict: Detection results with boxes, scores, and labels
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")

            # Prepare inputs
            inputs = self.processor(
                images=image, text=self.text_prompt, return_tensors="pt").to(self.device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[image.size[::-1]]
            )

            result = results[0]

            # Apply thresholds to filter results
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []

            if len(result["boxes"]) > 0:
                for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                    if score >= self.box_threshold:
                        filtered_boxes.append(box.tolist())
                        filtered_scores.append(score.item())
                        filtered_labels.append(label)

            return {
                "success": True,
                "image_size": image.size,
                "boxes": filtered_boxes,
                "scores": filtered_scores,
                "labels": filtered_labels,
                "num_detections": len(filtered_boxes)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "num_detections": 0
            }

    def crop_and_save_traffic_signs(self, image_path, detection_result, output_dir):
        """
        Crop detected traffic signs and save them to output directory.

        Args:
            image_path: Path to the original image
            detection_result: Detection results from detect_traffic_signs
            output_dir: Directory to save cropped images

        Returns:
            list: List of saved crop information
        """
        if not detection_result["success"] or detection_result["num_detections"] == 0:
            return []

        try:
            # Load original image
            image = Image.open(image_path).convert("RGB")
            image_name = Path(image_path).stem

            saved_crops = []

            # Crop each detected traffic sign
            for i, (box, score, label) in enumerate(zip(
                detection_result["boxes"],
                detection_result["scores"],
                detection_result["labels"]
            )):
                # Convert box coordinates
                x1, y1, x2, y2 = box
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.width, int(x2))
                y2 = min(image.height, int(y2))

                # Skip if box is too small or invalid
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                # Crop the image
                cropped_image = image.crop((x1, y1, x2, y2))

                # Create filename for the cropped image (without confidence score)
                clean_label = label.replace(" ", "_").replace(".", "")
                crop_filename = f"{image_name}_crop_{i+1}_{clean_label}.jpg"
                crop_path = output_dir / crop_filename

                # Save the cropped image
                cropped_image.save(crop_path, quality=95)

                saved_crops.append({
                    "crop_filename": crop_filename,
                    "crop_path": str(crop_path),
                    "box": box,
                    "label": label,
                    "crop_size": cropped_image.size
                })

            return saved_crops

        except Exception as e:
            print(f"Error cropping traffic signs from {image_path}: {str(e)}")
            return []

    def process_folder(self, input_folder, output_folder):
        """
        Process all images in input folder and save cropped traffic signs to output folder.

        Args:
            input_folder: Path to input image folder
            output_folder: Path to output folder for cropped images
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        # Validate input folder
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(
                f"Input folder does not exist or is not a directory: {input_folder}")

        # Create output folder if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Get list of image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        image_files = list(set(image_files))  # Remove duplicates
        image_files.sort()

        if not image_files:
            print(f"No image files found in {input_folder}")
            return

        print(f"Found {len(image_files)} images to process")
        print(f"Output folder: {output_folder}")

        # Process statistics
        stats = {
            "total_images": len(image_files),
            "processed": 0,
            "errors": 0,
            "total_detections": 0,
            "images_with_detections": 0,
            "total_cropped_saved": 0
        }

        # Process images with progress bar
        for image_path in tqdm(image_files, desc="Processing images"):
            # Detect traffic signs
            detection_result = self.detect_traffic_signs(image_path)

            # Update statistics
            stats["processed"] += 1
            if not detection_result["success"]:
                stats["errors"] += 1
                print(detection_result["error"])
                continue

            stats["total_detections"] += detection_result["num_detections"]
            if detection_result["num_detections"] > 0:
                stats["images_with_detections"] += 1

                # Crop and save traffic signs
                saved_crops = self.crop_and_save_traffic_signs(
                    image_path, detection_result, output_path)
                stats["total_cropped_saved"] += len(saved_crops)

        # Print final statistics
        print("\n" + "="*60)
        print("PROCESSING COMPLETED")
        print("="*60)
        print(f"Total images processed: {stats['total_images']}")
        print(
            f"Successfully processed: {stats['processed'] - stats['errors']}")
        print(f"Errors encountered: {stats['errors']}")
        print(f"Images with detections: {stats['images_with_detections']}")
        print(f"Total traffic signs detected: {stats['total_detections']}")
        print(f"Total cropped images saved: {stats['total_cropped_saved']}")
        if stats['processed'] > 0:
            print(
                f"Average detections per image: {stats['total_detections'] / stats['processed']:.2f}")
        print(f"Output folder: {output_folder}")
        print("="*60)


def main():
    """Main function to handle command line arguments and run the cropper."""
    parser = argparse.ArgumentParser(
        description="Crop traffic signs from images using Grounding DINO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python crop_traffic_signs.py /path/to/images /path/to/output
    python crop_traffic_signs.py ./input_images ./cropped_signs
        """
    )

    parser.add_argument(
        "input_folder",
        help="Path to folder containing input images"
    )

    parser.add_argument(
        "output_folder",
        help="Path to folder where cropped traffic signs will be saved"
    )

    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-base",
        help="HuggingFace model ID for Grounding DINO (default: IDEA-Research/grounding-dino-base)"
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.2,
        help="Box threshold for object detection (default: 0.2)"
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.2,
        help="Text threshold for object detection (default: 0.2)"
    )

    args = parser.parse_args()

    try:
        # Initialize the cropper
        cropper = TrafficSignCropper(model_id=args.model_id)

        # Update thresholds if provided
        cropper.box_threshold = args.box_threshold
        cropper.text_threshold = args.text_threshold

        # Process the folder
        cropper.process_folder(args.input_folder, args.output_folder)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
