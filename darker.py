import cv2
import os
import numpy as np
from tqdm import tqdm
import random
from typing import Optional, Union
from pathlib import Path


class Darker:
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        ratio: float = 0.5,
        phase: str = "train"
    ):
        """Initialize the Darker class for image/video darkening.

        Args:
            data_dir: Root directory of the dataset. Required for image processing.
                If None, process_images method cannot be called.
            ratio: Brightness reduction ratio between 0 and 1.
            phase: Processing phase, either "train" or "test".

        Raises:
            ValueError: If ratio is not between 0 and 1 or phase is invalid.
        """
        if not 0 <= ratio <= 1:
            raise ValueError("Ratio must be between 0 and 1")
        if phase not in ["train", "test"]:
            raise ValueError('Phase must be either "train" or "test"')

        self.ratio = ratio
        self.phase = phase
        self.data_dir = Path(data_dir) if data_dir else None

        if self.data_dir:
            base_dir = "our485" if phase == "train" else "eval15"
            self.high_dir = self.data_dir / base_dir / "high"
            self.low_dir = self.data_dir / base_dir / "low"
            os.makedirs(self.low_dir, exist_ok=True)

            if not self.high_dir.exists():
                raise FileNotFoundError(
                    f"High-quality images directory not found: {self.high_dir}"
                )

    @staticmethod
    def adjust_image(img: np.ndarray, ratio: float) -> np.ndarray:
        """Apply darkening effect to the input image.

        Args:
            img: Input image in BGR format.
            ratio: Brightness reduction ratio.

        Returns:
            Darkened image in BGR format.
        """
        if img is None:
            raise ValueError("Input image cannot be None")

        seed = random.uniform(0.5, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio * seed, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_images(self) -> None:
        """Batch process images to generate low-light versions.

        Raises:
            RuntimeError: If data_dir was not provided during initialization.
            FileNotFoundError: If no valid images found in high_dir.
        """
        if not self.data_dir:
            raise RuntimeError(
                "Data directory not provided during initialization")

        image_files = [
            f for f in os.listdir(self.high_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        if not image_files:
            raise FileNotFoundError(
                f"No valid images found in {self.high_dir}")

        print(f"Processing {len(image_files)} images...")
        for image_file in tqdm(image_files):
            high_img_path = self.high_dir / image_file
            high_img = cv2.imread(str(high_img_path))

            if high_img is None:
                print(f"Warning: Could not read image: {high_img_path}")
                continue

            try:
                dark_img = self.adjust_image(high_img, self.ratio)
                low_img_path = self.low_dir / image_file
                cv2.imwrite(str(low_img_path), dark_img)
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

        print("Image processing completed! Please check the dataset manually.")

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path] = "dark_video.mp4"
    ) -> None:
        """Generate a low-light version of the input video.

        Args:
            video_path: Path to the source video file.
            output_path: Path for the output darkened video.

        Raises:
            FileNotFoundError: If the input video file doesn't exist.
            RuntimeError: If video processing fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Ignore false positive linter error for VideoWriter_fourcc
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            out = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height))

            print(f"Processing video with {total_frames} frames...")
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    dark_frame = self.adjust_image(frame, self.ratio)
                    out.write(dark_frame)
                    pbar.update(1)

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"Video processing completed! Output saved to: {output_path}")


if __name__ == '__main__':
    # Example usage for image processing
    data_dir = "../datasets/NuScenes"  # Replace with your dataset path
    ratio = 0.1  # Brightness reduction ratio

    # Process both train and test phase image datasets
    for phase in ["train", "test"]:
        try:
            darker = Darker(data_dir, ratio=ratio, phase=phase)
            darker.process_images()
        except Exception as e:
            print(f"Error processing {phase} phase: {str(e)}")

    # Example usage for video processing
    # video_path = "examples/input.mp4"
    # darker_video = Darker(ratio=ratio)
    # darker_video.process_video(video_path, "examples/output_dark.mp4")
