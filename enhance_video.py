"""
Face-Focused Video Enhancement System
======================================
A modular Python pipeline for enhancing faces in low-quality videos.
No audio processing - silent output only.

Author: AI Systems Architect
License: MIT
"""

# ============================================================================
# video_reader.py
# ============================================================================

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Generator, Optional

class VideoReader:
    """Handles video loading and frame extraction."""
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to input video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Extract metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video loaded: {self.width}x{self.height} @ {self.fps:.2f} FPS")
        print(f"   Total frames: {self.frame_count}")
    
    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames one at a time.
        
        Yields:
            Tuple of (frame_index, frame_array)
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    
    def get_metadata(self) -> dict:
        """Return video metadata."""
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count
        }
    
    def release(self):
        """Release video capture resources."""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============================================================================
# face_restoration.py
# ============================================================================

import torch
from typing import Optional
import sys

class FaceRestorer:
    """Handles face detection and restoration using GFPGAN or CodeFormer."""
    
    def __init__(self, model_name: str = 'gfpgan', device: str = 'auto'):
        """
        Initialize face restoration model.
        
        Args:
            model_name: 'gfpgan' or 'codeformer'
            device: 'cpu', 'cuda', or 'auto'
        """
        self.model_name = model_name.lower()
        self.device = self._setup_device(device)
        self.model = None
        self.face_helper = None
        
        print(f"🤖 Initializing {self.model_name.upper()} on {self.device}")
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Determine computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """Load the face restoration model."""
        try:
            if self.model_name == 'gfpgan':
                self._load_gfpgan()
            elif self.model_name == 'codeformer':
                self._load_codeformer()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("⚠️  Make sure you have installed the required packages:")
            print("   pip install gfpgan realesrgan")
            sys.exit(1)
    
    def _load_gfpgan(self):
        """Load GFPGAN model."""
        try:
            from gfpgan import GFPGANer
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
            
            # Initialize face helper for detection
            self.face_helper = FaceRestoreHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self.device
            )
            
            # Initialize GFPGAN
            model_path = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            self.model = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
            print("✅ GFPGAN loaded successfully")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Install with: pip install gfpgan facexlib")
            sys.exit(1)
    
    def _load_codeformer(self):
        """Load CodeFormer model (placeholder - implement if needed)."""
        print("⚠️  CodeFormer support not yet implemented")
        print("   Falling back to GFPGAN")
        self._load_gfpgan()
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect and enhance all faces in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Enhanced frame with restored faces
        """
        if self.model is None:
            return frame
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Enhance with GFPGAN
            _, _, output = self.model.enhance(
                frame_rgb,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5  # Blending weight for natural results
            )
            
            # Convert back to BGR
            if output is not None:
                return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            else:
                return frame
                
        except Exception as e:
            print(f"⚠️  Frame enhancement failed: {e}")
            return frame


# ============================================================================
# upscaler.py
# ============================================================================

class VideoUpscaler:
    """Handles full-frame super-resolution using Real-ESRGAN."""
    
    def __init__(self, scale: int = 2, device: str = 'auto'):
        """
        Initialize upscaler.
        
        Args:
            scale: Upscaling factor (1, 2, or 4)
            device: Computation device
        """
        self.scale = scale
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.upsampler = None
        
        if scale > 1:
            print(f"🔍 Initializing Real-ESRGAN {scale}x upscaler on {self.device}")
            self._load_upsampler()
    
    def _load_upsampler(self):
        """Load Real-ESRGAN model."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Select model based on scale
            if self.scale == 2:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
                model_name = 'RealESRGAN_x2plus'
            else:  # scale == 4
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                model_name = 'RealESRGAN_x4plus'
            
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == 'cuda' else False,
                device=self.device
            )
            print(f"✅ {model_name} loaded successfully")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Install with: pip install realesrgan")
            sys.exit(1)
    
    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Upscaled frame
        """
        if self.scale == 1 or self.upsampler is None:
            return frame
        
        try:
            output, _ = self.upsampler.enhance(frame, outscale=self.scale)
            return output
        except Exception as e:
            print(f"⚠️  Upscaling failed: {e}")
            return frame


# ============================================================================
# pipeline.py
# ============================================================================

import time
from tqdm import tqdm

class EnhancementPipeline:
    """Orchestrates the complete video enhancement pipeline."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        face_model: str = 'gfpgan',
        upscale: int = 1,
        device: str = 'auto'
    ):
        """
        Initialize the enhancement pipeline.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            face_model: Face restoration model name
            upscale: Upscaling factor
            device: Computation device
        """
        self.input_path = input_path
        self.output_path = output_path
        
        print("="*60)
        print("🎬 Face-Focused Video Enhancement Pipeline")
        print("="*60)
        
        # Initialize components
        self.reader = VideoReader(input_path)
        self.face_restorer = FaceRestorer(model_name=face_model, device=device)
        self.upscaler = VideoUpscaler(scale=upscale, device=device)
        
        # Video writer will be initialized after first frame
        self.writer = None
        self.metadata = self.reader.get_metadata()
    
    def process(self):
        """Execute the full enhancement pipeline."""
        start_time = time.time()
        
        try:
            with self.reader:
                # Process frames with progress bar
                for frame_idx, frame in tqdm(
                    self.reader.read_frames(),
                    total=self.metadata['frame_count'],
                    desc="Enhancing",
                    unit="frame"
                ):
                    # Step 1: Face restoration
                    enhanced_frame = self.face_restorer.enhance_frame(frame)
                    
                    # Step 2: Optional upscaling
                    if self.upscaler.scale > 1:
                        enhanced_frame = self.upscaler.upscale_frame(enhanced_frame)
                    
                    # Initialize writer on first frame
                    if self.writer is None:
                        self._initialize_writer(enhanced_frame.shape)
                    
                    # Write frame
                    self.writer.write(enhanced_frame)
        
        finally:
            self._cleanup()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Enhancement complete in {elapsed:.1f}s")
        print(f"📁 Output saved to: {self.output_path}")
    
    def _initialize_writer(self, frame_shape: Tuple[int, int, int]):
        """Initialize video writer with appropriate codec."""
        height, width = frame_shape[:2]
        
        # Use mp4v codec for MP4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.metadata['fps'],
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError("Failed to initialize video writer")
        
        print(f"📝 Writing {width}x{height} @ {self.metadata['fps']:.2f} FPS")
    
    def _cleanup(self):
        """Release all resources."""
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()


# ============================================================================
# main.py
# ============================================================================

import argparse

def main():
    """Main entry point with CLI argument parsing."""
    
    parser = argparse.ArgumentParser(
        description='Face-Focused Video Enhancement System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic face enhancement
  python enhance_video.py --input input.mp4 --output output.mp4
  
  # With 2x upscaling
  python enhance_video.py --input input.mp4 --output output.mp4 --upscale 2
  
  # Force CPU processing
  python enhance_video.py --input input.mp4 --output output.mp4 --device cpu

Ethical Notice:
  Face restoration generates plausible but not guaranteed accurate details.
  DO NOT use for legal, forensic, or identity verification purposes.
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input video path (.mp4 or .avi)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output video path (.mp4 recommended)'
    )
    
    parser.add_argument(
        '--face_model',
        type=str,
        choices=['gfpgan', 'codeformer'],
        default='gfpgan',
        help='Face restoration model (default: gfpgan)'
    )
    
    parser.add_argument(
        '--upscale',
        type=int,
        choices=[1, 2, 4],
        default=1,
        help='Upscaling factor (default: 1 = no upscaling)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Computation device (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Display ethical disclaimer
    print("\n" + "="*60)
    print("⚠️  ETHICAL DISCLAIMER")
    print("="*60)
    print("Face restoration models generate PLAUSIBLE but not")
    print("necessarily ACCURATE facial details. This output must")
    print("NOT be used for:")
    print("  • Legal or forensic purposes")
    print("  • Identity verification")
    print("  • Creating misleading content")
    print("="*60 + "\n")
    
    # Execute pipeline
    try:
        pipeline = EnhancementPipeline(
            input_path=args.input,
            output_path=args.output,
            face_model=args.face_model,
            upscale=args.upscale,
            device=args.device
        )
        pipeline.process()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())