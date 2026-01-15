"""
=====================================================================
INFERENCE SCRIPT - SỬ DỤNG MÔ HÌNH ĐỂ PHÁT HIỆN DEEPFAKE
=====================================================================
Mô tả:
    Script này sử dụng các models đã train để phát hiện deepfake
    trên ảnh mới.
    
    Các chức năng:
    1. Load models từ checkpoints
    2. Preprocess ảnh đầu vào
    3. Tính các anomaly scores từ từng module
    4. Fusion scores để ra quyết định cuối cùng
    5. Visualization kết quả
    
Cách sử dụng:
    # Dự đoán một ảnh
    python inference.py --image path/to/image.jpg
    
    # Dự đoán batch ảnh
    python inference.py --input_dir path/to/images/ --output_dir results/
    
    # Dự đoán với specific models
    python inference.py --image image.jpg --models vqvae,transformer,cnn_vit
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# Import các modules
from config import (
    DataConfig, VQVAEConfig, TransformerConfig,
    DiffusionConfig, CNNViTConfig, GCNConfig,
    FusionConfig, TrainingConfig
)
from vqvae_module import VQVAE
from transformer_module import ImageGPT
from diffusion_module import DDPM
from cnn_vit_module import CNNViTClassifier
from gcn_module import GCNModule
from fusion_module import DeepfakeFusionModule, DeepfakeDetectionPipeline
from data_preprocessing import get_val_transforms, FacePreprocessor


# =====================================================================
# MODEL LOADER
# =====================================================================

class ModelLoader:
    """
    Helper class để load các models từ checkpoints
    """
    
    def __init__(self, checkpoint_dir=None, device=None):
        """
        Khởi tạo model loader
        
        Args:
            checkpoint_dir: Thư mục chứa checkpoints
            device: Device để load models
        """
        if checkpoint_dir is None:
            checkpoint_dir = DataConfig.CHECKPOINT_DIR
        
        if device is None:
            device = TrainingConfig.DEVICE
        
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        print(f"📂 Checkpoint directory: {checkpoint_dir}")
        print(f"📱 Device: {device}")
    
    def load_vqvae(self, checkpoint_name='vqvae_best.pth'):
        """Load VQ-VAE model"""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        model = VQVAE().to(self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded VQ-VAE từ {path}")
        else:
            print(f"⚠️  VQ-VAE checkpoint không tìm thấy: {path}")
        
        model.eval()
        return model
    
    def load_transformer(self, checkpoint_name='transformer_best.pth'):
        """Load Transformer model"""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        model = ImageGPT().to(self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded Transformer từ {path}")
        else:
            print(f"⚠️  Transformer checkpoint không tìm thấy: {path}")
        
        model.eval()
        return model
    
    def load_ddpm(self, checkpoint_name='ddpm_best.pth'):
        """Load DDPM model"""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        model = DDPM().to(self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded DDPM từ {path}")
        else:
            print(f"⚠️  DDPM checkpoint không tìm thấy: {path}")
        
        model.eval()
        return model
    
    def load_cnn_vit(self, checkpoint_name='cnn_vit_best.pth'):
        """Load CNN/ViT model"""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        model = CNNViTClassifier(
            extractor_type='cnn',
            backbone_name=CNNViTConfig.BACKBONE_NAME
        ).to(self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded CNN/ViT từ {path}")
        else:
            print(f"⚠️  CNN/ViT checkpoint không tìm thấy: {path}")
        
        model.eval()
        return model
    
    def load_gcn(self, checkpoint_name='gcn_best.pth'):
        """Load GCN model"""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        model = GCNModule().to(self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded GCN từ {path}")
        else:
            print(f"⚠️  GCN checkpoint không tìm thấy: {path}")
        
        model.eval()
        return model
    
    def load_fusion(self, checkpoint_name='fusion_best.pth'):
        """Load Fusion model"""
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        model = DeepfakeFusionModule().to(self.device)
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded Fusion từ {path}")
        else:
            print(f"⚠️  Fusion checkpoint không tìm thấy: {path}")
        
        model.eval()
        return model


# =====================================================================
# DEEPFAKE DETECTOR
# =====================================================================

class DeepfakeDetector:
    """
    Main class để phát hiện deepfake
    
    Sử dụng tất cả các modules đã train để đưa ra quyết định.
    """
    
    def __init__(
        self,
        checkpoint_dir=None,
        device=None,
        use_face_detection=True,
        threshold=0.5
    ):
        """
        Khởi tạo detector
        
        Args:
            checkpoint_dir: Thư mục checkpoints
            device: Device
            use_face_detection: Có dùng face detection không
            threshold: Ngưỡng phân loại (>threshold = Fake)
        """
        print("\n" + "=" * 70)
        print("🔍 KHỞI TẠO DEEPFAKE DETECTOR")
        print("=" * 70)
        
        if device is None:
            device = TrainingConfig.DEVICE
        
        self.device = device
        self.threshold = threshold
        
        # Model loader
        self.loader = ModelLoader(checkpoint_dir, device)
        
        # Load models
        self.models = {}
        self._load_models()
        
        # Image preprocessing
        self.transform = get_val_transforms()
        
        # Face preprocessing
        if use_face_detection:
            self.face_preprocessor = FacePreprocessor()
        else:
            self.face_preprocessor = None
        
        # Fusion module
        self.fusion = DeepfakeFusionModule(threshold=threshold).to(device)
        
        print("\n✅ Deepfake Detector khởi tạo thành công!")
    
    def _load_models(self):
        """Load tất cả các models có sẵn"""
        print("\n📂 Loading models...")
        
        # Thử load từng model
        try:
            self.models['vqvae'] = self.loader.load_vqvae()
            self.models['transformer'] = self.loader.load_transformer()
        except Exception as e:
            print(f"⚠️  Không thể load VQ-VAE/Transformer: {e}")
        
        try:
            self.models['ddpm'] = self.loader.load_ddpm()
        except Exception as e:
            print(f"⚠️  Không thể load DDPM: {e}")
        
        try:
            self.models['cnn_vit'] = self.loader.load_cnn_vit()
        except Exception as e:
            print(f"⚠️  Không thể load CNN/ViT: {e}")
        
        try:
            self.models['gcn'] = self.loader.load_gcn()
        except Exception as e:
            print(f"⚠️  Không thể load GCN: {e}")
        
        print(f"\n📊 Models đã load: {list(self.models.keys())}")
    
    def preprocess_image(self, image_path):
        """
        Tiền xử lý ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            tensor: Tensor ảnh đã preprocess (1, 3, H, W)
        """
        # Load ảnh
        image = Image.open(image_path).convert('RGB')
        
        # Face detection và crop (nếu enabled)
        if self.face_preprocessor:
            try:
                image = self.face_preprocessor.crop_face(image)
            except Exception as e:
                print(f"⚠️  Face detection failed, using original image: {e}")
        
        # Transform
        tensor = self.transform(image).unsqueeze(0)  # Add batch dim
        
        return tensor.to(self.device)
    
    @torch.no_grad()
    def compute_scores(self, image_tensor):
        """
        Tính tất cả anomaly scores
        
        Args:
            image_tensor: Tensor ảnh (B, 3, H, W)
            
        Returns:
            dict: Các anomaly scores
        """
        scores = {}
        features = {}
        
        # 1. Transformer perplexity score
        if 'vqvae' in self.models and 'transformer' in self.models:
            try:
                _, token_indices = self.models['vqvae'].encode(image_tensor)
                perplexity = self.models['transformer'].compute_anomaly_score(token_indices)
                scores['transformer'] = perplexity
            except Exception as e:
                print(f"⚠️  Transformer score error: {e}")
        
        # 2. DDPM anomaly score
        if 'ddpm' in self.models:
            try:
                diffusion_score = self.models['ddpm'].compute_anomaly_score(image_tensor)
                scores['diffusion'] = diffusion_score
            except Exception as e:
                print(f"⚠️  DDPM score error: {e}")
        
        # 3. CNN/ViT score
        if 'cnn_vit' in self.models:
            try:
                logits, cnn_features = self.models['cnn_vit'](image_tensor, return_features=True)
                probs = F.softmax(logits, dim=-1)
                scores['cnn_vit'] = probs[:, 1]  # Prob of Fake class
                features['cnn'] = cnn_features
            except Exception as e:
                print(f"⚠️  CNN/ViT score error: {e}")
        
        return scores, features
    
    @torch.no_grad()
    def detect(self, image_path):
        """
        Phát hiện deepfake cho một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            dict: Kết quả phân tích
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        
        # Compute scores
        scores, features = self.compute_scores(image_tensor)
        
        # Fusion
        if scores:
            output = self.fusion(scores, features)
            
            result = {
                'image': image_path,
                'prediction': 'FAKE' if output['predictions'].item() == 1 else 'REAL',
                'confidence': output['probs'].max().item(),
                'anomaly_score': output['anomaly_score'].item(),
                'individual_scores': {k: v.item() for k, v in scores.items()},
                'threshold': self.threshold
            }
        else:
            result = {
                'image': image_path,
                'prediction': 'UNKNOWN',
                'error': 'No models available to make prediction'
            }
        
        return result
    
    def detect_batch(self, image_paths, batch_size=8):
        """
        Phát hiện deepfake cho batch ảnh
        
        Args:
            image_paths: List đường dẫn ảnh
            batch_size: Kích thước batch
            
        Returns:
            list: List kết quả cho mỗi ảnh
        """
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing"):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                try:
                    result = self.detect(path)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'image': path,
                        'prediction': 'ERROR',
                        'error': str(e)
                    })
        
        return results
    
    def detect_directory(self, input_dir, output_file=None):
        """
        Phát hiện deepfake cho tất cả ảnh trong thư mục
        
        Args:
            input_dir: Thư mục chứa ảnh
            output_file: File để lưu kết quả (JSON)
            
        Returns:
            list: List kết quả
        """
        # Tìm tất cả ảnh trong thư mục
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    image_paths.append(os.path.join(root, f))
        
        print(f"\n📂 Tìm thấy {len(image_paths)} ảnh trong {input_dir}")
        
        # Detect
        results = self.detect_batch(image_paths)
        
        # Lưu kết quả
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Đã lưu kết quả vào: {output_file}")
        
        # Statistics
        fake_count = sum(1 for r in results if r.get('prediction') == 'FAKE')
        real_count = sum(1 for r in results if r.get('prediction') == 'REAL')
        error_count = sum(1 for r in results if 'error' in r)
        
        print(f"\n📊 Thống kê:")
        print(f"   - FAKE: {fake_count}")
        print(f"   - REAL: {real_count}")
        print(f"   - Errors: {error_count}")
        
        return results


# =====================================================================
# VISUALIZATION
# =====================================================================

def visualize_result(result, output_path=None):
    """
    Visualize kết quả phát hiện
    
    Args:
        result: Dict kết quả từ detect()
        output_path: Đường dẫn lưu visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib không được cài đặt. Skip visualization.")
        return
    
    # Load ảnh
    image = Image.open(result['image'])
    
    # Tạo figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Hiển thị ảnh
    axes[0].imshow(image)
    axes[0].set_title(f"Prediction: {result['prediction']}")
    axes[0].axis('off')
    
    # Hiển thị scores
    if 'individual_scores' in result:
        scores = result['individual_scores']
        names = list(scores.keys())
        values = list(scores.values())
        
        colors = ['red' if v > 0.5 else 'green' for v in values]
        
        bars = axes[1].bar(names, values, color=colors)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Individual Module Scores')
        axes[1].legend()
        
        # Add value labels
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"💾 Đã lưu visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():
    """
    Main function - Entry point
    """
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory containing model checkpoints')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 70)
    print("🔍 HỆ THỐNG PHÁT HIỆN DEEPFAKE")
    print("   Sequence Modeling + Diffusion + GCN Pipeline")
    print("=" * 70)
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = TrainingConfig.DEVICE
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    detector = DeepfakeDetector(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        threshold=args.threshold
    )
    
    # Run inference
    if args.image:
        # Single image
        print(f"\n🖼️  Processing: {args.image}")
        result = detector.detect(args.image)
        
        print(f"\n{'='*50}")
        print("📊 KẾT QUẢ:")
        print(f"{'='*50}")
        print(f"   Image: {result['image']}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result.get('confidence', 'N/A'):.4f}")
        print(f"   Anomaly Score: {result.get('anomaly_score', 'N/A'):.4f}")
        
        if 'individual_scores' in result:
            print(f"\n   Individual Scores:")
            for name, score in result['individual_scores'].items():
                print(f"      - {name}: {score:.4f}")
        
        if args.visualize:
            vis_path = os.path.join(args.output_dir, 'visualization.png')
            visualize_result(result, vis_path)
        
        # Save result
        result_path = os.path.join(args.output_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Kết quả đã lưu: {result_path}")
    
    elif args.input_dir:
        # Directory of images
        output_file = os.path.join(args.output_dir, 'results.json')
        results = detector.detect_directory(args.input_dir, output_file)
        
        if args.visualize and len(results) > 0:
            print("\n📊 Visualizing results...")
            for i, result in enumerate(results[:5]):  # Only first 5
                if 'error' not in result:
                    vis_path = os.path.join(args.output_dir, f'vis_{i}.png')
                    visualize_result(result, vis_path)
    
    else:
        print("⚠️  Cần cung cấp --image hoặc --input_dir")
        parser.print_help()
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH!")
    print("=" * 70)


if __name__ == "__main__":
    main()
