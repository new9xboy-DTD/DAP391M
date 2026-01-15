"""
=====================================================================
MÔ-ĐUN TIỀN XỬ LÝ DỮ LIỆU
=====================================================================
Mô tả:
    File này chứa các hàm và class để:
    1. Load và transform ảnh
    2. Tạo DataLoader cho training/validation/test
    3. Căn chỉnh và crop khuôn mặt
    4. Detect và trích xuất facial landmarks
    5. Phân tách nửa khuôn mặt (nếu cần)
    
Tác giả: DAP391M Team
Phiên bản: 1.0
=====================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

# Import config
from config import DataConfig, GCNConfig, TrainingConfig


# =====================================================================
# DATA TRANSFORMS
# =====================================================================

def get_train_transforms():
    """
    Định nghĩa các phép biến đổi cho training set
    
    Bao gồm data augmentation để tăng tính đa dạng của dữ liệu:
    - Random horizontal flip: Lật ảnh theo chiều ngang
    - Random rotation: Xoay ảnh ngẫu nhiên
    - Color jitter: Thay đổi độ sáng, tương phản, bão hòa, sắc độ
    - Normalization: Chuẩn hóa theo ImageNet statistics
    
    Returns:
        transforms.Compose: Composed transforms cho training
    """
    return transforms.Compose([
        # Resize về kích thước chuẩn
        transforms.Resize((DataConfig.IMG_SIZE, DataConfig.IMG_SIZE)),
        
        # Data augmentation - Tăng cường dữ liệu
        transforms.RandomHorizontalFlip(p=0.5),  # 50% lật ngang
        transforms.RandomRotation(degrees=10),   # Xoay ±10 độ
        transforms.ColorJitter(
            brightness=0.2,  # Thay đổi độ sáng ±20%
            contrast=0.2,    # Thay đổi độ tương phản ±20%
            saturation=0.2,  # Thay đổi độ bão hòa ±20%
            hue=0.1          # Thay đổi sắc độ ±10%
        ),
        
        # Chuyển sang tensor và chuẩn hóa
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Mean của ImageNet
            std=[0.229, 0.224, 0.225]    # Std của ImageNet
        )
    ])


def get_val_transforms():
    """
    Định nghĩa các phép biến đổi cho validation/test set
    
    KHÔNG có data augmentation, chỉ có preprocessing cơ bản:
    - Resize về kích thước chuẩn
    - Normalization
    
    Returns:
        transforms.Compose: Composed transforms cho validation/test
    """
    return transforms.Compose([
        transforms.Resize((DataConfig.IMG_SIZE, DataConfig.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_vqvae_transforms():
    """
    Transforms đặc biệt cho VQ-VAE training
    
    VQ-VAE cần pixel values trong khoảng [-1, 1] thay vì ImageNet normalization
    để reconstruction loss có ý nghĩa hơn.
    
    Returns:
        transforms.Compose: Transforms cho VQ-VAE
    """
    return transforms.Compose([
        transforms.Resize((DataConfig.IMG_SIZE, DataConfig.IMG_SIZE)),
        transforms.ToTensor(),
        # Scale từ [0, 1] sang [-1, 1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


# =====================================================================
# CUSTOM DATASETS
# =====================================================================

class DeepfakeDataset(Dataset):
    """
    Custom Dataset cho nhận diện deepfake
    
    Hỗ trợ:
    - Load ảnh từ cấu trúc thư mục Fake/Real
    - Trích xuất facial landmarks (tùy chọn)
    - Custom transforms
    
    Args:
        root_dir (str): Đường dẫn đến thư mục chứa Fake/ và Real/
        transform: Transforms áp dụng cho ảnh
        extract_landmarks (bool): Có trích xuất landmarks không
        real_only (bool): Chỉ load ảnh thật (cho unsupervised training)
    """
    
    def __init__(self, root_dir, transform=None, extract_landmarks=False, real_only=False):
        """
        Khởi tạo dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.extract_landmarks = extract_landmarks
        self.real_only = real_only
        
        # List để lưu đường dẫn ảnh và nhãn
        self.image_paths = []
        self.labels = []
        
        # Class mapping: Fake=0, Real=1 (theo thứ tự alphabet)
        self.class_to_idx = {'Fake': 0, 'Real': 1}
        self.idx_to_class = {0: 'Fake', 1: 'Real'}
        
        # Load danh sách ảnh
        self._load_images()
        
        print(f"📊 Loaded {len(self)} images from {root_dir}")
        if not real_only:
            fake_count = sum(1 for label in self.labels if label == 0)
            real_count = sum(1 for label in self.labels if label == 1)
            print(f"   - Fake: {fake_count}, Real: {real_count}")
    
    def _load_images(self):
        """
        Load danh sách đường dẫn ảnh từ thư mục
        """
        if self.real_only:
            # Chỉ load ảnh thật cho unsupervised training
            classes_to_load = ['Real']
        else:
            # Load cả Fake và Real
            classes_to_load = ['Fake', 'Real']
        
        for class_name in classes_to_load:
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"⚠️  Không tìm thấy thư mục: {class_dir}")
                continue
            
            # Các định dạng ảnh được hỗ trợ
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            
            for filename in os.listdir(class_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        """Trả về số lượng ảnh trong dataset"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Lấy một sample từ dataset
        
        Args:
            idx: Index của sample
            
        Returns:
            Tuple chứa:
            - image: Tensor ảnh đã transform
            - label: Nhãn (0=Fake, 1=Real)
            - landmarks: Facial landmarks (nếu extract_landmarks=True)
        """
        # Load ảnh
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Áp dụng transform
        if self.transform:
            image = self.transform(image)
        
        # Trích xuất landmarks nếu cần
        if self.extract_landmarks:
            landmarks = self._extract_landmarks(img_path)
            return image, label, landmarks
        
        return image, label
    
    def _extract_landmarks(self, img_path):
        """
        Trích xuất facial landmarks từ ảnh
        
        Hiện tại trả về placeholder, sẽ được implement sau
        khi tích hợp thư viện detect landmarks (dlib hoặc mediapipe)
        
        Args:
            img_path: Đường dẫn đến ảnh
            
        Returns:
            numpy.ndarray: Mảng tọa độ landmarks với shape (num_landmarks, 2)
        """
        # TODO: Implement landmark detection sử dụng dlib hoặc mediapipe
        # Hiện tại trả về zeros làm placeholder
        return np.zeros((GCNConfig.NUM_LANDMARKS, 2), dtype=np.float32)


class RealOnlyDataset(DeepfakeDataset):
    """
    Dataset chỉ chứa ảnh thật
    
    Dùng cho:
    - Huấn luyện VQ-VAE + Transformer (unsupervised)
    - Huấn luyện DDPM (diffusion model)
    
    Các model này chỉ cần học phân phối của ảnh thật,
    sau đó phát hiện deepfake dựa trên việc ảnh không phù hợp phân phối.
    """
    
    def __init__(self, root_dir, transform=None, extract_landmarks=False):
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            extract_landmarks=extract_landmarks,
            real_only=True  # Chỉ load ảnh thật
        )


# =====================================================================
# DATA LOADERS
# =====================================================================

def create_dataloaders(mode='supervised', extract_landmarks=False):
    """
    Tạo DataLoaders cho training, validation và test
    
    Args:
        mode: Chế độ training
            - 'supervised': Load cả Fake và Real với labels
            - 'unsupervised': Chỉ load ảnh Real (cho VQ-VAE, Diffusion)
            - 'vqvae': Dùng transforms đặc biệt cho VQ-VAE
        extract_landmarks: Có trích xuất landmarks không (cho GCN)
        
    Returns:
        dict: Dictionary chứa train_loader, val_loader, test_loader
    """
    print(f"\n📂 Đang tạo DataLoaders (mode: {mode})...")
    
    # Chọn transforms phù hợp
    if mode == 'vqvae':
        train_transform = get_vqvae_transforms()
        val_transform = get_vqvae_transforms()
    else:
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
    
    # Chọn Dataset class phù hợp
    if mode in ['unsupervised', 'vqvae']:
        # Chỉ dùng ảnh thật
        DatasetClass = RealOnlyDataset
        print("   → Chế độ unsupervised: Chỉ load ảnh thật")
    else:
        DatasetClass = DeepfakeDataset
        print("   → Chế độ supervised: Load cả Fake và Real")
    
    # Kiểm tra xem dataset có tồn tại không
    if not os.path.exists(DataConfig.TRAIN_DIR):
        print(f"\n⚠️  CẢNH BÁO: Không tìm thấy thư mục dataset tại: {DataConfig.TRAIN_DIR}")
        print("   Dataset sẽ được cấu hình sau. Hiện tại trả về None.")
        return None
    
    # Tạo datasets
    train_dataset = DatasetClass(
        root_dir=DataConfig.TRAIN_DIR,
        transform=train_transform,
        extract_landmarks=extract_landmarks
    )
    
    val_dataset = DatasetClass(
        root_dir=DataConfig.VAL_DIR,
        transform=val_transform,
        extract_landmarks=extract_landmarks
    )
    
    test_dataset = DatasetClass(
        root_dir=DataConfig.TEST_DIR,
        transform=val_transform,
        extract_landmarks=extract_landmarks
    )
    
    # Tạo DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=True,  # Shuffle trong training
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=True,  # Tăng tốc transfer data lên GPU
        drop_last=True  # Bỏ batch cuối nếu không đủ size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=DataConfig.NUM_WORKERS,
        pin_memory=True
    )
    
    # In thống kê
    print(f"\n📊 Thống kê DataLoaders:")
    print(f"   - Train: {len(train_dataset)} ảnh, {len(train_loader)} batches")
    print(f"   - Val: {len(val_dataset)} ảnh, {len(val_loader)} batches")
    print(f"   - Test: {len(test_dataset)} ảnh, {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


# =====================================================================
# FACE PREPROCESSING (Căn chỉnh khuôn mặt)
# =====================================================================

class FacePreprocessor:
    """
    Class để tiền xử lý khuôn mặt
    
    Bao gồm:
    - Face detection (phát hiện khuôn mặt)
    - Face alignment (căn chỉnh khuôn mặt)
    - Face cropping (cắt khuôn mặt)
    - Landmark extraction (trích xuất điểm mốc)
    
    Lưu ý: Cần cài đặt thêm thư viện:
    - dlib: pip install dlib
    - face_recognition: pip install face-recognition
    - mediapipe: pip install mediapipe
    """
    
    def __init__(self, detector='dlib'):
        """
        Khởi tạo preprocessor
        
        Args:
            detector: Loại detector ('dlib' hoặc 'mediapipe')
        """
        self.detector = detector
        self._initialized = False
        
        # Lazy initialization - chỉ khởi tạo khi cần
        self._face_detector = None
        self._landmark_predictor = None
    
    def _init_detector(self):
        """
        Khởi tạo face detector
        
        Chỉ gọi khi thực sự cần detect face
        """
        if self._initialized:
            return
        
        try:
            if self.detector == 'dlib':
                import dlib
                # Detector khuôn mặt
                self._face_detector = dlib.get_frontal_face_detector()
                # Landmark predictor (cần download model file)
                # Model file: shape_predictor_68_face_landmarks.dat
                # Download từ: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
                model_path = "shape_predictor_68_face_landmarks.dat"
                if os.path.exists(model_path):
                    self._landmark_predictor = dlib.shape_predictor(model_path)
                else:
                    print(f"⚠️  Không tìm thấy landmark model: {model_path}")
                    print("   Download từ: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                    
            elif self.detector == 'mediapipe':
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                )
            
            self._initialized = True
            print(f"✅ Khởi tạo {self.detector} detector thành công!")
            
        except ImportError as e:
            print(f"❌ Không thể import {self.detector}: {e}")
            print(f"   Cài đặt: pip install {self.detector}")
    
    def detect_face(self, image):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: numpy array hoặc PIL Image
            
        Returns:
            list: Danh sách bounding boxes [(x, y, w, h), ...]
        """
        self._init_detector()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.detector == 'dlib' and self._face_detector:
            faces = self._face_detector(image, 1)
            return [(face.left(), face.top(), face.width(), face.height()) 
                    for face in faces]
        
        return []
    
    def extract_landmarks(self, image):
        """
        Trích xuất facial landmarks
        
        Args:
            image: numpy array hoặc PIL Image
            
        Returns:
            numpy.ndarray: Mảng landmarks với shape (68, 2) cho dlib
                          hoặc (468, 3) cho mediapipe
        """
        self._init_detector()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.detector == 'dlib':
            if self._face_detector and self._landmark_predictor:
                faces = self._face_detector(image, 1)
                if len(faces) > 0:
                    shape = self._landmark_predictor(image, faces[0])
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                    return landmarks
                    
        elif self.detector == 'mediapipe':
            if hasattr(self, '_face_mesh'):
                results = self._face_mesh.process(image)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    h, w = image.shape[:2]
                    return np.array([
                        [lm.x * w, lm.y * h, lm.z * w]
                        for lm in landmarks.landmark
                    ])
        
        # Trả về zeros nếu không detect được
        if self.detector == 'mediapipe':
            return np.zeros((468, 3), dtype=np.float32)
        return np.zeros((68, 2), dtype=np.float32)
    
    def crop_face(self, image, margin=0.2):
        """
        Cắt khuôn mặt từ ảnh với margin bổ sung
        
        Args:
            image: numpy array hoặc PIL Image
            margin: Tỷ lệ margin thêm vào xung quanh face (default: 20%)
            
        Returns:
            PIL.Image: Ảnh khuôn mặt đã crop
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Detect face
        faces = self.detect_face(image)
        
        if len(faces) == 0:
            # Nếu không detect được, trả về ảnh gốc đã resize
            return image.resize((DataConfig.IMG_SIZE, DataConfig.IMG_SIZE))
        
        # Lấy face đầu tiên
        x, y, w, h = faces[0]
        
        # Thêm margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        # Tính bounding box mới với margin
        left = max(0, x - margin_w)
        top = max(0, y - margin_h)
        right = min(image.width, x + w + margin_w)
        bottom = min(image.height, y + h + margin_h)
        
        # Crop và resize
        face_img = image.crop((left, top, right, bottom))
        face_img = face_img.resize((DataConfig.IMG_SIZE, DataConfig.IMG_SIZE))
        
        return face_img


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Đảo ngược normalization để hiển thị ảnh
    
    Args:
        tensor: Tensor đã normalize
        mean: Giá trị mean đã dùng để normalize
        std: Giá trị std đã dùng để normalize
        
    Returns:
        numpy.ndarray: Ảnh có pixel values trong [0, 1]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Chuyển từ (C, H, W) sang (H, W, C)
    img = tensor.permute(1, 2, 0).numpy()
    
    return img


def inverse_vqvae_normalize(tensor):
    """
    Đảo ngược VQ-VAE normalization (từ [-1, 1] về [0, 1])
    
    Args:
        tensor: Tensor đã normalize với mean=0.5, std=0.5
        
    Returns:
        numpy.ndarray: Ảnh có pixel values trong [0, 1]
    """
    return inverse_normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def compute_dataset_statistics(data_loader, num_batches=100):
    """
    Tính mean và std của dataset
    
    Có thể dùng để tính custom normalization thay vì dùng ImageNet values
    
    Args:
        data_loader: DataLoader chứa dữ liệu
        num_batches: Số batches để tính (dùng sampling nếu dataset lớn)
        
    Returns:
        tuple: (mean, std) với shape (3,)
    """
    print("📊 Đang tính mean và std của dataset...")
    
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_pixels = 0
    
    for i, batch in enumerate(tqdm(data_loader, desc="Computing stats")):
        if i >= num_batches:
            break
        
        # Lấy images (có thể là tuple (images, labels) hoặc chỉ images)
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        
        # Tính trên CPU để tiết kiệm VRAM
        images = images.float()
        
        # Tính sum và squared sum
        channels_sum += images.sum(dim=[0, 2, 3])
        channels_squared_sum += (images ** 2).sum(dim=[0, 2, 3])
        num_pixels += images.shape[0] * images.shape[2] * images.shape[3]
    
    # Tính mean và std
    mean = channels_sum / num_pixels
    std = torch.sqrt(channels_squared_sum / num_pixels - mean ** 2)
    
    print(f"   Mean: {mean.tolist()}")
    print(f"   Std: {std.tolist()}")
    
    return mean.tolist(), std.tolist()


# =====================================================================
# MAIN (Test)
# =====================================================================

if __name__ == "__main__":
    """
    Test data loading khi chạy file này trực tiếp
    """
    print("=" * 70)
    print("🧪 TEST DATA PREPROCESSING MODULE")
    print("=" * 70)
    
    # Test tạo dataloaders
    print("\n1. Test tạo DataLoaders (supervised mode):")
    dataloaders = create_dataloaders(mode='supervised')
    
    if dataloaders:
        # Test lấy một batch
        print("\n2. Test lấy một batch từ train_loader:")
        train_loader = dataloaders['train']
        images, labels = next(iter(train_loader))
        print(f"   - Images shape: {images.shape}")
        print(f"   - Labels: {labels}")
        print(f"   - Min pixel value: {images.min():.4f}")
        print(f"   - Max pixel value: {images.max():.4f}")
    
    # Test FacePreprocessor
    print("\n3. Test FacePreprocessor:")
    preprocessor = FacePreprocessor(detector='dlib')
    print("   ✓ FacePreprocessor đã khởi tạo")
    
    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TEST!")
    print("=" * 70)
