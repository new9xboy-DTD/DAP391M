# ViXNet Web Application - Visual Guide

## Application Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    ViXNet Model Visualization                │
│              Vision Transformer + Xception Network           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   📊 Model Architecture                      │
│                                                              │
│  Input Image (224x224x3)                                    │
│           │                                                   │
│     ┌─────┴─────┐                                           │
│     │           │                                            │
│  Xception    Vision Transformer                             │
│  (2048D)        (192D)                                       │
│     │           │                                            │
│     └─────┬─────┘                                           │
│           │                                                   │
│    Feature Fusion (512D)                                    │
│           │                                                   │
│    Classification Head                                       │
│           │                                                   │
│     Fake / Real                                             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────┬──────────────────────────────────────┐
│  🖼️ Image Inference  │      🔧 Model Analysis               │
│                      │                                       │
│  ┌────────────────┐  │  ┌─────────────────────────────────┐│
│  │                │  │  │                                  ││
│  │  Drag & Drop   │  │  │     Drag & Drop Model File      ││
│  │     Image      │  │  │         (.pth)                  ││
│  │                │  │  │                                  ││
│  └────────────────┘  │  └─────────────────────────────────┘│
│                      │                                       │
│  Results:            │  Results:                            │
│  • Real/Fake         │  • AUC Score                         │
│  • Confidence        │  • ROC Curve                         │
│  • Probabilities     │  • Confusion Matrix                  │
└──────────────────────┴──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       📈 Results Display                     │
│                                                              │
│  Image Prediction:              Model Analysis:              │
│  ┌──────────────┐              ┌──────────────────────┐     │
│  │  ✓ Real      │              │  AUC: 0.9945         │     │
│  │  98.76%      │              │  Accuracy: 98.23%    │     │
│  └──────────────┘              │  Samples: 1000       │     │
│                                 └──────────────────────┘     │
│  Probabilities:                                              │
│  Fake: ██ 1.24%                 ROC Curve:                   │
│  Real: ████████████ 98.76%      [Interactive Chart]         │
└─────────────────────────────────────────────────────────────┘
```

## User Flow

### 1. Initial Load
```
User opens http://localhost:3000
         ↓
Frontend React app loads
         ↓
Checks backend health (GET /api/health)
         ↓
Loads model info (GET /api/model-info)
         ↓
Displays architecture visualization
```

### 2. Image Inference Flow
```
User drags image to dropzone
         ↓
Image uploaded to backend (POST /api/predict)
         ↓
Backend processes:
  • Resize to 224x224
  • Normalize pixel values
  • Run through ViXNet model
  • Calculate softmax probabilities
         ↓
Results returned to frontend
         ↓
Display prediction with confidence
```

### 3. Model Analysis Flow
```
User drags .pth file to dropzone
         ↓
Model uploaded to backend (POST /api/analyze-model)
         ↓
Backend processes:
  • Load model checkpoint
  • Validate architecture
  • Run inference on test dataset
  • Calculate AUC score
  • Generate ROC curve
  • Create confusion matrix
         ↓
Results returned to frontend
         ↓
Display comprehensive metrics and charts
```

## API Architecture

```
┌─────────────┐         HTTP          ┌──────────────┐
│   React     │ ◄─────────────────► │    Flask     │
│  Frontend   │      JSON/Files      │   Backend    │
│             │                       │              │
│  - UI       │                       │  - Routes    │
│  - State    │                       │  - Logic     │
│  - Charts   │                       │  - Model     │
└─────────────┘                       └──────┬───────┘
                                             │
                                             │
                                      ┌──────▼───────┐
                                      │   ViXNet     │
                                      │    Model     │
                                      │              │
                                      │  - Xception  │
                                      │  - ViT       │
                                      │  - Fusion    │
                                      └──────────────┘
```

## Component Hierarchy

```
App (Main Container)
├── Header
│   └── Title & Subtitle
│
├── ModelArchitecture
│   ├── Architecture Diagram
│   │   ├── Input Layer
│   │   ├── Branches (Xception + ViT)
│   │   ├── Fusion Layer
│   │   └── Classification Head
│   └── Model Info Panel
│       └── Metrics Display
│
├── Dropzone Container
│   ├── ImageDropzone
│   │   ├── Drag & Drop Area
│   │   ├── File Input
│   │   ├── Preview
│   │   └── Upload Status
│   │
│   └── ModelDropzone
│       ├── Drag & Drop Area
│       ├── File Input
│       ├── Analysis Status
│       └── Upload Status
│
├── ResultsDisplay
│   ├── Prediction Results
│   │   ├── Prediction Badge
│   │   ├── Confidence Meter
│   │   └── Probability Bars
│   │
│   └── Analysis Results
│       ├── Metrics Grid
│       │   ├── AUC Card
│       │   ├── Accuracy Card
│       │   └── Samples Card
│       ├── Confusion Matrix
│       ├── ROC Curve (Recharts)
│       └── Model Details
│
└── Footer
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                         User Actions                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │              │
    ▼             ▼              ▼
┌─────────┐  ┌─────────┐  ┌──────────┐
│  Upload │  │  Upload │  │  View    │
│  Image  │  │  Model  │  │  Info    │
└────┬────┘  └────┬────┘  └────┬─────┘
     │            │             │
     ▼            ▼             ▼
┌─────────────────────────────────────┐
│          API Layer (axios)          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Flask Backend Endpoints        │
│  • /api/predict                     │
│  • /api/analyze-model               │
│  • /api/model-info                  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         ViXNet Model                │
│  • Image Preprocessing              │
│  • Model Inference                  │
│  • Metrics Calculation              │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         Results (JSON)              │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      React State Update             │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│       UI Re-render                  │
│  • Display Results                  │
│  • Update Charts                    │
│  • Show Metrics                     │
└─────────────────────────────────────┘
```

## Color Scheme

```
Primary Colors:
  • Purple Gradient: #667eea → #764ba2
  • Used for: Headers, buttons, highlights

Prediction Colors:
  • Real (Green): #27ae60 → #2ecc71
  • Fake (Red): #e74c3c → #c0392b

UI Elements:
  • Background: White (#fff) with transparency
  • Secondary BG: #f8f9fa
  • Borders: #e0e0e0
  • Text: #333 (dark), #666 (medium), #999 (light)

Status Colors:
  • Success: #27ae60
  • Warning: #f39c12
  • Error: #e74c3c
  • Info: #3498db
```

## Responsive Breakpoints

```
Desktop (> 1100px):
  • Two-column layout for dropzones
  • Full-width charts
  • Side-by-side metrics

Tablet (768px - 1100px):
  • Single-column layout for dropzones
  • Stacked metrics
  • Full-width charts

Mobile (< 768px):
  • Single-column layout
  • Stacked elements
  • Smaller fonts
  • Touch-optimized buttons
```

## State Management

```
App.js maintains global state:
  • modelInfo: Current model information
  • predictionResult: Latest image prediction
  • analysisResult: Latest model analysis
  • isHealthy: Backend health status
  • loading: Loading state

Components receive state via props:
  • Parent → Child communication
  • Callback functions for updates
  • No external state management (Redux, etc.)
```

## Performance Considerations

```
Backend:
  • Model loaded once on startup
  • Kept in memory for fast inference
  • Test dataset loaded on-demand for AUC
  • Uses PyTorch's eval() mode (no gradients)

Frontend:
  • React hooks for efficient rendering
  • Components only re-render on state change
  • Charts use Recharts (optimized library)
  • Images converted to base64 for preview

Network:
  • Images sent as FormData (multipart)
  • JSON responses compressed
  • Proxy setup to avoid CORS in dev
  • Results streamed for large responses
```

This visual guide provides a comprehensive overview of the application's structure, flow, and design decisions.
