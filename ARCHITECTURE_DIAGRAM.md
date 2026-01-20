# Multi-Model System Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER WORKFLOW                            │
└─────────────────────────────────────────────────────────────────┘

1. User Opens Web App
   │
   ├─► No Model Loaded Initially (Welcome Screen)
   │
2. User Uploads Model File (.pth)
   │
   ├─► File Selected → Dropdown Appears
   │
3. User Selects Dataset
   │
   ├─► Choose from: default, dataset2, dataset3, etc.
   │
4. User Clicks "Analyze Model"
   │
   ├─► Backend Processing:
   │   ├─► Load checkpoint file
   │   ├─► Detect model type (ViXNet/Xception/ViT)
   │   ├─► Create appropriate model
   │   ├─► Load selected dataset
   │   ├─► Calculate AUC & Accuracy
   │   └─► Return results
   │
5. Model Loaded → Image Inference Available
   │
   ├─► User can now upload images
   │   ├─► Drag & drop image
   │   ├─► Get Real/Fake prediction
   │   └─► View confidence scores
   │
6. User Can Upload Another Model (Optional)
   └─► Go back to step 2


┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   Flask Server   │
│   (Port 5000)    │
└────────┬─────────┘
         │
         ├─► GET /api/health
         │   └─► Check if backend is running
         │
         ├─► GET /api/datasets
         │   └─► List available datasets from Config
         │
         ├─► GET /api/model-info
         │   └─► Return current model information
         │
         ├─► POST /api/analyze-model
         │   ├─► Receive: model file + dataset key
         │   ├─► Process via model_factory:
         │   │   ├─► detect_model_type()
         │   │   └─► load_model_from_checkpoint()
         │   ├─► Create test loader for dataset
         │   ├─► Calculate AUC
         │   └─► Return: model info + metrics
         │
         ├─► POST /api/calculate-auc
         │   ├─► Receive: dataset key
         │   ├─► Use current loaded model
         │   └─► Return: AUC results
         │
         └─► POST /api/predict
             ├─► Receive: image file
             ├─► Preprocess image
             ├─► Run inference
             └─► Return: prediction + confidence

┌─────────────────────────────────────────────────────────────────┐
│                    MODEL FACTORY SYSTEM                          │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  Model Checkpoint │
                    │   (uploaded.pth)  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ detect_model_type│
                    │   (Automatic)     │
                    └─────────┬─────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  ViXNet  │  │Xception  │  │   ViT    │
        │   Model  │  │  Only    │  │  Only    │
        └──────────┘  └──────────┘  └──────────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Loaded Model    │
                    │  Ready for Use   │
                    └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     DATASET CONFIGURATION                        │
└─────────────────────────────────────────────────────────────────┘

Config.DATASETS = {
    'default': {
        'name': 'Default Dataset',
        'test': '/path/to/Test/'
            ├── Fake/
            │   ├── fake1.jpg
            │   ├── fake2.jpg
            │   └── ...
            └── Real/
                ├── real1.jpg
                ├── real2.jpg
                └── ...
    },
    'dataset2': {...},
    'dataset3': {...}
}

┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND COMPONENTS                           │
└─────────────────────────────────────────────────────────────────┘

App.js (Main Container)
├── Header
│   └── Title + Subtitle
│
├── ModelArchitecture (if model loaded)
│   └── Display model info + metrics
│
├── Model Upload Section
│   └── ModelDropzone Component
│       ├── File input (drag & drop)
│       ├── Dataset selector (dropdown)
│       └── Analyze button
│
├── Image Inference Section (if model loaded)
│   └── ImageDropzone Component
│       └── Image upload + prediction
│
├── Results Section (if results available)
│   └── ResultsDisplay Component
│       ├── Prediction results
│       └── Model analysis results
│
└── Footer

┌─────────────────────────────────────────────────────────────────┐
│                    DATA FLOW SEQUENCE                            │
└─────────────────────────────────────────────────────────────────┘

Frontend                  Backend                   Model Factory
   │                         │                            │
   │─── Upload Model ──────►│                            │
   │    + Dataset Key        │                            │
   │                         │                            │
   │                         │──── Load Checkpoint ──────►│
   │                         │                            │
   │                         │                            │─── Detect Type
   │                         │                            │
   │                         │                            │─── Create Model
   │                         │                            │
   │                         │◄─── Return Model ──────────│
   │                         │                            │
   │                         │                            │
   │                         │─── Load Dataset            │
   │                         │                            │
   │                         │─── Calculate AUC           │
   │                         │                            │
   │◄── Return Results ──────│                            │
   │    (AUC, Accuracy)      │                            │
   │                         │                            │
   │                         │                            │
   │─── Upload Image ───────►│                            │
   │                         │                            │
   │                         │─── Inference ──────────────►│
   │                         │                            │
   │◄── Prediction ──────────│◄─── Result ────────────────│
   │    (Real/Fake)          │                            │

┌─────────────────────────────────────────────────────────────────┐
│                    KEY IMPROVEMENTS                              │
└─────────────────────────────────────────────────────────────────┘

Before:                          After:
├── Always loads ViXNet      →   ├── No default loading
├── Single model type        →   ├── Multiple model types
├── Fixed dataset            →   ├── Selectable datasets
├── Immediate model required →   ├── Optional model loading
└── Manual configuration     →   └── Automatic detection

Benefits:
✓ Memory efficient (no unused models)
✓ Flexible model testing
✓ Easy dataset comparison
✓ User-friendly workflow
✓ Scalable architecture
