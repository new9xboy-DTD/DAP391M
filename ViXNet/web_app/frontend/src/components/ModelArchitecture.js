import React from 'react';
import './ModelArchitecture.css';

const ModelArchitecture = ({ modelInfo }) => {
  if (!modelInfo) {
    return (
      <div className="model-architecture">
        <div className="no-model">
          <p>No model loaded</p>
        </div>
      </div>
    );
  }

  const { architecture, loaded, epoch, metrics } = modelInfo;

  return (
    <div className="model-architecture">
      <div className="architecture-diagram">
        <div className="layer-column">
          <div className="input-layer">
            <h3>Input Image</h3>
            <p>224 × 224 × 3</p>
          </div>
        </div>

        <div className="branches">
          <div className="branch xception-branch">
            <div className="branch-header">
              <h3>🔷 Xception Branch</h3>
              <p>CNN - Global Spatial Features</p>
            </div>
            <div className="branch-body">
              <div className="layer">Pretrained Xception</div>
              <div className="arrow">↓</div>
              <div className="layer">Global Average Pooling</div>
              <div className="arrow">↓</div>
              <div className="feature-output">
                Features: {architecture?.xception_dim || 2048}D
              </div>
            </div>
          </div>

          <div className="branch vit-branch">
            <div className="branch-header">
              <h3>🔶 ViT Branch</h3>
              <p>Transformer - Patch-wise Attention</p>
            </div>
            <div className="branch-body">
              <div className="layer">Vision Transformer (Tiny)</div>
              <div className="arrow">↓</div>
              <div className="layer">Patch Embedding + Self-Attention</div>
              <div className="arrow">↓</div>
              <div className="feature-output">
                Features: {architecture?.vit_dim || 192}D
              </div>
            </div>
          </div>
        </div>

        <div className="fusion-section">
          <div className="arrow-merge">↓</div>
          <div className="fusion-layer">
            <h3>🔗 Feature Fusion</h3>
            <p>Concatenation + Dense Layers</p>
            <div className="fusion-detail">
              Output: {architecture?.fusion_dim || 512}D
            </div>
          </div>
        </div>

        <div className="classification-section">
          <div className="arrow">↓</div>
          <div className="classification-layer">
            <h3>🎯 Classification Head</h3>
            <p>Binary Classification</p>
            <div className="output">
              <span className="output-class fake">Fake</span>
              <span className="output-class real">Real</span>
            </div>
          </div>
        </div>
      </div>

      {loaded && (
        <div className="model-info-panel">
          <h3>📊 Model Information</h3>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Status:</span>
              <span className="info-value status-loaded">✓ Loaded</span>
            </div>
            <div className="info-item">
              <span className="info-label">Epoch:</span>
              <span className="info-value">{epoch}</span>
            </div>
            {metrics?.accuracy && (
              <div className="info-item">
                <span className="info-label">Validation Accuracy:</span>
                <span className="info-value">{(metrics.accuracy * 100).toFixed(2)}%</span>
              </div>
            )}
            {metrics?.precision && (
              <div className="info-item">
                <span className="info-label">Precision:</span>
                <span className="info-value">{(metrics.precision * 100).toFixed(2)}%</span>
              </div>
            )}
            {metrics?.recall && (
              <div className="info-item">
                <span className="info-label">Recall:</span>
                <span className="info-value">{(metrics.recall * 100).toFixed(2)}%</span>
              </div>
            )}
            {metrics?.f1 && (
              <div className="info-item">
                <span className="info-label">F1 Score:</span>
                <span className="info-value">{(metrics.f1 * 100).toFixed(2)}%</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelArchitecture;
