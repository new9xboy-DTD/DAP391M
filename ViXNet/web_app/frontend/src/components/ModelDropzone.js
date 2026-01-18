import React, { useState, useCallback } from 'react';
import { analyzeModel } from '../utils/api';
import './ModelDropzone.css';

const ModelDropzone = ({ onAnalysisComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [fileName, setFileName] = useState(null);
  const [error, setError] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(null);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      await handleFile(files[0]);
    }
  }, []);

  const handleChange = async (e) => {
    e.preventDefault();
    setError(null);
    
    if (e.target.files && e.target.files[0]) {
      await handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    // Validate file type
    if (!file.name.endsWith('.pth') && !file.name.endsWith('.pt')) {
      setError('Please upload a PyTorch model file (.pth or .pt)');
      return;
    }

    setFileName(file.name);
    setAnalyzing(true);

    try {
      const result = await analyzeModel(file);
      
      if (result.warning) {
        setError(result.warning + ': ' + result.error);
      }
      
      onAnalysisComplete(result);
    } catch (err) {
      setError(err.response?.data?.error || 'Model analysis failed');
      onAnalysisComplete(null);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="model-dropzone">
      <div
        className={`dropzone ${dragActive ? 'drag-active' : ''} ${analyzing ? 'analyzing' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="model-upload"
          accept=".pth,.pt"
          onChange={handleChange}
          className="file-input"
        />
        
        <label htmlFor="model-upload" className="upload-label">
          {analyzing ? (
            <div className="analyzing-container">
              <div className="spinner-small"></div>
              <p className="analyzing-text">Analyzing model and calculating AUC...</p>
              <p className="analyzing-hint">This may take a few moments</p>
            </div>
          ) : fileName ? (
            <div className="file-info">
              <div className="file-icon">✅</div>
              <p className="file-name">{fileName}</p>
              <p className="file-hint">Model loaded successfully</p>
            </div>
          ) : (
            <>
              <div className="upload-icon">🔧</div>
              <p className="upload-text">
                Drag & drop a model file or click to upload
              </p>
              <p className="upload-hint">Supports: .pth, .pt (PyTorch models)</p>
            </>
          )}
        </label>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {fileName && !analyzing && (
        <button 
          className="reset-button"
          onClick={() => {
            setFileName(null);
            onAnalysisComplete(null);
            setError(null);
          }}
        >
          Upload Another Model
        </button>
      )}
    </div>
  );
};

export default ModelDropzone;
