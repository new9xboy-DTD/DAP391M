import React, { useState, useCallback } from 'react';
import { predictImage } from '../utils/api';
import './ImageDropzone.css';

const ImageDropzone = ({ onPredictionComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState(null);
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
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);

    // Upload and predict
    setUploading(true);
    try {
      const result = await predictImage(file);
      onPredictionComplete(result);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed');
      onPredictionComplete(null);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="image-dropzone">
      <div
        className={`dropzone ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="image-upload"
          accept="image/*"
          onChange={handleChange}
          className="file-input"
        />
        
        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            {uploading && (
              <div className="upload-overlay">
                <div className="spinner-small"></div>
                <p>Analyzing image...</p>
              </div>
            )}
          </div>
        ) : (
          <label htmlFor="image-upload" className="upload-label">
            <div className="upload-icon">📸</div>
            <p className="upload-text">
              {uploading ? 'Processing...' : 'Drag & drop an image or click to upload'}
            </p>
            <p className="upload-hint">Supports: JPG, PNG, JPEG</p>
          </label>
        )}
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {preview && !uploading && (
        <button 
          className="reset-button"
          onClick={() => {
            setPreview(null);
            onPredictionComplete(null);
            setError(null);
          }}
        >
          Upload Another Image
        </button>
      )}
    </div>
  );
};

export default ImageDropzone;
