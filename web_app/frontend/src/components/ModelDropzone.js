import React, { useState, useCallback, useEffect } from 'react';
import { analyzeModel, getDatasets } from '../utils/api';
import './ModelDropzone.css';

const ModelDropzone = ({ onAnalysisComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [fileName, setFileName] = useState(null);
  const [error, setError] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('default');
  const [pendingFile, setPendingFile] = useState(null);

  useEffect(() => {
    // Load available datasets on mount
    const loadDatasets = async () => {
      try {
        const response = await getDatasets();
        setDatasets(response.datasets || []);
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    loadDatasets();
  }, []);

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
      setPendingFile(files[0]);
    }
  }, []);

  const handleChange = async (e) => {
    e.preventDefault();
    setError(null);
    
    if (e.target.files && e.target.files[0]) {
      setPendingFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!pendingFile) return;

    // Validate file type
    if (!pendingFile.name.endsWith('.pth') && !pendingFile.name.endsWith('.pt')) {
      setError('Please upload a PyTorch model file (.pth or .pt)');
      return;
    }

    setFileName(pendingFile.name);
    setAnalyzing(true);

    try {
      const result = await analyzeModel(pendingFile, selectedDataset);
      console.log("server response:", result);
      console.log("result.success:", result.success);
      console.log("result.model_info:", result.model_info);
      
      // Check if response has model_info (successful analysis)
      if (result.model_info) {
        console.log('Model analysis successful:', result);
        onAnalysisComplete(result);
        setPendingFile(null); // Clear pending file after successful analysis
      } else if (result.warning) {
        console.warn('Warning from API:', result.warning, result.error);
        setError(result.warning + ': ' + result.error);
        // Still display the result even if there's a warning
        onAnalysisComplete(result);
        setPendingFile(null);
      } else {
        console.error('Unexpected response structure:', result);
        setError('Unexpected response from server');
      }
    } catch (err) {
      console.error('Error analyzing model:', err);
      setError(err.response?.data?.error || 'Model analysis failed');
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
          ) : pendingFile ? (
            <div className="file-info">
              <div className="file-icon">📦</div>
              <p className="file-name">{pendingFile.name}</p>
              <p className="file-hint">Ready to analyze - select dataset below</p>
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

      {pendingFile && !analyzing && !fileName && datasets.length > 0 && (
        <div className="dataset-selector">
          <label htmlFor="dataset-select">
            <strong>Select Dataset for Evaluation:</strong>
          </label>
          <select 
            id="dataset-select"
            value={selectedDataset} 
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="dataset-dropdown"
          >
            {datasets.map((ds) => (
              <option key={ds.key} value={ds.key}>
                {ds.name}
              </option>
            ))}
          </select>
          <button 
            className="analyze-button"
            onClick={handleAnalyze}
          >
            Analyze Model
          </button>
        </div>
      )}

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
            setPendingFile(null);
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
