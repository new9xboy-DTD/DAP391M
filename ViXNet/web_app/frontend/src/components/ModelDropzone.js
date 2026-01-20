import React, { useState, useCallback, useEffect, useRef } from 'react';
import { analyzeModel, getDatasets } from '../utils/api';

const ModelDropzone = ({ onAnalysisComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [fileName, setFileName] = useState(null);
  const [error, setError] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('default');
  const [pendingFile, setPendingFile] = useState(null);
  const fileInputRef = useRef(null);

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
    <div className="w-full">
      <div
        className={`
          border-3 border-dashed rounded-2xl p-10 text-center cursor-pointer 
          transition-all duration-300 min-h-[300px] flex items-center justify-center
          ${dragActive 
            ? 'border-secondary-500 bg-secondary-100 scale-[1.02]' 
            : analyzing 
              ? 'border-primary-500 bg-primary-50' 
              : 'border-gray-300 bg-gray-50 hover:border-secondary-500 hover:bg-secondary-50'
          }
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="model-upload"
          ref={fileInputRef}
          accept=".pth,.pt"
          onChange={handleChange}
          className="hidden"
        />
        
        <label htmlFor="model-upload" className="flex flex-col items-center cursor-pointer w-full">
          {analyzing ? (
            <div className="flex flex-col items-center">
              <div className="w-10 h-10 border-4 border-primary-200 border-t-primary-500 rounded-full animate-spin mb-4"></div>
              <p className="text-xl text-gray-800 font-semibold mb-2">Analyzing model and calculating AUC...</p>
              <p className="text-sm text-gray-500">This may take a few moments</p>
            </div>
          ) : fileName ? (
            <div className="flex flex-col items-center">
              <div className="text-6xl mb-5">✅</div>
              <p className="text-xl text-gray-800 font-semibold mb-2 break-all">{fileName}</p>
              <p className="text-sm text-green-500 font-medium">Model loaded successfully</p>
            </div>
          ) : pendingFile ? (
            <div className="flex flex-col items-center">
              <div className="text-6xl mb-5">📦</div>
              <p className="text-xl text-gray-800 font-semibold mb-2 break-all">{pendingFile.name}</p>
              <p className="text-sm text-gray-500 font-medium">Ready to analyze - select dataset below</p>
            </div>
          ) : (
            <>
              <div className="text-6xl mb-5">🔧</div>
              <p className="text-xl text-gray-800 font-medium mb-2">
                Drag & drop a model file or click to upload
              </p>
              <p className="text-sm text-gray-400">Supports: .pth, .pt (PyTorch models)</p>
            </>
          )}
        </label>
      </div>

      {pendingFile && !analyzing && !fileName && datasets.length > 0 && (
        <div className="mt-5 p-5 bg-gray-50 rounded-xl border-2 border-gray-200">
          <label htmlFor="dataset-select" className="block mb-3 text-gray-800 font-medium">
            <strong>Select Dataset for Evaluation:</strong>
          </label>
          <select 
            id="dataset-select"
            value={selectedDataset} 
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="w-full p-3 border-2 border-gray-300 rounded-lg text-gray-800 bg-white cursor-pointer 
                       transition-all duration-300 hover:border-secondary-500 focus:outline-none focus:border-primary-500 
                       focus:ring-4 focus:ring-primary-500/10 mb-4"
          >
            {datasets.map((ds) => (
              <option key={ds.key} value={ds.key}>
                {ds.name}
              </option>
            ))}
          </select>
          <button 
            className="w-full py-4 px-6 bg-gradient-to-r from-primary-500 to-secondary-500 text-white rounded-lg font-semibold 
                       transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-primary-500/40 active:translate-y-0"
            onClick={handleAnalyze}
          >
            Analyze Model
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded-lg mt-4 border border-red-200">
          ⚠️ {error}
        </div>
      )}

      {fileName && !analyzing && (
        <button 
          className="mt-5 px-8 py-3 bg-gradient-to-r from-primary-500 to-secondary-500 text-white rounded-lg font-semibold 
                     transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-primary-500/40 active:translate-y-0"
          onClick={() => {
            setFileName(null);
            setPendingFile(null);
            onAnalysisComplete(null);
            setError(null);
            // Reset the file input element to allow selecting the same file again
            if (fileInputRef.current) {
              fileInputRef.current.value = '';
            }
          }}
        >
          Upload Another Model
        </button>
      )}
    </div>
  );
};

export default ModelDropzone;
