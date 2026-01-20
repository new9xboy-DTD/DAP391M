import React, { useState, useCallback } from 'react';
import { predictImage } from '../utils/api';

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

  const handleFile = useCallback(async (file) => {
    // Validate file type (both MIME type and extension)
    const validExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'];
    const fileName = file.name.toLowerCase();
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
    
    if (!file.type.startsWith('image/') || !hasValidExtension) {
      setError('Please upload a valid image file (JPG, PNG, JPEG, etc.)');
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
  }, [onPredictionComplete]);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(null);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      await handleFile(files[0]);
    }
  }, [handleFile]);

  const handleChange = async (e) => {
    e.preventDefault();
    setError(null);
    
    if (e.target.files && e.target.files[0]) {
      await handleFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full">
      <div
        className={`
          border-3 border-dashed rounded-2xl p-10 text-center cursor-pointer 
          transition-all duration-300 min-h-[300px] flex items-center justify-center
          ${dragActive 
            ? 'border-primary-500 bg-primary-100 scale-[1.02]' 
            : uploading 
              ? 'border-secondary-500 bg-purple-50' 
              : 'border-gray-300 bg-gray-50 hover:border-primary-500 hover:bg-primary-50'
          }
        `}
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
          className="hidden"
        />
        
        {preview ? (
          <div className="relative w-full max-w-md mx-auto">
            <img src={preview} alt="Preview" className="w-full max-h-96 object-contain rounded-xl shadow-lg" />
            {uploading && (
              <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center rounded-xl text-white">
                <div className="w-10 h-10 border-4 border-white/30 border-t-white rounded-full animate-spin mb-3"></div>
                <p className="font-medium">Analyzing image...</p>
              </div>
            )}
          </div>
        ) : (
          <label htmlFor="image-upload" className="flex flex-col items-center cursor-pointer w-full">
            <div className="text-6xl mb-5">📸</div>
            <p className="text-xl text-gray-800 font-medium mb-2">
              {uploading ? 'Processing...' : 'Drag & drop an image or click to upload'}
            </p>
            <p className="text-sm text-gray-400">Supports: JPG, PNG, JPEG</p>
          </label>
        )}
      </div>

      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded-lg mt-4 border border-red-200">
          ⚠️ {error}
        </div>
      )}

      {preview && !uploading && (
        <button 
          className="mt-5 px-8 py-3 bg-gradient-to-r from-primary-500 to-secondary-500 text-white rounded-lg font-semibold 
                     transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-primary-500/40 active:translate-y-0"
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
