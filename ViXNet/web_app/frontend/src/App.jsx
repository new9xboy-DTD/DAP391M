import React, { useState, useEffect } from 'react';
import ImageDropzone from './components/ImageDropzone';
import ModelDropzone from './components/ModelDropzone';
import ModelArchitecture from './components/ModelArchitecture';
import ResultsDisplay from './components/ResultsDisplay';
import { checkHealth, getModelInfo } from './utils/api';

function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isHealthy, setIsHealthy] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check API health on mount (but don't load model)
    const initializeApp = async () => {
      try {
        const health = await checkHealth();
        setIsHealthy(health.status === 'healthy');
        
        // Check if a model is already loaded (from previous session)
        if (health.model_loaded) {
          const info = await getModelInfo();
          setModelInfo(info);
        }
      } catch (error) {
        console.error('Failed to initialize app:', error);
        setIsHealthy(false);
      } finally {
        setLoading(false);
      }
    };

    initializeApp();
  }, []);

  const handleModelAnalyzed = (result) => {
    setAnalysisResult(result);
    if (result && result.model_info) {
      setModelInfo(result.model_info);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-white p-5">
        <div className="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin mb-4"></div>
        <p className="text-lg font-medium">Loading Application...</p>
      </div>
    );
  }

  if (!isHealthy) {
    return (
      <div className="min-h-screen flex items-center justify-center p-5">
        <div className="bg-white/95 rounded-2xl p-10 max-w-xl text-center shadow-2xl">
          <h2 className="text-2xl font-bold text-red-500 mb-4">⚠️ Backend API Not Available</h2>
          <p className="text-gray-600 mb-4">Please ensure the Flask backend is running on port 5000</p>
          <code className="block bg-gray-100 p-4 rounded-lg font-mono text-sm">
            cd ViXNet/web_app/backend && python app.py
          </code>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-5 text-center">
      {/* Header */}
      <header className="bg-white/95 p-8 rounded-2xl mb-8 shadow-xl backdrop-blur">
        <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-primary-500 to-secondary-500 bg-clip-text text-transparent">
          🧠 Multi-Model Deepfake Detection
        </h1>
        <p className="text-gray-500 text-lg mt-3">
          Support for ViXNet, Xception Only, and ViT Only models
        </p>
      </header>

      <div className="max-w-7xl mx-auto">
        {/* Model Architecture Section */}
        {modelInfo && modelInfo.loaded && (
          <section className="bg-white/95 p-8 rounded-2xl mb-8 shadow-lg">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">📊 Current Model Architecture</h2>
            <ModelArchitecture modelInfo={modelInfo} />
          </section>
        )}

        {/* Drag and Drop Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Model Upload Section */}
          <section className="bg-white/95 p-8 rounded-2xl shadow-lg">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">🔧 Model Upload & Analysis</h2>
            <p className="text-gray-500 mb-6">
              Drag and drop a model file (.pth) and select a dataset to analyze
            </p>
            <ModelDropzone 
              onAnalysisComplete={handleModelAnalyzed}
            />
          </section>

          {/* Image Upload Section - Only show if model is loaded */}
          {modelInfo && modelInfo.loaded && (
            <section className="bg-white/95 p-8 rounded-2xl shadow-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">🖼️ Image Inference</h2>
              <p className="text-gray-500 mb-6">
                Drag and drop an image to detect if it's real or fake
              </p>
              <ImageDropzone 
                onPredictionComplete={setPredictionResult}
              />
            </section>
          )}
        </div>

        {/* Results Section */}
        {(predictionResult || analysisResult) && (
          <section className="bg-white/95 p-8 rounded-2xl shadow-lg mb-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">📈 Results</h2>
            <ResultsDisplay 
              predictionResult={predictionResult}
              analysisResult={analysisResult}
            />
          </section>
        )}

        {/* Instructions when no model loaded */}
        {(!modelInfo || !modelInfo.loaded) && (
          <section className="bg-gradient-to-br from-primary-500/10 to-secondary-500/10 border-2 border-dashed border-primary-500 p-8 rounded-2xl mt-8">
            <h3 className="text-xl font-bold text-gray-800 mb-4">👋 Welcome!</h3>
            <p className="text-gray-600 mb-4">To get started:</p>
            <ol className="text-left text-gray-600 leading-relaxed pl-6 list-decimal space-y-2">
              <li>Upload a trained model file (.pth or .pt)</li>
              <li>Select a dataset for evaluation</li>
              <li>Wait for the model analysis to complete</li>
              <li>Once loaded, you can upload images for deepfake detection</li>
            </ol>
          </section>
        )}
      </div>

      {/* Footer */}
      <footer className="text-white/80 mt-12 py-6 text-sm">
        <p>Multi-Model Deepfake Detection System</p>
      </footer>
    </div>
  );
}

export default App;
