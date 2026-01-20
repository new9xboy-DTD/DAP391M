import React, { useState, useEffect } from 'react';
import './App.css';
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
      <div className="App">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading Application...</p>
        </div>
      </div>
    );
  }

  if (!isHealthy) {
    return (
      <div className="App">
        <div className="error">
          <h2>⚠️ Backend API Not Available</h2>
          <p>Please ensure the Flask backend is running on port 5000</p>
          <code>cd ViXNet/web_app/backend && python app.py</code>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 Multi-Model Deepfake Detection</h1>
        <p className="subtitle">Support for ViXNet, Xception Only, and ViT Only models</p>
      </header>

      <div className="container">
        {/* Model Architecture Section */}
        {modelInfo && modelInfo.loaded && (
          <section className="section">
            <h2>📊 Current Model Architecture</h2>
            <ModelArchitecture modelInfo={modelInfo} />
          </section>
        )}

        {/* Drag and Drop Sections */}
        <div className="dropzone-container">
          {/* Model Upload Section */}
          <section className="section">
            <h2>🔧 Model Upload & Analysis</h2>
            <p className="section-description">
              Drag and drop a model file (.pth) and select a dataset to analyze
            </p>
            <ModelDropzone 
              onAnalysisComplete={handleModelAnalyzed}
            />
          </section>

          {/* Image Upload Section - Only show if model is loaded */}
          {modelInfo && modelInfo.loaded && (
            <section className="section">
              <h2>🖼️ Image Inference</h2>
              <p className="section-description">
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
          <section className="section">
            <h2>📈 Results</h2>
            <ResultsDisplay 
              predictionResult={predictionResult}
              analysisResult={analysisResult}
            />
          </section>
        )}

        {/* Instructions when no model loaded */}
        {(!modelInfo || !modelInfo.loaded) && (
          <section className="section info-section">
            <h3>👋 Welcome!</h3>
            <p>To get started:</p>
            <ol>
              <li>Upload a trained model file (.pth or .pt)</li>
              <li>Select a dataset for evaluation</li>
              <li>Wait for the model analysis to complete</li>
              <li>Once loaded, you can upload images for deepfake detection</li>
            </ol>
          </section>
        )}
      </div>

      <footer className="App-footer">
        <p>Multi-Model Deepfake Detection System</p>
      </footer>
    </div>
  );
}

export default App;
