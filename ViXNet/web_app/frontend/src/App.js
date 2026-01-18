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
    // Check API health and load model info on mount
    const initializeApp = async () => {
      try {
        const health = await checkHealth();
        setIsHealthy(health.status === 'healthy');
        
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
    if (result.model_info) {
      setModelInfo(result.model_info);
    }
  };

  if (loading) {
    return (
      <div className="App">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading ViXNet...</p>
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
        <h1>🧠 ViXNet Model Visualization</h1>
        <p className="subtitle">Vision Transformer + Xception Network for Deepfake Detection</p>
      </header>

      <div className="container">
        {/* Model Architecture Section */}
        <section className="section">
          <h2>📊 Model Architecture</h2>
          <ModelArchitecture modelInfo={modelInfo} />
        </section>

        {/* Drag and Drop Sections */}
        <div className="dropzone-container">
          {/* Image Upload Section */}
          <section className="section">
            <h2>🖼️ Image Inference</h2>
            <p className="section-description">
              Drag and drop an image to detect if it's real or fake
            </p>
            <ImageDropzone 
              onPredictionComplete={setPredictionResult}
            />
          </section>

          {/* Model Upload Section */}
          <section className="section">
            <h2>🔧 Model Analysis</h2>
            <p className="section-description">
              Drag and drop a model file (.pth) to analyze and calculate AUC
            </p>
            <ModelDropzone 
              onAnalysisComplete={handleModelAnalyzed}
            />
          </section>
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
      </div>

      <footer className="App-footer">
        <p>ViXNet - Expert Systems with Applications (Q1 Journal)</p>
      </footer>
    </div>
  );
}

export default App;
