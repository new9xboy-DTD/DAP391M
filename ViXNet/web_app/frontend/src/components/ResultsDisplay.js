import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ResultsDisplay.css';

const ResultsDisplay = ({ predictionResult, analysisResult }) => {
  return (
    <div className="results-display">
      {/* Image Prediction Results */}
      {predictionResult && (
        <div className="result-card prediction-card">
          <h3>🖼️ Image Prediction Results</h3>
          
          <div className="prediction-result">
            <div className={`prediction-badge ${predictionResult.prediction.toLowerCase()}`}>
              {predictionResult.prediction === 'Real' ? '✓' : '✗'} {predictionResult.prediction}
            </div>
            
            <div className="confidence-meter">
              <div className="confidence-label">
                Confidence: {(predictionResult.confidence * 100).toFixed(2)}%
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill"
                  style={{ 
                    width: `${predictionResult.confidence * 100}%`,
                    background: predictionResult.prediction === 'Real' 
                      ? 'linear-gradient(90deg, #27ae60, #2ecc71)' 
                      : 'linear-gradient(90deg, #e74c3c, #c0392b)'
                  }}
                />
              </div>
            </div>

            <div className="probabilities">
              <h4>Class Probabilities</h4>
              <div className="probability-grid">
                <div className="probability-item">
                  <span className="prob-label fake-label">Fake:</span>
                  <span className="prob-value">{(predictionResult.probabilities.Fake * 100).toFixed(2)}%</span>
                  <div className="prob-bar">
                    <div 
                      className="prob-fill fake-fill"
                      style={{ width: `${predictionResult.probabilities.Fake * 100}%` }}
                    />
                  </div>
                </div>
                <div className="probability-item">
                  <span className="prob-label real-label">Real:</span>
                  <span className="prob-value">{(predictionResult.probabilities.Real * 100).toFixed(2)}%</span>
                  <div className="prob-bar">
                    <div 
                      className="prob-fill real-fill"
                      style={{ width: `${predictionResult.probabilities.Real * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Analysis Results */}
      {analysisResult && analysisResult.model_info && (
        <div className="result-card analysis-card">
          <h3>🔧 Model Analysis Results</h3>
          
          {/* AUC Results */}
          {analysisResult.model_info.auc_results && (
            <div className="auc-section">
              <div className="auc-header">
                <h4>📊 Model Performance Metrics</h4>
              </div>
              
              <div className="metrics-grid">
                <div className="metric-card highlight">
                  <div className="metric-icon">📈</div>
                  <div className="metric-label">AUC Score</div>
                  <div className="metric-value large">
                    {analysisResult.model_info.auc_results.auc.toFixed(4)}
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-icon">🎯</div>
                  <div className="metric-label">Accuracy</div>
                  <div className="metric-value">
                    {(analysisResult.model_info.auc_results.accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                
                <div className="metric-card">
                  <div className="metric-icon">📦</div>
                  <div className="metric-label">Test Samples</div>
                  <div className="metric-value">
                    {analysisResult.model_info.auc_results.num_samples}
                  </div>
                </div>
              </div>

              {/* Confusion Matrix */}
              {analysisResult.model_info.auc_results.confusion_matrix && (
                <div className="confusion-matrix-section">
                  <h4>Confusion Matrix</h4>
                  <div className="confusion-matrix">
                    <div className="cm-labels">
                      <div className="cm-label-col">
                        <div>Predicted</div>
                      </div>
                      <div className="cm-label-row">
                        <div>Actual</div>
                      </div>
                    </div>
                    <table className="cm-table">
                      <thead>
                        <tr>
                          <th></th>
                          <th>Fake</th>
                          <th>Real</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <th>Fake</th>
                          <td className="cm-cell true-negative">
                            {analysisResult.model_info.auc_results.confusion_matrix[0][0]}
                          </td>
                          <td className="cm-cell false-positive">
                            {analysisResult.model_info.auc_results.confusion_matrix[0][1]}
                          </td>
                        </tr>
                        <tr>
                          <th>Real</th>
                          <td className="cm-cell false-negative">
                            {analysisResult.model_info.auc_results.confusion_matrix[1][0]}
                          </td>
                          <td className="cm-cell true-positive">
                            {analysisResult.model_info.auc_results.confusion_matrix[1][1]}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* ROC Curve */}
              {analysisResult.model_info.auc_results.roc_curve && (
                <div className="roc-curve-section">
                  <h4>ROC Curve</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart
                      data={analysisResult.model_info.auc_results.roc_curve.fpr.map((fpr, i) => ({
                        fpr: fpr,
                        tpr: analysisResult.model_info.auc_results.roc_curve.tpr[i]
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="fpr" 
                        label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
                      />
                      <YAxis 
                        label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                      />
                      <Tooltip />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="tpr" 
                        stroke="#667eea" 
                        strokeWidth={3}
                        dot={false}
                        name="ROC Curve"
                      />
                      <Line 
                        type="monotone" 
                        data={[{fpr: 0, tpr: 0}, {fpr: 1, tpr: 1}]}
                        dataKey="tpr"
                        stroke="#999" 
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        name="Random Classifier"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="roc-info">
                    AUC = {analysisResult.model_info.auc_results.auc.toFixed(4)}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Model Info */}
          <div className="model-details">
            <h4>Model Details</h4>
            <div className="detail-grid">
              <div className="detail-item">
                <span className="detail-label">Epoch:</span>
                <span className="detail-value">{analysisResult.model_info.epoch}</span>
              </div>
              {analysisResult.model_info.metrics?.accuracy && (
                <div className="detail-item">
                  <span className="detail-label">Validation Accuracy:</span>
                  <span className="detail-value">
                    {(analysisResult.model_info.metrics.accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {analysisResult.model_info.architecture && (
                <>
                  <div className="detail-item">
                    <span className="detail-label">Architecture:</span>
                    <span className="detail-value">{analysisResult.model_info.architecture.name}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Xception Features:</span>
                    <span className="detail-value">{analysisResult.model_info.architecture.xception_dim}D</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">ViT Features:</span>
                    <span className="detail-value">{analysisResult.model_info.architecture.vit_dim}D</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Fusion Dimension:</span>
                    <span className="detail-value">{analysisResult.model_info.architecture.fusion_dim}D</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
