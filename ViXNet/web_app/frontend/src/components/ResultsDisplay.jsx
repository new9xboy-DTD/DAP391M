import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ResultsDisplay = ({ predictionResult, analysisResult }) => {
  return (
    <div className="flex flex-col gap-8">
      {/* Image Prediction Results */}
      {predictionResult && (
        <div className="bg-white p-8 rounded-2xl shadow-lg">
          <h3 className="text-2xl font-bold text-gray-800 mb-6 pb-3 border-b-2 border-gray-100">
            🖼️ Image Prediction Results
          </h3>
          
          <div className="flex flex-col gap-6">
            <div className={`
              self-center py-5 px-12 rounded-2xl text-3xl font-bold shadow-lg
              ${predictionResult.prediction === 'Real' 
                ? 'bg-gradient-to-r from-green-500 to-green-400 text-white' 
                : 'bg-gradient-to-r from-red-500 to-red-400 text-white'
              }
            `}>
              {predictionResult.prediction === 'Real' ? '✓' : '✗'} {predictionResult.prediction}
            </div>
            
            <div className="w-full">
              <div className="text-xl font-semibold text-gray-800 mb-3">
                Confidence: {(predictionResult.confidence * 100).toFixed(2)}%
              </div>
              <div className="h-8 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                <div 
                  className="h-full rounded-full transition-all duration-500"
                  style={{ 
                    width: `${predictionResult.confidence * 100}%`,
                    background: predictionResult.prediction === 'Real' 
                      ? 'linear-gradient(90deg, #27ae60, #2ecc71)' 
                      : 'linear-gradient(90deg, #e74c3c, #c0392b)'
                  }}
                />
              </div>
            </div>

            <div className="bg-gray-50 p-5 rounded-xl">
              <h4 className="text-lg font-semibold text-gray-700 mb-4">Class Probabilities</h4>
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-[80px_80px_1fr] items-center gap-3">
                  <span className="font-semibold text-red-500">Fake:</span>
                  <span className="font-semibold text-gray-800">{(predictionResult.probabilities.Fake * 100).toFixed(2)}%</span>
                  <div className="h-5 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-full transition-all duration-500"
                      style={{ width: `${predictionResult.probabilities.Fake * 100}%` }}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-[80px_80px_1fr] items-center gap-3">
                  <span className="font-semibold text-green-500">Real:</span>
                  <span className="font-semibold text-gray-800">{(predictionResult.probabilities.Real * 100).toFixed(2)}%</span>
                  <div className="h-5 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full transition-all duration-500"
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
        <div className="bg-white p-8 rounded-2xl shadow-lg">
          <h3 className="text-2xl font-bold text-gray-800 mb-6 pb-3 border-b-2 border-gray-100">
            🔧 Model Analysis Results
          </h3>
          
          {/* AUC Results */}
          {analysisResult.model_info.auc_results && (
            <div className="flex flex-col gap-6">
              <div>
                <h4 className="text-xl font-semibold text-gray-800 mb-4">📊 Model Performance Metrics</h4>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="bg-gradient-to-br from-primary-100 to-secondary-100 border-2 border-primary-500 p-5 rounded-xl text-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
                  <div className="text-4xl mb-3">📈</div>
                  <div className="text-sm text-gray-500 font-medium mb-2">AUC Score</div>
                  <div className="text-4xl font-bold bg-gradient-to-r from-primary-500 to-secondary-500 bg-clip-text text-transparent">
                    {analysisResult.model_info.auc_results.auc.toFixed(4)}
                  </div>
                </div>
                
                <div className="bg-gradient-to-br from-gray-50 to-white border-2 border-gray-200 p-5 rounded-xl text-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
                  <div className="text-4xl mb-3">🎯</div>
                  <div className="text-sm text-gray-500 font-medium mb-2">Accuracy</div>
                  <div className="text-3xl font-bold text-gray-800">
                    {(analysisResult.model_info.auc_results.accuracy * 100).toFixed(2)}%
                  </div>
                </div>
                
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 border-2 border-purple-300 p-5 rounded-xl text-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
                  <div className="text-4xl mb-3">⚖️</div>
                  <div className="text-sm text-gray-500 font-medium mb-2">F1 Score</div>
                  <div className="text-3xl font-bold text-purple-700">
                    {analysisResult.model_info.auc_results.f1_score !== undefined 
                      ? (analysisResult.model_info.auc_results.f1_score * 100).toFixed(2) + '%'
                      : 'N/A'}
                  </div>
                </div>
                
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-300 p-5 rounded-xl text-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
                  <div className="text-4xl mb-3">🎯</div>
                  <div className="text-sm text-gray-500 font-medium mb-2">Precision</div>
                  <div className="text-3xl font-bold text-blue-700">
                    {analysisResult.model_info.auc_results.precision !== undefined 
                      ? (analysisResult.model_info.auc_results.precision * 100).toFixed(2) + '%'
                      : 'N/A'}
                  </div>
                </div>
                
                <div className="bg-gradient-to-br from-orange-50 to-orange-100 border-2 border-orange-300 p-5 rounded-xl text-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
                  <div className="text-4xl mb-3">🔍</div>
                  <div className="text-sm text-gray-500 font-medium mb-2">Recall</div>
                  <div className="text-3xl font-bold text-orange-700">
                    {analysisResult.model_info.auc_results.recall !== undefined 
                      ? (analysisResult.model_info.auc_results.recall * 100).toFixed(2) + '%'
                      : 'N/A'}
                  </div>
                </div>
                
                <div className="bg-gradient-to-br from-gray-50 to-white border-2 border-gray-200 p-5 rounded-xl text-center transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">
                  <div className="text-4xl mb-3">📦</div>
                  <div className="text-sm text-gray-500 font-medium mb-2">Test Samples</div>
                  <div className="text-3xl font-bold text-gray-800">
                    {analysisResult.model_info.auc_results.num_samples}
                  </div>
                </div>
              </div>

              {/* Confusion Matrix */}
              {analysisResult.model_info.auc_results.confusion_matrix && (
                <div className="bg-gray-50 p-5 rounded-xl">
                  <h4 className="text-lg font-semibold text-gray-700 mb-4">Confusion Matrix</h4>
                  <div className="flex justify-center">
                    <div className="inline-block">
                      <div className="text-center text-sm font-semibold text-gray-600 mb-2">Predicted</div>
                      <div className="flex">
                        <div className="flex flex-col justify-center mr-2">
                          <div className="text-sm font-semibold text-gray-600 transform -rotate-90 whitespace-nowrap" style={{width: '20px'}}>Actual</div>
                        </div>
                        <table className="border-collapse bg-white rounded-lg overflow-hidden" style={{width: '220px'}}>
                          <thead>
                            <tr>
                              <th className="p-3 bg-gray-100 font-semibold text-gray-800 border border-gray-200 w-16"></th>
                              <th className="p-3 bg-gray-100 font-semibold text-gray-800 border border-gray-200 w-20">Fake</th>
                              <th className="p-3 bg-gray-100 font-semibold text-gray-800 border border-gray-200 w-20">Real</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <th className="p-3 bg-gray-100 font-semibold text-gray-800 border border-gray-200">Fake</th>
                              <td className="p-3 text-center text-lg font-semibold bg-green-100 text-green-800 border border-gray-200" style={{width: '70px', height: '50px'}}>
                                {analysisResult.model_info.auc_results.confusion_matrix[0][0]}
                              </td>
                              <td className="p-3 text-center text-lg font-semibold bg-red-100 text-red-800 border border-gray-200" style={{width: '70px', height: '50px'}}>
                                {analysisResult.model_info.auc_results.confusion_matrix[0][1]}
                              </td>
                            </tr>
                            <tr>
                              <th className="p-3 bg-gray-100 font-semibold text-gray-800 border border-gray-200">Real</th>
                              <td className="p-3 text-center text-lg font-semibold bg-red-100 text-red-800 border border-gray-200" style={{width: '70px', height: '50px'}}>
                                {analysisResult.model_info.auc_results.confusion_matrix[1][0]}
                              </td>
                              <td className="p-3 text-center text-lg font-semibold bg-green-100 text-green-800 border border-gray-200" style={{width: '70px', height: '50px'}}>
                                {analysisResult.model_info.auc_results.confusion_matrix[1][1]}
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ROC Curve */}
              {analysisResult.model_info.auc_results.roc_curve && (
                <div className="bg-gray-50 p-5 rounded-xl">
                  <h4 className="text-lg font-semibold text-gray-700 mb-4">ROC Curve</h4>
                  <div className="flex justify-center">
                    <div style={{width: '350px', height: '350px'}}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="fpr" 
                            type="number"
                            domain={[0, 1]}
                            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -10 }}
                            tick={{ fontSize: 11 }}
                          />
                          <YAxis 
                            type="number"
                            domain={[0, 1]}
                            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 5 }}
                            tick={{ fontSize: 11 }}
                          />
                          <Tooltip />
                          <Legend wrapperStyle={{ paddingTop: '10px' }} />
                          <Line 
                            type="monotone"
                            data={analysisResult.model_info.auc_results.roc_curve.fpr.map((fpr, i) => ({
                              fpr: fpr,
                              tpr: analysisResult.model_info.auc_results.roc_curve.tpr[i]
                            }))}
                            dataKey="tpr" 
                            stroke="#0ea5e9" 
                            strokeWidth={2}
                            dot={false}
                            name="ROC Curve"
                          />
                          <Line 
                            type="monotone"
                            data={[{fpr: 0, tpr: 0}, {fpr: 1, tpr: 1}]}
                            dataKey="tpr"
                            stroke="#999" 
                            strokeWidth={1.5}
                            strokeDasharray="5 5"
                            dot={false}
                            name="Random Classifier"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <p className="text-center text-sm font-semibold text-primary-500 mt-3">
                    AUC = {analysisResult.model_info.auc_results.auc.toFixed(4)}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Model Info */}
          <div className="bg-gray-50 p-5 rounded-xl mt-5">
            <h4 className="text-lg font-semibold text-gray-700 mb-4">Model Details</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="flex justify-between items-center p-3 bg-white rounded-lg border-l-4 border-primary-500">
                <span className="font-semibold text-gray-500 text-sm">Epoch:</span>
                <span className="font-semibold text-gray-800">{analysisResult.model_info.epoch}</span>
              </div>
              {analysisResult.model_info.metrics?.accuracy && (
                <div className="flex justify-between items-center p-3 bg-white rounded-lg border-l-4 border-primary-500">
                  <span className="font-semibold text-gray-500 text-sm">Validation Accuracy:</span>
                  <span className="font-semibold text-gray-800">
                    {(analysisResult.model_info.metrics.accuracy * 100).toFixed(2)}%
                  </span>
                </div>
              )}
              {analysisResult.model_info.architecture && (
                <>
                  <div className="flex justify-between items-center p-3 bg-white rounded-lg border-l-4 border-primary-500">
                    <span className="font-semibold text-gray-500 text-sm">Architecture:</span>
                    <span className="font-semibold text-gray-800">{analysisResult.model_info.architecture.name}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white rounded-lg border-l-4 border-primary-500">
                    <span className="font-semibold text-gray-500 text-sm">Xception Features:</span>
                    <span className="font-semibold text-gray-800">{analysisResult.model_info.architecture.xception_dim}D</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white rounded-lg border-l-4 border-primary-500">
                    <span className="font-semibold text-gray-500 text-sm">ViT Features:</span>
                    <span className="font-semibold text-gray-800">{analysisResult.model_info.architecture.vit_dim}D</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-white rounded-lg border-l-4 border-primary-500">
                    <span className="font-semibold text-gray-500 text-sm">Fusion Dimension:</span>
                    <span className="font-semibold text-gray-800">{analysisResult.model_info.architecture.fusion_dim}D</span>
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
