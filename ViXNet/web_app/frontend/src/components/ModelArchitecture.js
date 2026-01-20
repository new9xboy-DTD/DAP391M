import React from 'react';

const ModelArchitecture = ({ modelInfo }) => {
  if (!modelInfo) {
    return (
      <div className="w-full">
        <div className="text-center py-10 text-gray-400 text-lg">
          <p>No model loaded</p>
        </div>
      </div>
    );
  }

  const { architecture, loaded, epoch, metrics } = modelInfo;

  return (
    <div className="w-full">
      <div className="flex flex-col items-center gap-5 p-5 bg-gradient-to-b from-gray-50 to-white rounded-xl">
        {/* Input Layer */}
        <div className="w-full flex justify-center">
          <div className="bg-gradient-to-r from-primary-500 to-secondary-500 text-white py-5 px-10 rounded-xl text-center shadow-lg">
            <h3 className="text-xl font-semibold mb-1">Input Image</h3>
            <p className="text-sm opacity-90">224 × 224 × 3</p>
          </div>
        </div>

        {/* Branches */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
          {/* Xception Branch */}
          <div className="bg-white border-2 border-primary-500 rounded-xl p-5 shadow-md">
            <div className="text-center mb-4 pb-4 border-b-2 border-gray-100">
              <h3 className="text-xl font-semibold mb-1">🔷 Xception Branch</h3>
              <p className="text-sm text-gray-500">CNN - Global Spatial Features</p>
            </div>
            <div className="flex flex-col items-center gap-3">
              <div className="w-full bg-gray-100 p-3 rounded-lg text-center font-medium">Pretrained Xception</div>
              <span className="text-2xl text-gray-400">↓</span>
              <div className="w-full bg-gray-100 p-3 rounded-lg text-center font-medium">Global Average Pooling</div>
              <span className="text-2xl text-gray-400">↓</span>
              <div className="w-full bg-gradient-to-r from-primary-100 to-secondary-100 p-3 rounded-lg text-center font-semibold text-gray-800">
                Features: {architecture?.xception_dim || 2048}D
              </div>
            </div>
          </div>

          {/* ViT Branch */}
          <div className="bg-white border-2 border-secondary-500 rounded-xl p-5 shadow-md">
            <div className="text-center mb-4 pb-4 border-b-2 border-gray-100">
              <h3 className="text-xl font-semibold mb-1">🔶 ViT Branch</h3>
              <p className="text-sm text-gray-500">Transformer - Patch-wise Attention</p>
            </div>
            <div className="flex flex-col items-center gap-3">
              <div className="w-full bg-gray-100 p-3 rounded-lg text-center font-medium">Vision Transformer (Tiny)</div>
              <span className="text-2xl text-gray-400">↓</span>
              <div className="w-full bg-gray-100 p-3 rounded-lg text-center font-medium">Patch Embedding + Self-Attention</div>
              <span className="text-2xl text-gray-400">↓</span>
              <div className="w-full bg-gradient-to-r from-primary-100 to-secondary-100 p-3 rounded-lg text-center font-semibold text-gray-800">
                Features: {architecture?.vit_dim || 192}D
              </div>
            </div>
          </div>
        </div>

        {/* Fusion Section */}
        <div className="w-full max-w-2xl flex flex-col items-center">
          <span className="text-3xl text-gray-400">↓</span>
          <div className="w-full bg-gradient-to-r from-primary-500 to-secondary-500 text-white p-5 rounded-xl text-center shadow-lg">
            <h3 className="text-xl font-semibold mb-1">🔗 Feature Fusion</h3>
            <p className="text-sm opacity-90 mb-3">Concatenation + Dense Layers</p>
            <div className="bg-white/20 py-2 px-4 rounded-lg text-sm">
              Output: {architecture?.fusion_dim || 512}D
            </div>
          </div>
        </div>

        {/* Classification Section */}
        <div className="w-full max-w-xl flex flex-col items-center">
          <span className="text-2xl text-gray-400">↓</span>
          <div className="w-full bg-white border-2 border-primary-500 p-5 rounded-xl text-center shadow-lg">
            <h3 className="text-xl font-semibold text-gray-800 mb-1">🎯 Classification Head</h3>
            <p className="text-sm text-gray-500 mb-4">Binary Classification</p>
            <div className="flex gap-5 justify-center">
              <span className="py-3 px-6 bg-red-50 text-red-500 border-2 border-red-200 rounded-lg font-semibold text-lg">
                Fake
              </span>
              <span className="py-3 px-6 bg-green-50 text-green-500 border-2 border-green-200 rounded-lg font-semibold text-lg">
                Real
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Info Panel */}
      {loaded && (
        <div className="mt-8 bg-gray-50 p-5 rounded-xl border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">📊 Model Information</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="flex flex-col gap-1">
              <span className="text-sm font-medium text-gray-500">Status:</span>
              <span className="text-lg font-semibold text-green-500">✓ Loaded</span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-sm font-medium text-gray-500">Epoch:</span>
              <span className="text-lg font-semibold text-gray-800">{epoch}</span>
            </div>
            {metrics?.accuracy && (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-gray-500">Validation Accuracy:</span>
                <span className="text-lg font-semibold text-gray-800">{(metrics.accuracy * 100).toFixed(2)}%</span>
              </div>
            )}
            {metrics?.precision && (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-gray-500">Precision:</span>
                <span className="text-lg font-semibold text-gray-800">{(metrics.precision * 100).toFixed(2)}%</span>
              </div>
            )}
            {metrics?.recall && (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-gray-500">Recall:</span>
                <span className="text-lg font-semibold text-gray-800">{(metrics.recall * 100).toFixed(2)}%</span>
              </div>
            )}
            {metrics?.f1 && (
              <div className="flex flex-col gap-1">
                <span className="text-sm font-medium text-gray-500">F1 Score:</span>
                <span className="text-lg font-semibold text-gray-800">{(metrics.f1 * 100).toFixed(2)}%</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelArchitecture;
