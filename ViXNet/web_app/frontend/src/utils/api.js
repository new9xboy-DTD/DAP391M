import axios from 'axios';

const API_BASE_URL = '/api';

export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export const getDatasets = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/datasets`);
    return response.data;
  } catch (error) {
    console.error('Failed to get datasets:', error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/model-info`);
    return response.data;
  } catch (error) {
    console.error('Failed to get model info:', error);
    throw error;
  }
};

export const predictImage = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
};

export const analyzeModel = async (modelFile, dataset = 'default') => {
  try {
    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('dataset', dataset);

    const response = await axios.post(`${API_BASE_URL}/analyze-model`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'json',
    });
    console.log("typeof response.data:", typeof response.data);
    
    // Handle response - if it's already an object, return it
    if (typeof response.data === 'object') {
      return response.data;
    }
    
    // If it's a string, sanitize and parse
    const safeData = response.data
      .replace(/Infinity/g, 'null')
      .replace(/NaN/g, 'null');
    return JSON.parse(safeData);
  } catch (error) {
    console.error('Model analysis failed:', error);
    throw error;
  }
};

export const calculateAUC = async (dataset = 'default') => {
  try {
    const response = await axios.post(`${API_BASE_URL}/calculate-auc`, { dataset });
    return response.data;
  } catch (error) {
    console.error('AUC calculation failed:', error);
    throw error;
  }
};
