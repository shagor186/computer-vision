import React, { useState, useEffect } from 'react';
import axios from 'axios';

const TrainModel = () => {
  const [training, setTraining] = useState(false);
  const [message, setMessage] = useState('');
  const [modelStatus, setModelStatus] = useState(null);

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/model-status');
      setModelStatus(response.data);
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const handleTrainModel = async () => {
    setTraining(true);
    setMessage('');

    try {
      const response = await axios.post('http://localhost:5000/api/train-model');
      setMessage(`✅ ${response.data.message}`);
      checkModelStatus(); // Update status after training
    } catch (error) {
      setMessage(`❌ Error: ${error.response?.data?.message || error.message}`);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-2xl mx-auto px-4">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
            Train Face Recognition Model
          </h1>

          {/* Model Status */}
          {modelStatus && (
            <div className={`p-4 rounded-lg mb-6 ${
              modelStatus.model_exists 
                ? 'bg-green-50 border border-green-200 text-green-700'
                : 'bg-yellow-50 border border-yellow-200 text-yellow-700'
            }`}>
              <div className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  {modelStatus.model_exists ? (
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  ) : (
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  )}
                </svg>
                <span className="font-semibold">{modelStatus.message}</span>
              </div>
            </div>
          )}

          {message && (
            <div className={`p-4 rounded-lg mb-6 ${
              message.includes('✅') 
                ? 'bg-green-50 text-green-700 border border-green-200'
                : 'bg-red-50 text-red-700 border border-red-200'
            }`}>
              {message}
            </div>
          )}

          {/* Training Information */}
          <div className="bg-blue-50 rounded-lg p-6 mb-6">
            <h3 className="font-semibold text-blue-800 mb-3">ℹ️ About Model Training</h3>
            <ul className="text-blue-700 space-y-2 text-sm">
              <li>• <strong>MTCNN</strong> detects faces in your uploaded images</li>
              <li>• <strong>FaceNet</strong> generates 512-dimensional face embeddings</li>
              <li>• <strong>SVM Classifier</strong> learns to recognize different persons</li>
              <li>• Training time depends on number of persons and images</li>
              <li>• Model will be saved as <code>svm_face_model.pkl</code></li>
            </ul>
          </div>

          {/* Train Button */}
          <div className="text-center">
            <button
              onClick={handleTrainModel}
              disabled={training}
              className="bg-green-600 text-white py-3 px-8 rounded-lg hover:bg-green-700 focus:ring-4 focus:ring-green-200 disabled:opacity-50 disabled:cursor-not-allowed transition text-lg font-semibold"
            >
              {training ? (
                <div className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Training Model...
                </div>
              ) : (
                '🚀 Train Model Now'
              )}
            </button>
          </div>

          {/* Requirements */}
          <div className="mt-8 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold text-gray-700 mb-2">📋 Before Training:</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Make sure you have added persons using "Add Person" page</li>
              <li>• Each person should have multiple face images</li>
              <li>• Ensure good quality images for better accuracy</li>
              <li>• Training may take several minutes depending on data size</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainModel;