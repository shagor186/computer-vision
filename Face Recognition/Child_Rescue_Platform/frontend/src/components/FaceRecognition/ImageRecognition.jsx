import React, { useState, useRef } from 'react';
import axios from 'axios';

const ImageRecognition = ({ onBack }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [recognizing, setRecognizing] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setRecognitionResult(null);
      setError('');
    }
  };

  const recognizeImage = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setRecognizing(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await axios.post('http://127.0.0.1:5000/api/face/recognize-image', formData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      setRecognitionResult(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Recognition failed');
    } finally {
      setRecognizing(false);
    }
  };

  const clearSelection = () => {
    setSelectedImage(null);
    setRecognitionResult(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <button
            onClick={onBack}
            className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-4"
          >
            ← Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Image Recognition</h1>
          <p className="text-gray-600">
            Upload an image to recognize faces using the trained model
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          {/* Image Upload */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Image</h2>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageSelect}
                accept=".jpg,.jpeg,.png,.bmp,.webp"
                className="hidden"
                id="recognition-upload"
              />
              <label
                htmlFor="recognition-upload"
                className="cursor-pointer inline-flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
                </svg>
                Select Image
              </label>
              <p className="text-gray-500 mt-2">
                Supported formats: JPG, JPEG, PNG, BMP, WEBP
              </p>
            </div>

            {/* Image Preview */}
            {selectedImage && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Selected Image</h3>
                <div className="flex flex-col items-center">
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="Preview"
                    className="max-w-full h-64 object-contain rounded-lg border"
                  />
                  <button
                    onClick={clearSelection}
                    className="mt-4 px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-lg transition-colors"
                  >
                    Clear Selection
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Recognition Controls */}
          <div className="text-center mb-8">
            <button
              onClick={recognizeImage}
              disabled={!selectedImage || recognizing}
              className={`px-8 py-4 text-white font-semibold rounded-lg text-lg transition-colors ${
                !selectedImage || recognizing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {recognizing ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Recognizing...
                </span>
              ) : (
                'Recognize Face'
              )}
            </button>
          </div>

          {/* Results */}
          {recognitionResult && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-green-800 mb-4">Recognition Result</h3>
              {recognitionResult.error ? (
                <p className="text-red-600">{recognitionResult.error}</p>
              ) : (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="font-medium">Person:</span>{' '}
                    <span className="font-semibold text-green-700">{recognitionResult.person}</span>
                  </div>
                  <div>
                    <span className="font-medium">Confidence:</span>{' '}
                    <span className="font-semibold text-green-700">
                      {(recognitionResult.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="font-medium">Status:</span>{' '}
                    <span className="font-semibold text-green-700">{recognitionResult.status}</span>
                  </div>
                  {recognitionResult.report_matches > 0 && (
                    <div className="col-span-2">
                      <span className="font-medium">Report Matches:</span>{' '}
                      <span className="font-semibold text-orange-700">
                        {recognitionResult.report_matches} active reports
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageRecognition;