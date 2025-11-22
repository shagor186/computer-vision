import React, { useState, useRef } from 'react';
import axios from 'axios';

const VideoRecognition = ({ onBack }) => {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [recognitionResults, setRecognitionResults] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleVideoSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedVideo(file);
      setRecognitionResults([]);
      setError('');
      
      // Create preview URL for the video
      const videoURL = URL.createObjectURL(file);
      if (videoRef.current) {
        videoRef.current.src = videoURL;
      }
    }
  };

  const processVideo = async () => {
    if (!selectedVideo) {
      setError('Please select a video first');
      return;
    }

    setProcessing(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      const formData = new FormData();
      formData.append('video', selectedVideo);

      const response = await axios.post('http://127.0.0.1:5000/api/face/recognize-video', formData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });

      setRecognitionResults(response.data.results || []);
    } catch (error) {
      setError(error.response?.data?.error || 'Video processing failed');
    } finally {
      setProcessing(false);
    }
  };

  const clearSelection = () => {
    setSelectedVideo(null);
    setRecognitionResults([]);
    setError('');
    if (videoRef.current) {
      videoRef.current.src = '';
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 py-8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <button
            onClick={onBack}
            className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-4"
          >
            ← Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Video Recognition</h1>
          <p className="text-gray-600">
            Upload a video file to recognize faces throughout the video
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          {/* Video Upload */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Video</h2>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleVideoSelect}
                accept=".mp4,.avi,.mov,.mkv,.webm"
                className="hidden"
                id="video-upload"
              />
              <label
                htmlFor="video-upload"
                className="cursor-pointer inline-flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                </svg>
                Select Video File
              </label>
              <p className="text-gray-500 mt-2">
                Supported formats: MP4, AVI, MOV, MKV, WEBM
              </p>
            </div>

            {/* Video Preview */}
            {selectedVideo && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Video Preview</h3>
                <div className="flex flex-col items-center">
                  <video
                    ref={videoRef}
                    controls
                    className="max-w-full h-64 rounded-lg border"
                  >
                    Your browser does not support the video tag.
                  </video>
                  <div className="mt-4 flex space-x-4">
                    <button
                      onClick={clearSelection}
                      className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-semibold rounded-lg transition-colors"
                    >
                      Clear Selection
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Processing Controls */}
          <div className="text-center mb-8">
            <button
              onClick={processVideo}
              disabled={!selectedVideo || processing}
              className={`px-8 py-4 text-white font-semibold rounded-lg text-lg transition-colors ${
                !selectedVideo || processing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {processing ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing Video...
                </span>
              ) : (
                'Process Video for Recognition'
              )}
            </button>
          </div>

          {/* Results */}
          {recognitionResults.length > 0 && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-green-800 mb-4">
                Recognition Results ({recognitionResults.length} faces found)
              </h3>
              <div className="space-y-4">
                {recognitionResults.map((result, index) => (
                  <div key={index} className="bg-white rounded-lg p-4 border">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="font-medium">Person:</span>{' '}
                        <span className="font-semibold text-green-700">{result.person}</span>
                      </div>
                      <div>
                        <span className="font-medium">Confidence:</span>{' '}
                        <span className="font-semibold text-green-700">
                          {(result.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="col-span-2">
                        <span className="font-medium">Timestamp:</span>{' '}
                        <span className="font-semibold text-gray-700">
                          {result.timestamp || 'N/A'}
                        </span>
                      </div>
                      {result.report_matches > 0 && (
                        <div className="col-span-2">
                          <span className="font-medium">Report Matches:</span>{' '}
                          <span className="font-semibold text-orange-700">
                            {result.report_matches} active reports
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}

          {/* Instructions */}
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-800 mb-2">How it works:</h3>
            <ul className="list-disc list-inside text-blue-700 text-sm space-y-1">
              <li>Upload a video file containing faces to recognize</li>
              <li>The system will process the video and extract faces from frames</li>
              <li>Each detected face will be matched against the trained database</li>
              <li>Results will show all recognized persons with confidence levels</li>
              <li>Matching reports will be highlighted for immediate attention</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoRecognition;