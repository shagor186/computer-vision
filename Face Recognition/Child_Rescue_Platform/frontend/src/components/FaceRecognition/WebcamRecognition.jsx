// import React, { useState, useRef, useEffect } from 'react';

// const WebcamRecognition = ({ onBack }) => {
//   const [isWebcamOn, setIsWebcamOn] = useState(false);
//   const [recognitionResult, setRecognitionResult] = useState(null);
//   const videoRef = useRef(null);
//   const canvasRef = useRef(null);
//   const streamRef = useRef(null);

//   const startWebcam = async () => {
//     try {
//       const stream = await navigator.mediaDevices.getUserMedia({ 
//         video: { width: 640, height: 480 } 
//       });
//       if (videoRef.current) {
//         videoRef.current.srcObject = stream;
//         streamRef.current = stream;
//       }
//       setIsWebcamOn(true);
//     } catch (error) {
//       console.error('Error accessing webcam:', error);
//       alert('Cannot access webcam. Please check permissions.');
//     }
//   };

//   const stopWebcam = () => {
//     if (streamRef.current) {
//       streamRef.current.getTracks().forEach(track => track.stop());
//       streamRef.current = null;
//     }
//     setIsWebcamOn(false);
//     setRecognitionResult(null);
//   };

//   const captureAndRecognize = () => {
//     if (!videoRef.current || !canvasRef.current) return;

//     const video = videoRef.current;
//     const canvas = canvasRef.current;
//     const context = canvas.getContext('2d');

//     // Draw current video frame to canvas
//     context.drawImage(video, 0, 0, canvas.width, canvas.height);

//     // Convert canvas to blob and send for recognition
//     canvas.toBlob(async (blob) => {
//       try {
//         const token = localStorage.getItem('token');
//         const formData = new FormData();
//         formData.append('image', blob, 'webcam-capture.jpg');

//         const response = await fetch('http://127.0.0.1:5000/api/face/recognize-image', {
//           method: 'POST',
//           headers: {
//             'Authorization': `Bearer ${token}`
//           },
//           body: formData
//         });

//         const result = await response.json();
//         setRecognitionResult(result);
//       } catch (error) {
//         console.error('Recognition error:', error);
//         setRecognitionResult({ error: 'Recognition failed' });
//       }
//     }, 'image/jpeg');
//   };

//   useEffect(() => {
//     return () => {
//       if (streamRef.current) {
//         streamRef.current.getTracks().forEach(track => track.stop());
//       }
//     };
//   }, []);

//   return (
//     <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 py-8">
//       <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
//         {/* Header */}
//         <div className="text-center mb-8">
//           <button
//             onClick={onBack}
//             className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-4"
//           >
//             ← Back to Dashboard
//           </button>
//           <h1 className="text-3xl font-bold text-gray-900 mb-2">Webcam Recognition</h1>
//           <p className="text-gray-600">
//             Real-time face recognition using your webcam
//           </p>
//         </div>

//         <div className="bg-white rounded-2xl shadow-lg p-8">
//           <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
//             {/* Webcam Section */}
//             <div>
//               <h2 className="text-xl font-semibold text-gray-900 mb-4">Webcam Feed</h2>
//               <div className="border-2 border-dashed border-gray-300 rounded-lg p-4">
//                 <video
//                   ref={videoRef}
//                   autoPlay
//                   playsInline
//                   muted
//                   className="w-full h-64 bg-gray-200 rounded-lg object-cover"
//                 />
//                 <canvas
//                   ref={canvasRef}
//                   width="640"
//                   height="480"
//                   className="hidden"
//                 />
//               </div>

//               {/* Controls */}
//               <div className="mt-6 flex justify-center space-x-4">
//                 {!isWebcamOn ? (
//                   <button
//                     onClick={startWebcam}
//                     className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
//                   >
//                     Start Webcam
//                   </button>
//                 ) : (
//                   <>
//                     <button
//                       onClick={captureAndRecognize}
//                       className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
//                     >
//                       Capture & Recognize
//                     </button>
//                     <button
//                       onClick={stopWebcam}
//                       className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-lg transition-colors"
//                     >
//                       Stop Webcam
//                     </button>
//                   </>
//                 )}
//               </div>
//             </div>

//             {/* Results Section */}
//             <div>
//               <h2 className="text-xl font-semibold text-gray-900 mb-4">Recognition Results</h2>
//               {recognitionResult ? (
//                 <div className={`p-6 rounded-lg ${
//                   recognitionResult.error 
//                     ? 'bg-red-50 border border-red-200' 
//                     : 'bg-green-50 border border-green-200'
//                 }`}>
//                   {recognitionResult.error ? (
//                     <div>
//                       <h3 className="text-lg font-semibold text-red-800 mb-2">Error</h3>
//                       <p className="text-red-700">{recognitionResult.error}</p>
//                     </div>
//                   ) : (
//                     <div>
//                       <h3 className="text-lg font-semibold text-green-800 mb-4">Person Identified</h3>
//                       <div className="space-y-3">
//                         <div>
//                           <span className="font-medium">Name:</span>{' '}
//                           <span className="font-semibold text-green-700">{recognitionResult.person}</span>
//                         </div>
//                         <div>
//                           <span className="font-medium">Confidence:</span>{' '}
//                           <span className="font-semibold text-green-700">
//                             {(recognitionResult.confidence * 100).toFixed(2)}%
//                           </span>
//                         </div>
//                         {recognitionResult.report_matches > 0 && (
//                           <div className="bg-orange-50 border border-orange-200 rounded p-3">
//                             <span className="font-medium text-orange-800">
//                               ⚠️ {recognitionResult.report_matches} active reports found for this person
//                             </span>
//                           </div>
//                         )}
//                       </div>
//                     </div>
//                   )}
//                 </div>
//               ) : (
//                 <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
//                   <p className="text-gray-500">No recognition results yet.</p>
//                   <p className="text-sm text-gray-400 mt-2">
//                     Start webcam and capture an image to see results here.
//                   </p>
//                 </div>
//               )}

//               {/* Instructions */}
//               <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
//                 <h3 className="font-semibold text-blue-800 mb-2">How to use:</h3>
//                 <ol className="list-decimal list-inside text-blue-700 text-sm space-y-1">
//                   <li>Click "Start Webcam" to enable your camera</li>
//                   <li>Position your face clearly in the frame</li>
//                   <li>Click "Capture & Recognize" to identify the person</li>
//                   <li>View results in this panel</li>
//                 </ol>
//               </div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default WebcamRecognition;



import React, { useState, useRef, useEffect } from 'react';
import { Camera, Square, Play } from 'lucide-react';

const WebcamComponent = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState('');
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    // Cleanup on component unmount
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startWebcam = async () => {
    try {
      setError('');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user' // Use front camera
        } 
      });
      
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setIsRecording(true);
    } catch (err) {
      setError('Cannot access webcam: ' + err.message);
      console.error('Webcam error:', err);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsRecording(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to data URL and create download
    const dataUrl = canvas.toDataURL('image/jpeg');
    downloadImage(dataUrl);
  };

  const downloadImage = (dataUrl) => {
    const link = document.createElement('a');
    link.download = `webcam-capture-${new Date().getTime()}.jpg`;
    link.href = dataUrl;
    link.click();
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center mb-6">
          <Camera className="h-6 w-6 text-blue-600 mr-2" />
          <h1 className="text-2xl font-bold text-gray-900">Webcam Component</h1>
        </div>

        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6">
          {/* Webcam Feed */}
          <div>
            <div className="bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-64 md:h-96 object-cover"
              />
            </div>

            <div className="flex justify-center space-x-4 mt-4">
              {!isRecording ? (
                <button
                  onClick={startWebcam}
                  className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 flex items-center transition duration-200"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Webcam
                </button>
              ) : (
                <>
                  <button
                    onClick={capturePhoto}
                    className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 flex items-center transition duration-200"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Capture Photo
                  </button>
                  <button
                    onClick={stopWebcam}
                    className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 flex items-center transition duration-200"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    Stop Webcam
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Instructions */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Webcam Instructions
            </h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• Click "Start Webcam" to begin camera feed</li>
              <li>• Click "Capture Photo" to take a picture</li>
              <li>• Click "Stop Webcam" to turn off camera</li>
              <li>• Allow camera permissions when prompted</li>
              <li>• Ensure good lighting for better quality</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebcamComponent;