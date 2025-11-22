import React, { useRef, useState } from "react";
import Webcam from "react-webcam";

const WebcamCapture = () => {
  const webcamRef = useRef(null);
  const [images, setImages] = useState([]);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) setImages([imageSrc, ...images]);
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-4">Webcam Capture</h2>

      {/* Live Camera */}
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: "user" }}
        className="w-full max-w-lg rounded border"
      />

      <button
        onClick={capture}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Capture
      </button>

      {/* Captured Images */}
      <div className="grid grid-cols-3 gap-4 mt-4">
        {images.map((img, idx) => (
          <div key={idx} className="relative">
            <img src={img} alt={`capture-${idx}`} className="w-full h-40 object-cover rounded" />
            <button
              onClick={() => setImages(images.filter((_, i) => i !== idx))}
              className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded"
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default WebcamCapture;
