import React, { useRef } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const WebcamComponent = () => {
  const webcamRef = useRef(null);

  const captureSnapshot = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const res = await axios.post("http://127.0.0.1:5500/save_snapshot", { image: imageSrc });
      alert("Snapshot saved: " + res.data.filename);
    } catch (error) {
      console.error(error);
      alert("Error saving snapshot");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/png"
        className="rounded-lg border-4 border-gray-300"
      />
      <button
        onClick={captureSnapshot}
        className="mt-4 px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Capture
      </button>
    </div>
  );
};

export default WebcamComponent;
