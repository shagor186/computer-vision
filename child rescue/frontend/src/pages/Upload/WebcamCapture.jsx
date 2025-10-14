import { useRef } from "react";
import axiosClient from "../../utils/axiosClient";

const WebcamCapture = () => {
  const videoRef = useRef();

  const startWebcam = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    videoRef.current.play();
  };

  const captureFrame = async () => {
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async blob => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");
      await axiosClient.post("/upload/webcam", formData);
      alert("Frame uploaded!");
    }, "image/jpeg");
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Webcam Capture</h2>
      <video ref={videoRef} className="border w-96 h-64 mb-2"></video>
      <div>
        <button onClick={startWebcam} className="bg-green-600 text-white px-4 py-2 mr-2">Start Webcam</button>
        <button onClick={captureFrame} className="bg-blue-600 text-white px-4 py-2">Capture & Upload</button>
      </div>
    </div>
  );
};

export default WebcamCapture;
