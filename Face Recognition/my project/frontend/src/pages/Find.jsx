import React, { useState } from "react";
import Sidebar from "../components/Sidebar";

const Find = () => {
  const [activeTab, setActiveTab] = useState(null);

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col bg-gray-100">
        <div className="p-6 space-y-4">
          <h1 className="text-3xl font-bold text-gray-700 mb-4">Find</h1>

          {/* Rows / Tabs */}
          <div className="grid grid-cols-3 gap-4">
            <div
              onClick={() => setActiveTab(activeTab === "image" ? null : "image")}
              className="cursor-pointer bg-white p-4 rounded-lg shadow-md hover:bg-blue-50 text-center font-semibold"
            >
              Upload Image
            </div>
            <div
              onClick={() => setActiveTab(activeTab === "video" ? null : "video")}
              className="cursor-pointer bg-white p-4 rounded-lg shadow-md hover:bg-blue-50 text-center font-semibold"
            >
              Upload Video
            </div>
            <div
              onClick={() => setActiveTab(activeTab === "webcam" ? null : "webcam")}
              className="cursor-pointer bg-white p-4 rounded-lg shadow-md hover:bg-blue-50 text-center font-semibold"
            >
              Open Webcam
            </div>
          </div>

          {/* Dynamic Content */}
          <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
            {activeTab === "image" && (
              <div>
                <h2 className="text-xl font-semibold mb-2">Upload Image</h2>
                <input type="file" accept="image/*" className="border p-2 rounded w-full" />
                <button className="mt-3 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                  Submit Image
                </button>
              </div>
            )}

            {activeTab === "video" && (
              <div>
                <h2 className="text-xl font-semibold mb-2">Upload Video</h2>
                <input type="file" accept="video/*" className="border p-2 rounded w-full" />
                <button className="mt-3 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                  Submit Video
                </button>
              </div>
            )}

            {activeTab === "webcam" && (
              <div>
                <h2 className="text-xl font-semibold mb-2">Open Webcam</h2>
                <video
                  className="border w-full rounded"
                  autoPlay
                  playsInline
                  muted
                  id="webcamVideo"
                ></video>
                <button
                  onClick={() => {
                    const video = document.getElementById("webcamVideo");
                    if (navigator.mediaDevices.getUserMedia) {
                      navigator.mediaDevices
                        .getUserMedia({ video: true })
                        .then((stream) => (video.srcObject = stream))
                        .catch((err) => console.error("Error opening webcam:", err));
                    }
                  }}
                  className="mt-3 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                >
                  Start Webcam
                </button>
              </div>
            )}

            {!activeTab && <p className="text-gray-500">Click a tab above to start.</p>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Find;
