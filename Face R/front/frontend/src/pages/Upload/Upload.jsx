import React, { useState } from "react";
import { Routes, Route } from "react-router-dom";
import UploadSidebar from "../../components/UploadSidebar";
import ImageUpload from "./ImageUpload";
import VideoUpload from "./VideoUpload";
import WebcamCapture from "./WebcamCapture";

const Upload = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-screen">
      {/* Mobile toggle */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="md:hidden absolute top-4 left-4 z-50 px-3 py-2 bg-blue-500 text-white rounded"
      >
        {sidebarOpen ? "Close" : "Menu"}
      </button>

      {/* Sidebar */}
      <div
        className={`fixed md:relative z-40 transition-transform transform ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0 w-60 bg-gray-200 min-h-screen`}
      >
        <UploadSidebar />
      </div>

      {/* Main content */}
      <div className="flex-1 p-6 md:ml-60 bg-gray-100">
        <Routes>
          <Route path="images" element={<ImageUpload />} />
          <Route path="videos" element={<VideoUpload />} />
          <Route path="webcam" element={<WebcamCapture />} />
          <Route
            path="/"
            element={
              <div className="text-2xl font-semibold">
                Select an option from sidebar
              </div>
            }
          />
        </Routes>
      </div>
    </div>
  );
};

export default Upload;
