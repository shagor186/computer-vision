import React, { useState } from "react";
import axios from "axios";

const VideoUpload = () => {
  const [videos, setVideos] = useState([]);
  const [uploaded, setUploaded] = useState([]);

  const handleFileChange = (e) => {
    setVideos(e.target.files);
  };

  const handleUpload = async () => {
    if (videos.length === 0) {
      alert("Please select videos first!");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < videos.length; i++) {
      formData.append("videos", videos[i]);
    }

    try {
      const res = await axios.post("http://127.0.0.1:5500/upload_videos", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploaded(res.data.files);
      alert("Videos uploaded successfully!");
    } catch (error) {
      console.error(error);
      alert("Upload failed!");
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <h1 className="text-2xl font-bold mb-4 text-gray-700">🎥 Multiple Video Upload</h1>
      <input
        type="file"
        multiple
        accept="video/*"
        onChange={handleFileChange}
        className="mb-4 border border-gray-300 p-2 rounded-lg"
      />
      <button
        onClick={handleUpload}
        className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        Upload Videos
      </button>

      {uploaded.length > 0 && (
        <div className="mt-6 w-full max-w-2xl">
          <h2 className="text-lg font-semibold text-green-600 mb-2">Uploaded Videos:</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {uploaded.map((file, index) => (
              <video
                key={index}
                controls
                className="w-full rounded-lg shadow"
                src={`http://127.0.0.1:5500/videos/${file}`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
