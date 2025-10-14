import React, { useState } from "react";

const VideoUpload = () => {
  const [videos, setVideos] = useState([]);

  const handleFiles = (files) => {
    setVideos([...videos, ...Array.from(files)]);
  };

  const handleDelete = (index) => {
    const newVideos = [...videos];
    newVideos.splice(index, 1);
    setVideos(newVideos);
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Video Upload</h2>

      <input
        type="file"
        accept="video/*"
        multiple
        onChange={(e) => handleFiles(e.target.files)}
        className="mb-4"
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {videos.map((file, idx) => (
          <div key={idx} className="relative border rounded shadow hover:shadow-lg transition">
            <video controls className="w-full h-48 rounded">
              <source src={URL.createObjectURL(file)} type={file.type} />
            </video>
            <button
              onClick={() => handleDelete(idx)}
              className="absolute top-2 right-2 bg-red-500 text-white rounded px-2 py-1"
            >
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default VideoUpload;
