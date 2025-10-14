import React, { useState } from "react";

const ImageUpload = () => {
  const [images, setImages] = useState([]);

  const handleFiles = (files) => {
    setImages([...images, ...Array.from(files)]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const handleDelete = (index) => {
    const newImages = [...images];
    newImages.splice(index, 1);
    setImages(newImages);
  };

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Multiple Image Upload</h2>

      <div
        className="border-4 border-dashed border-gray-400 p-10 text-center mb-4 rounded hover:border-blue-500 transition"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        Drag & Drop Images Here or
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={(e) => handleFiles(e.target.files)}
          className="ml-2"
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {images.map((file, idx) => (
          <div key={idx} className="relative border rounded shadow hover:shadow-lg transition">
            <img
              src={URL.createObjectURL(file)}
              alt={file.name}
              className="w-full h-48 object-cover rounded"
            />
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

export default ImageUpload;
