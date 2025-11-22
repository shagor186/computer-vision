import { useState } from "react";
import axiosClient from "../../utils/axiosClient";

const ImageUpload = () => {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);
    await axiosClient.post("/upload/file", formData, { headers: { "Content-Type": "multipart/form-data" } });
    alert("Image uploaded!");
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Upload Image</h2>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <button className="bg-blue-600 text-white px-4 py-2 mt-2" onClick={handleUpload}>Upload</button>
    </div>
  );
};

export default ImageUpload;
