import React, { useState } from "react";
import Sidebar from "../components/Sidebar";

const Report = () => {
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [location, setLocation] = useState("");
  const [images, setImages] = useState([]);

  const handleImageChange = (e) => {
    setImages([...e.target.files]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!id || !name || !age || !location || images.length === 0) {
      alert("Please fill all fields and select images");
      return;
    }

    const formData = new FormData();
    formData.append("id", id);
    formData.append("name", name);
    formData.append("age", age);
    formData.append("location", location);

    // Multiple images append
    images.forEach((img) => {
      formData.append("images[]", img);
    });

    console.log("FormData ready to submit:", formData);

    // পরে: axios.post("http://127.0.0.1:5000/report", formData)

    alert("Report submitted successfully (UI only)!");
    setId("");
    setName("");
    setAge("");
    setLocation("");
    setImages([]);
    e.target.reset();
  };

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col bg-gray-100">
        <div className="p-6">
          <h1 className="text-3xl font-bold text-gray-700 mb-6">Add New Report</h1>

          <form
            onSubmit={handleSubmit}
            className="bg-white p-6 rounded-lg shadow-md space-y-4 max-w-lg"
          >
            <div>
              <label className="block font-medium text-gray-700 mb-1">ID</label>
              <input
                type="text"
                value={id}
                onChange={(e) => setId(e.target.value)}
                className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                required
              />
            </div>

            <div>
              <label className="block font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                required
              />
            </div>

            <div>
              <label className="block font-medium text-gray-700 mb-1">Age</label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                required
              />
            </div>

            <div>
              <label className="block font-medium text-gray-700 mb-1">Location</label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full border rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400"
                required
              />
            </div>

            <div>
              <label className="block font-medium text-gray-700 mb-1">Upload Images</label>
              <input
                type="file"
                multiple
                accept="image/*"
                onChange={handleImageChange}
                className="w-full"
                required
              />
            </div>

            <button
              type="submit"
              className="w-full bg-green-500 hover:bg-green-600 text-white py-2 rounded-lg font-semibold transition"
            >
              Submit Report
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Report;
