// import React, { useState } from "react";
// import { submitPersonData } from "../api/api";

// function UploadForm() {
//   const [formData, setFormData] = useState({
//     id: "",
//     name: "",
//     age: "",
//     location: "",
//     images: []
//   });

//   const handleChange = (e) => {
//     if (e.target.name === "images") {
//       setFormData({ ...formData, images: e.target.files });
//     } else {
//       setFormData({ ...formData, [e.target.name]: e.target.value });
//     }
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     const data = new FormData();
//     data.append("id", formData.id);
//     data.append("name", formData.name);
//     data.append("age", formData.age);
//     data.append("location", formData.location);
//     for (let i = 0; i < formData.images.length; i++) {
//       data.append("images", formData.images[i]);
//     }

//     try {
//       const res = await submitPersonData(data);
//       alert(res.message);
//     } catch (err) {
//       alert("Error submitting form");
//       console.error(err);
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gray-100 flex items-center justify-center">
//       <form
//         onSubmit={handleSubmit}
//         className="bg-white p-6 rounded shadow-md w-96"
//       >
//         <h2 className="text-2xl mb-4 text-center">Person Info Upload</h2>

//         <input
//           type="text"
//           name="id"
//           placeholder="ID"
//           onChange={handleChange}
//           required
//           className="w-full p-2 mb-2 border rounded"
//         />

//         <input
//           type="text"
//           name="name"
//           placeholder="Name"
//           onChange={handleChange}
//           required
//           className="w-full p-2 mb-2 border rounded"
//         />

//         <input
//           type="number"
//           name="age"
//           placeholder="Age"
//           onChange={handleChange}
//           required
//           className="w-full p-2 mb-2 border rounded"
//         />

//         <input
//           type="text"
//           name="location"
//           placeholder="Location"
//           onChange={handleChange}
//           required
//           className="w-full p-2 mb-2 border rounded"
//         />

//         <input
//           type="file"
//           name="images"
//           multiple
//           onChange={handleChange}
//           required
//           className="w-full mb-2"
//         />

//         <button
//           type="submit"
//           className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
//         >
//           Submit
//         </button>
//       </form>
//     </div>
//   );
// }

// export default UploadForm;






import React, { useState } from "react";
import axios from "axios";

const UploadForm = () => {
  const [formData, setFormData] = useState({
    id: "",
    name: "",
    age: "",
    location: "",
    image: null,
  });

  const handleChange = (e) => {
    if (e.target.name === "image") {
      setFormData({ ...formData, image: e.target.files[0] });
    } else {
      setFormData({ ...formData, [e.target.name]: e.target.value });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const form = new FormData();
    for (const key in formData) form.append(key, formData[key]);

    await axios.post("http://127.0.0.1:5500/upload", form);
    alert("✅ Form submitted successfully!");
  };

  return (
    <div className="max-w-md mx-auto bg-white p-6 rounded-2xl shadow-lg mt-10">
      <h2 className="text-xl font-semibold mb-4 text-center">Person Form</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input name="id" onChange={handleChange} placeholder="Unique ID" className="border p-2 w-full rounded" required />
        <input name="name" onChange={handleChange} placeholder="Name" className="border p-2 w-full rounded" required />
        <input name="age" onChange={handleChange} type="number" placeholder="Age" className="border p-2 w-full rounded" required />
        <input name="location" onChange={handleChange} placeholder="Location" className="border p-2 w-full rounded" required />
        <input name="image" onChange={handleChange} type="file" className="border p-2 w-full rounded" required />
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded w-full">Submit</button>
      </form>
    </div>
  );
};

export default UploadForm;
