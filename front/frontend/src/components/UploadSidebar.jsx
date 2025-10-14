import React from "react";
import { NavLink } from "react-router-dom";
import { FaImage, FaVideo, FaCamera } from "react-icons/fa";

const UploadSidebar = () => {
  const linkClass = ({ isActive }) =>
    `flex items-center gap-3 p-3 rounded hover:bg-gray-400 transition ${
      isActive ? "bg-blue-500 text-white" : ""
    }`;

  return (
    <div className="w-60 p-4 h-full min-h-screen">
      <h2 className="text-xl font-bold mb-4">Upload Options</h2>
      <nav className="flex flex-col gap-2">
        <NavLink to="/upload/images" className={linkClass}>
          <FaImage /> Multiple Images
        </NavLink>
        <NavLink to="/upload/videos" className={linkClass}>
          <FaVideo /> Video Upload
        </NavLink>
        <NavLink to="/upload/webcam" className={linkClass}>
          <FaCamera /> Webcam Capture
        </NavLink>
      </nav>
    </div>
  );
};

export default UploadSidebar;
