import React from "react";
import { FaBell, FaUserCircle, FaSearch } from "react-icons/fa";

const TopNavbar = () => {
  return (
    <div className="flex justify-between items-center p-4 bg-white shadow-md">
      <div className="flex items-center gap-2">
        <FaSearch className="text-gray-500" />
        <input
          type="text"
          placeholder="Search..."
          className="border rounded px-2 py-1 focus:outline-none"
        />
      </div>
      <div className="flex items-center gap-4">
        <FaBell className="text-gray-600 text-xl cursor-pointer" />
        <FaUserCircle className="text-gray-600 text-3xl cursor-pointer" />
      </div>
    </div>
  );
};

export default TopNavbar;
