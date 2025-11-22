import React from "react";
import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";

const Settings = () => {
  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <div className="flex-1 flex flex-col">
        {/* Navbar with Logout button */}
        <Navbar>
          <button
            onClick={() => window.location.replace("/")}
            className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
          >
            Logout
          </button>
        </Navbar>

        {/* Settings content */}
        <div className="p-6">
          <h1 className="text-3xl font-bold text-gray-700 mb-4">Settings</h1>
          <p className="text-gray-600">
            Manage your account, password, and preferences here.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Settings;
