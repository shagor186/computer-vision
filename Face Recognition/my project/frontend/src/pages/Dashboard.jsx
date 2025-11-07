import React from "react";
import Sidebar from "../components/Sidebar";

const Dashboard = () => {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col bg-gray-100">
        <div className="p-6">
          <h1 className="text-3xl font-bold text-gray-700 mb-4">
            Dashboard Overview
          </h1>
          <p className="text-gray-600">
            This is your main dashboard area. You can view reports, find data, and get notifications here.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
