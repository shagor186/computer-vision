import React from "react";
import Sidebar from "../components/Sidebar";

const Notification = () => {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col bg-gray-100">
        <div className="p-6">
          <h1 className="text-3xl font-bold text-gray-700 mb-4">Notifications</h1>
          <p className="text-gray-600">All system alerts and user notifications appear here.</p>
        </div>
      </div>
    </div>
  );
};

export default Notification;
