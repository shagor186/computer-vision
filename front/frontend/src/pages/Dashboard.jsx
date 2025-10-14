import React from "react";
import TopNavbar from "../components/TopNavbar";

const Dashboard = () => {
  return (
    <div>
      <TopNavbar />

      <div className="p-6">
        <h2 className="text-3xl font-bold mb-6">Dashboard</h2>

        {/* Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="p-4 bg-blue-500 text-white rounded shadow">Users: 120</div>
          <div className="p-4 bg-green-500 text-white rounded shadow">Orders: 75</div>
          <div className="p-4 bg-yellow-500 text-white rounded shadow">Revenue: $5000</div>
          <div className="p-4 bg-red-500 text-white rounded shadow">Alerts: 3</div>
        </div>

        {/* Placeholder for Charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white p-6 rounded shadow h-64 flex items-center justify-center">
            Chart 1
          </div>
          <div className="bg-white p-6 rounded shadow h-64 flex items-center justify-center">
            Chart 2
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
