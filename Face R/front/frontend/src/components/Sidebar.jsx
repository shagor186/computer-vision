import React, { useState } from "react";
import { NavLink } from "react-router-dom";
import { FaTachometerAlt, FaUpload, FaBell, FaCog, FaChartBar } from "react-icons/fa";

const Sidebar = () => {
  const [collapsed, setCollapsed] = useState(false);

  const linkClass = ({ isActive }) =>
    `flex items-center gap-3 p-3 rounded hover:bg-gray-700 transition ${
      isActive ? "bg-blue-500 text-white" : "text-white"
    }`;

  return (
    <div
      className={`bg-gray-900 text-white h-screen p-4 transition-all duration-300 ${
        collapsed ? "w-20" : "w-60"
      }`}
    >
      <div className="flex justify-between items-center mb-8">
        {!collapsed && <h1 className="text-2xl font-bold">My App</h1>}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-white text-xl"
        >
          {collapsed ? "➤" : "⬅"}
        </button>
      </div>

      <nav className="flex flex-col gap-2">
        <NavLink to="/" className={linkClass}>
          <FaTachometerAlt /> {!collapsed && "Dashboard"}
        </NavLink>
        <NavLink to="/reports" className={linkClass}>
          <FaChartBar /> {!collapsed && "Reports"}
        </NavLink>
        <NavLink to="/upload" className={linkClass}>
          <FaUpload /> {!collapsed && "Upload"}
        </NavLink>
        <NavLink to="/notification" className={linkClass}>
          <FaBell /> {!collapsed && "Notification"}
        </NavLink>
        <NavLink to="/settings" className={linkClass}>
          <FaCog /> {!collapsed && "Settings"}
        </NavLink>
      </nav>
    </div>
  );
};

export default Sidebar;
