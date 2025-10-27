import React from "react";
import { Link, useLocation } from "react-router-dom";
import { LayoutDashboard, FileText, Search, Bell, Settings } from "lucide-react";

const Sidebar = () => {
  const location = useLocation();
  const menu = [
    { name: "Dashboard", icon: <LayoutDashboard size={20} />, path: "/dashboard" },
    { name: "Report", icon: <FileText size={20} />, path: "/report" },
    { name: "Find", icon: <Search size={20} />, path: "/find" },
    { name: "Notification", icon: <Bell size={20} />, path: "/notification" },
    { name: "Settings", icon: <Settings size={20} />, path: "/settings" },
  ];

  return (
    <div className="w-64 bg-gray-800 text-white h-screen p-5 flex flex-col">
      <h1 className="text-2xl font-bold mb-8 text-center text-blue-400">
        MyApp
      </h1>
      <ul className="space-y-4">
        {menu.map((item) => (
          <li key={item.name}>
            <Link
              to={item.path}
              className={`flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition ${
                location.pathname === item.path ? "bg-gray-700" : ""
              }`}
            >
              {item.icon}
              <span>{item.name}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;
