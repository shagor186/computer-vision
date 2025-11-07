import { NavLink } from "react-router-dom";

const Sidebar = () => {
  const links = [
    { name: "Dashboard", path: "/" },
    { name: "Report", path: "/report" },
    { name: "Upload", path: "/upload" },
    { name: "Matches", path: "/matches" },
    { name: "Notifications", path: "/notifications" },
    { name: "Settings", path: "/settings" },
  ];

  return (
    <div className="w-64 bg-white shadow-md h-full p-4">
      <h2 className="text-xl font-bold mb-6">Child Rescue AI</h2>
      <ul>
        {links.map((link) => (
          <li key={link.name} className="mb-3">
            <NavLink
              to={link.path}
              className={({ isActive }) =>
                isActive ? "font-semibold text-blue-600" : "text-gray-700"
              }
            >
              {link.name}
            </NavLink>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;
