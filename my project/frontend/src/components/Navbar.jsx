import React from "react";

const Navbar = () => {
  return (
    <div className="w-full bg-white shadow-md py-3 px-6 flex justify-between items-center">
      <h2 className="text-xl font-semibold"></h2>
      <div className="flex gap-4">

        {/* Logout Button */}
        <button
          onClick={() => window.location.replace("/")}
          className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
        >
          Logout
        </button>
      </div>
    </div>
  );
};

export default Navbar;
