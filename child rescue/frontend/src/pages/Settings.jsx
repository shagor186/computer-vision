import { useState, useEffect } from "react";
import axiosClient from "../utils/axiosClient";

const Settings = () => {
  const [user, setUser] = useState({ username: "", email: "" });

  useEffect(() => {
    // Example: fetch user info
    const storedUser = JSON.parse(localStorage.getItem("user"));
    if (storedUser) setUser(storedUser);
  }, []);

  const handleUpdate = async () => {
    // Example: call backend update API
    alert("Settings updated!");
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Settings</h2>
      <input placeholder="Username" className="border p-2 mb-2" value={user.username} onChange={e => setUser({...user, username: e.target.value})}/>
      <input placeholder="Email" className="border p-2 mb-2" value={user.email} onChange={e => setUser({...user, email: e.target.value})}/>
      <button className="bg-blue-600 text-white px-4 py-2" onClick={handleUpdate}>Update</button>
    </div>
  );
};

export default Settings;
