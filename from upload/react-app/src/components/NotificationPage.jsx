import React, { useEffect, useState } from "react";
import axios from "axios";

const NotificationPage = () => {
  const [people, setPeople] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:5500/people").then(res => setPeople(res.data));
  }, []);

  return (
    <div className="p-6">
      <h2 className="text-xl font-semibold mb-4">Notifications</h2>
      {people.map(p => (
        <div key={p.id} className="bg-gray-100 p-4 mb-2 rounded-xl shadow">
          <p><strong>ID:</strong> {p.id}</p>
          <p><strong>Name:</strong> {p.name}</p>
          <p><strong>Age:</strong> {p.age}</p>
          <p><strong>Location:</strong> {p.location}</p>
        </div>
      ))}
    </div>
  );
};

export default NotificationPage;
