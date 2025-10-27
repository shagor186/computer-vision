import React, { useEffect, useState } from "react";
import axios from "axios";

const CardPage = () => {
  const [people, setPeople] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:5500/people").then(res => setPeople(res.data));
  }, []);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 p-6">
      {people.map(p => (
        <div key={p.id} className="bg-white rounded-2xl shadow-lg p-4 text-center">
          <img
            src={`http://127.0.0.1:5500/static/images/${p.image}`}
            alt={p.name}
            className="w-48 h-48 object-cover rounded-xl mb-3 mx-auto"
          />
          <h3 className="font-bold text-lg">{p.name}</h3>
          <p className="text-gray-600 text-sm">ID: {p.id}</p>
          <p className="text-gray-600 text-sm">📍 {p.location}</p>
        </div>
      ))}
    </div>
  );
};

export default CardPage;
