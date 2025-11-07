import { useEffect, useState } from "react";
import axiosClient from "../utils/axiosClient";

const Matches = () => {
  const [matches, setMatches] = useState([]);

  useEffect(() => {
    axiosClient.get("/face/matches").then(res => setMatches(res.data.matches));
  }, []);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Matched Persons</h2>
      {matches.length === 0 ? <p>No matches found</p> : matches.map((m,i) => (
        <div key={i} className="border p-2 mb-2">{m.name}</div>
      ))}
    </div>
  );
};

export default Matches;
