import { useEffect, useState } from "react";
import axiosClient from "../utils/axiosClient";
import StatCard from "../components/StatCard";

const Dashboard = () => {
  const [stats, setStats] = useState({ total: 0, missing: 0, found: 0 });

  useEffect(() => {
    axiosClient.get("/report/stats").then((res) => setStats(res.data));
  }, []);

  return (
    <div className="grid grid-cols-3 gap-4">
      <StatCard title="Total Persons" value={stats.total} />
      <StatCard title="Missing" value={stats.missing} />
      <StatCard title="Found" value={stats.found} />
    </div>
  );
};

export default Dashboard;
