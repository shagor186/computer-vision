import { useEffect, useState } from "react";
import axiosClient from "../utils/axiosClient";

const Notification = () => {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    axiosClient.get("/report/notifications").then(res => setNotifications(res.data.notifications));
  }, []);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">Notifications</h2>
      {notifications.length === 0 ? <p>No notifications</p> : notifications.map((n,i) => (
        <div key={i} className="border p-2 mb-2">{n.message}</div>
      ))}
    </div>
  );
};

export default Notification;
