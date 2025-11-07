import React, { useState } from "react";

const NotificationIcon = () => {
  const [count, setCount] = useState(0); // initial notification count

  // submit function
  const handleSubmit = () => {
    // নতুন notification যুক্ত করতে
    setCount(prev => prev + 1);
  };

  return (
    <div>
      {/* Notification Icon */}
      <div className="relative inline-block text-3xl text-gray-700">
        🔔
        {count > 0 && (
          <span className="absolute -top-2 -right-2 bg-red-600 text-white text-xs font-bold w-5 h-5 flex items-center justify-center rounded-full">
            {count > 9 ? "9+" : count}
          </span>
        )}
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Submit
      </button>
    </div>
  );
};

export default NotificationIcon;
