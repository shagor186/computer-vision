import React, { useState } from "react";
import { Link } from "react-router-dom";

const ForgotPassword = () => {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");

  const handleReset = (e) => {
    e.preventDefault();

    if (!email) {
      setMessage("Please enter your email address.");
      return;
    }

    console.log("Password reset request for:", email);
    // Flask backend এ reset request যাবে পরে

    setMessage("If this email exists, password reset link sent!");
    setEmail("");
  };

  return (
    <div className="w-full max-w-md p-8 bg-white rounded-2xl shadow-lg">
      <h2 className="text-2xl font-bold text-center mb-6">Forgot Password</h2>
      <form onSubmit={handleReset} className="space-y-5">
        <div>
          <label className="block text-gray-700 font-medium mb-2">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded-lg font-semibold hover:bg-blue-600 transition"
        >
          Send Reset Link
        </button>
      </form>

      {message && (
        <p className="text-center text-sm text-green-600 mt-4">{message}</p>
      )}

      <p className="text-sm text-center mt-6">
        <Link to="/" className="text-blue-600 font-medium hover:underline">
          Back to Login
        </Link>
      </p>
    </div>
  );
};

export default ForgotPassword;
