import { useState } from "react";
import axios from "axios";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();

    const res = await axios.post("http://127.0.0.1:5500/api/login", {
      email,
      password
    });

    setMsg(res.data.msg);
  };

  return (
    <div>
      <h2>Login</h2>

      <form onSubmit={handleLogin}>
        <input 
          type="email" 
          placeholder="Email"
          onChange={(e)=> setEmail(e.target.value)}
        />

        <input 
          type="password" 
          placeholder="Password"
          onChange={(e)=> setPassword(e.target.value)}
        />

        <button type="submit">Login</button>
      </form>

      <p>{msg}</p>
    </div>
  );
}
