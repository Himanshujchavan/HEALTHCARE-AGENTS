import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { loginUser } from "../services/authService";
import { HeartPulse } from "lucide-react";

function Login() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    username: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    setLoading(true);

    try {
      const res = await loginUser(form);

      localStorage.setItem("token", res.access_token);

      navigate("/dashboard");
    } catch (err) {
      alert("Invalid credentials");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-blue-500 to-indigo-600">
      <div className="bg-white p-10 rounded-2xl shadow-xl w-95">
        <div className="flex flex-col items-center mb-6">
          <HeartPulse size={40} className="text-blue-600 mb-2" />

          <h1 className="text-2xl font-bold text-gray-800">
            AI Health Companion
          </h1>

          <p className="text-sm text-gray-500">Smart health analysis system</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-sm text-gray-600">Username</label>

            <input
              type="text"
              placeholder="Enter username"
              className="border w-full p-2 mt-1 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              onChange={(e) => setForm({ ...form, username: e.target.value })}
            />
          </div>

          <div>
            <label className="text-sm text-gray-600">Password</label>

            <input
              type="password"
              placeholder="Enter password"
              className="border w-full p-2 mt-1 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              onChange={(e) => setForm({ ...form, password: e.target.value })}
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition"
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="text-sm text-center text-gray-500 mt-4">
          Don't have an account?
          <Link to="/register" className="text-blue-600 ml-1">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
