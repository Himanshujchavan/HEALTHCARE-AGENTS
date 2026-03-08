import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { registerUser } from "../services/authService";
import { HeartPulse, ShieldCheck, Activity } from "lucide-react";

function Register() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);

    try {
      await registerUser({
        username: form.username,
        email: form.email,
        password: form.password,
      });

      navigate("/");
    } catch (err) {
      setError("Registration failed");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen grid md:grid-cols-2">
      {/* LEFT PANEL */}

      <div className="hidden md:flex flex-col justify-center bg-gradient-to-br from-blue-600 to-indigo-700 text-white p-12">
        <div className="flex items-center mb-6">
          <HeartPulse size={36} className="mr-3" />

          <h1 className="text-3xl font-bold">AI Health Companion</h1>
        </div>

        <p className="text-lg mb-10 opacity-90">
          Advanced AI-powered health analysis platform. Monitor your health data
          and receive intelligent insights.
        </p>

        <div className="space-y-6">
          <div className="flex items-center">
            <Activity className="mr-3" />

            <span>AI-powered health analysis</span>
          </div>

          <div className="flex items-center">
            <ShieldCheck className="mr-3" />

            <span>Secure medical data protection</span>
          </div>

          <div className="flex items-center">
            <HeartPulse className="mr-3" />

            <span>Personalized health insights</span>
          </div>
        </div>
      </div>

      {/* RIGHT PANEL */}

      <div className="flex items-center justify-center bg-gray-50">
        <div className="bg-white p-10 rounded-2xl shadow-xl w-[420px]">
          <h2 className="text-2xl font-bold mb-2">Create Account</h2>

          <p className="text-gray-500 mb-6">
            Start monitoring your health with AI insights
          </p>

          {error && (
            <div className="bg-red-100 text-red-600 p-2 rounded mb-4 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-sm text-gray-600">Username</label>

              <input
                type="text"
                className="border w-full p-2 rounded-lg mt-1 focus:ring-2 focus:ring-blue-500"
                onChange={(e) => setForm({ ...form, username: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm text-gray-600">Email</label>

              <input
                type="email"
                className="border w-full p-2 rounded-lg mt-1 focus:ring-2 focus:ring-blue-500"
                onChange={(e) => setForm({ ...form, email: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm text-gray-600">Password</label>

              <input
                type="password"
                className="border w-full p-2 rounded-lg mt-1 focus:ring-2 focus:ring-blue-500"
                onChange={(e) => setForm({ ...form, password: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm text-gray-600">Confirm Password</label>

              <input
                type="password"
                className="border w-full p-2 rounded-lg mt-1 focus:ring-2 focus:ring-blue-500"
                onChange={(e) =>
                  setForm({ ...form, confirmPassword: e.target.value })
                }
              />
            </div>

            <button
              type="submit"
              className="w-full bg-blue-600 text-white p-2 rounded-lg hover:bg-blue-700 transition"
            >
              {loading ? "Creating Account..." : "Create Account"}
            </button>
          </form>

          <p className="text-sm text-center text-gray-500 mt-6">
            Already have an account?
            <Link to="/" className="text-blue-600 ml-1">
              Login
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Register;
