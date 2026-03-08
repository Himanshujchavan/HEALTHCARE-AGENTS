import { BrowserRouter, Routes, Route } from "react-router-dom";

import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import SubmitHealth from "./pages/SubmitHealth";
// import Analysis from "./pages/Analysis";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/submit" element={<SubmitHealth />} />
        {/* <Route path="/analysis/:id" element={<Analysis />} /> */}
      </Routes>
    </BrowserRouter>
  );
}

export default App;
