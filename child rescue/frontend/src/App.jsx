import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";

import Dashboard from "./pages/Dashboard";
import Report from "./pages/Report";
import Upload from "./pages/Upload/ImageUpload";
import Matches from "./pages/Matches";
import Notification from "./pages/Notification";
import Settings from "./pages/Settings";

function App() {
  return (
    <Router>
      <div className="flex h-screen">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Navbar />
          <div className="p-6 flex-1 overflow-auto bg-gray-50">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/report" element={<Report />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/matches" element={<Matches />} />
              <Route path="/notifications" element={<Notification />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;
