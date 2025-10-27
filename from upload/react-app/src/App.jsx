// import React from "react";
// import UploadForm from "./components/UploadForm";

// function App() {
//   return <UploadForm />;
// }

// export default App;


// import React from "react";
// import WebcamComponent from "./components/WebcamComponent";

// function App() {
//   return <WebcamComponent />;
// }

// export default App;




// import React from "react";
// import VideoUpload from "./components/VideoUpload";

// function App() {
//   return <VideoUpload />;
// }

// export default App;




import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import UploadForm from "./components/UploadForm";
import NotificationPage from "./components/NotificationPage";
import CardPage from "./components/CardPage";

const App = () => {
  return (
    <BrowserRouter>
      <div className="flex justify-center gap-4 p-4 bg-gray-200">
        <Link to="/">Form</Link>
        <Link to="/notifications">Notifications</Link>
        <Link to="/cards">Cards</Link>
      </div>
      <Routes>
        <Route path="/" element={<UploadForm />} />
        <Route path="/notifications" element={<NotificationPage />} />
        <Route path="/cards" element={<CardPage />} />
      </Routes>
    </BrowserRouter>
  );
};

export default App;
