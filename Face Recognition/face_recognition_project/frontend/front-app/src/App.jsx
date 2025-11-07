// import React from "react";
// import UploadForm from "./components/UploadForm";
// import PredictForm from "./components/PredictForm";
// import VideoPredictForm from "./components/VideoPredictForm";
// import WebcamPredict from "./components/WebcamPredict";
// import NotificationSidebar from "./components/NotificationSidebar";

// export default function App(){
//   return (
//     <div className="min-h-screen">
//       <header className="bg-white shadow">
//         <div className="max-w-5xl mx-auto py-6 px-4">
//           <h1 className="text-2xl font-semibold">Face Recognition System</h1>
//         </div>
//       </header>
//       <main className="max-w-5xl mx-auto my-8 px-4 grid grid-cols-1 md:grid-cols-2 gap-6">
//         <div>
//           <UploadForm />
//           <PredictForm />
//           <VideoPredictForm />
//           <WebcamPredict />
//         </div>
//       </main>
//       <NotificationSidebar />
//     </div>
//   );
// }



import UploadForm from "./components/UploadForm";
import PredictForm from "./components/PredictForm";
import VideoPredictForm from "./components/VideoPredictForm";
import WebcamPredict from "./components/WebcamPredict";
import NotificationSidebar from "./components/NotificationSidebar";

export default function App(){
  return (
    <div className="flex">
      <div className="flex-1 p-8">
        <h1 className="text-2xl font-bold mb-6 text-center">Face Recognition App</h1>
        <UploadForm />
      </div>
      <div>
        <PredictForm />
        <VideoPredictForm />
        <WebcamPredict />
      </div>
      <NotificationSidebar />
    </div>
  );
}
