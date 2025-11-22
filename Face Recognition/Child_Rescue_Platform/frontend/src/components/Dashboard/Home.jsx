// import React, { useState, useEffect } from 'react';
// import axios from 'axios';

// const Home = ({ user, onLogout }) => {
//   const [userProfile, setUserProfile] = useState(null);
//   const [loading, setLoading] = useState(true);

//   useEffect(() => {
//     const fetchUserProfile = async () => {
//       try {
//         const token = localStorage.getItem('token');
//         const response = await axios.get('http://127.0.0.1:5000/api/user/profile', {
//           headers: {
//             'Authorization': `Bearer ${token}`
//           }
//         });
//         setUserProfile(response.data.user);
//       } catch (error) {
//         console.error('Failed to fetch user profile:', error);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchUserProfile();
//   }, []);

//   const handleLogout = () => {
//     onLogout();
//   };

//   if (loading) {
//     return (
//       <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-100 flex items-center justify-center">
//         <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
//           <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
//           <p className="text-gray-600">Loading your dashboard...</p>
//         </div>
//       </div>
//     );
//   }

//   const displayUser = userProfile || user;

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-purple-50 to-pink-100">
//       {/* Navigation */}
//       <nav className="bg-white shadow-lg">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex justify-between items-center h-16">
//             <div className="flex items-center">
//               <h1 className="text-2xl font-bold text-gray-800">Dashboard</h1>
//             </div>
//             <div className="flex items-center space-x-4">
//               <span className="text-gray-700 hidden md:block">
//                 Welcome, {displayUser?.email}
//               </span>
//               <button
//                 onClick={handleLogout}
//                 className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors"
//               >
//                 Logout
//               </button>
//             </div>
//           </div>
//         </div>
//       </nav>

//       {/* Main Content */}
//       <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
//         {/* Welcome Section */}
//         <div className="text-center mb-12">
//           <div className="w-24 h-24 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-6">
//             <span className="text-white text-2xl font-bold">
//               {displayUser?.email?.charAt(0).toUpperCase()}
//             </span>
//           </div>
//           <h1 className="text-4xl font-bold text-gray-900 mb-4">
//             Welcome to Your Dashboard!
//           </h1>
//           <p className="text-xl text-gray-600 max-w-2xl mx-auto">
//             Hello <strong>{displayUser?.email}</strong>, you have successfully signed in to your account.
//           </p>
//         </div>

//         {/* Stats Cards */}
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
//           <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-blue-500 hover:shadow-xl transition-shadow">
//             <div className="flex items-center">
//               <div className="bg-blue-100 p-3 rounded-lg">
//                 <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
//                 </svg>
//               </div>
//               <div className="ml-4">
//                 <h3 className="text-lg font-semibold text-gray-900">Profile Status</h3>
//                 <p className="text-gray-600">Active</p>
//               </div>
//             </div>
//           </div>

//           <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-green-500 hover:shadow-xl transition-shadow">
//             <div className="flex items-center">
//               <div className="bg-green-100 p-3 rounded-lg">
//                 <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
//                 </svg>
//               </div>
//               <div className="ml-4">
//                 <h3 className="text-lg font-semibold text-gray-900">Account Type</h3>
//                 <p className="text-gray-600">Standard</p>
//               </div>
//             </div>
//           </div>

//           <div className="bg-white rounded-2xl shadow-lg p-6 border-l-4 border-purple-500 hover:shadow-xl transition-shadow">
//             <div className="flex items-center">
//               <div className="bg-purple-100 p-3 rounded-lg">
//                 <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
//                 </svg>
//               </div>
//               <div className="ml-4">
//                 <h3 className="text-lg font-semibold text-gray-900">Member Since</h3>
//                 <p className="text-gray-600">
//                   {displayUser?.created_at ? new Date(displayUser.created_at).toLocaleDateString() : 'Recently'}
//                 </p>
//               </div>
//             </div>
//           </div>
//         </div>

//         {/* Quick Actions */}
//         <div className="bg-white rounded-2xl shadow-lg p-8">
//           <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Actions</h2>
//           <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
//             <button className="bg-gray-50 hover:bg-gray-100 p-6 rounded-xl transition-colors text-left group">
//               <div className="bg-blue-100 p-3 rounded-lg w-12 h-12 flex items-center justify-center mb-4 group-hover:bg-blue-200 transition-colors">
//                 <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
//                 </svg>
//               </div>
//               <h3 className="font-semibold text-gray-900 mb-2">Edit Profile</h3>
//               <p className="text-gray-600 text-sm">Update your personal information</p>
//             </button>

//             <button className="bg-gray-50 hover:bg-gray-100 p-6 rounded-xl transition-colors text-left group">
//               <div className="bg-red-100 p-3 rounded-lg w-12 h-12 flex items-center justify-center mb-4 group-hover:bg-red-200 transition-colors">
//                 <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
//                 </svg>
//               </div>
//               <h3 className="font-semibold text-gray-900 mb-2">Change Password</h3>
//               <p className="text-gray-600 text-sm">Update your security settings</p>
//             </button>

//             <button className="bg-gray-50 hover:bg-gray-100 p-6 rounded-xl transition-colors text-left group">
//               <div className="bg-yellow-100 p-3 rounded-lg w-12 h-12 flex items-center justify-center mb-4 group-hover:bg-yellow-200 transition-colors">
//                 <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
//                 </svg>
//               </div>
//               <h3 className="font-semibold text-gray-900 mb-2">Notifications</h3>
//               <p className="text-gray-600 text-sm">Manage your alerts</p>
//             </button>

//             <button className="bg-gray-50 hover:bg-gray-100 p-6 rounded-xl transition-colors text-left group">
//               <div className="bg-green-100 p-3 rounded-lg w-12 h-12 flex items-center justify-center mb-4 group-hover:bg-green-200 transition-colors">
//                 <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
//                   <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
//                 </svg>
//               </div>
//               <h3 className="font-semibold text-gray-900 mb-2">Privacy</h3>
//               <p className="text-gray-600 text-sm">Control your data</p>
//             </button>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Home;





// ...........................................................

import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-800">👁️ Face Recognition System</h1>
            </div>
            <div className="flex space-x-4">
              <Link to="/add-person" className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition">
                Add Person
              </Link>
              <Link to="/train-model" className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 transition">
                Train Model
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-800 mb-6">
            Intelligent Face Recognition System
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Advanced face detection and recognition using Deep Learning and SVM. 
            Add persons, train models, and recognize faces in real-time.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
          {/* Add Person Card */}
          <Link to="/add-person" className="block">
            <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition cursor-pointer border-2 border-blue-200">
              <div className="text-blue-500 text-center mb-4">
                <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 text-center mb-2">Add Person</h3>
              <p className="text-gray-600 text-center">
                Add new persons to the dataset by uploading multiple images
              </p>
            </div>
          </Link>

          {/* Train Model Card */}
          <Link to="/train-model" className="block">
            <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition cursor-pointer border-2 border-green-200">
              <div className="text-green-500 text-center mb-4">
                <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 text-center mb-2">Train Model</h3>
              <p className="text-gray-600 text-center">
                Train the SVM model with the collected face dataset
              </p>
            </div>
          </Link>

          {/* Image Recognition Card */}
          <Link to="/image-recognition" className="block">
            <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition cursor-pointer border-2 border-purple-200">
              <div className="text-purple-500 text-center mb-4">
                <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 text-center mb-2">Image Recognition</h3>
              <p className="text-gray-600 text-center">
                Upload an image and detect faces with recognition
              </p>
            </div>
          </Link>

          {/* Webcam Recognition Card */}
          <Link to="/webcam-recognition" className="block">
            <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition cursor-pointer border-2 border-red-200">
              <div className="text-red-500 text-center mb-4">
                <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 text-center mb-2">Webcam Recognition</h3>
              <p className="text-gray-600 text-center">
                Real-time face recognition using your webcam
              </p>
            </div>
          </Link>
        </div>

        {/* Quick Stats */}
        <div className="mt-16 bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-800 text-center mb-8">System Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4">
              <div className="text-3xl font-bold text-blue-600 mb-2">MTCNN</div>
              <p className="text-gray-600">Advanced Face Detection</p>
            </div>
            <div className="text-center p-4">
              <div className="text-3xl font-bold text-green-600 mb-2">FaceNet</div>
              <p className="text-gray-600">Deep Face Embeddings</p>
            </div>
            <div className="text-center p-4">
              <div className="text-3xl font-bold text-purple-600 mb-2">SVM</div>
              <p className="text-gray-600">Machine Learning Classification</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;