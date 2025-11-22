import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Stats from './Stats';
import QuickActions from './QuickActions';
import AddPerson from '../FaceRecognition/AddPerson';
import TrainModel from '../FaceRecognition/TrainModel';
import ImageRecognition from '../FaceRecognition/ImageRecognition';
import WebcamRecognition from '../FaceRecognition/WebcamRecognition';
import VideoRecognition from '../FaceRecognition/VideoRecognition';
import ReportList from '../Reports/ReportList';
import CreateReport from '../Reports/CreateReport';

const Home = ({ currentView, onNavigate }) => {
  const [stats, setStats] = useState({
    reports: { total: 0, active: 0, resolved: 0 },
    recognition: { total: 0, today: 0 },
    notifications: { unread: 0 },
    people: { total: 0 },
    model: { trained: false }
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (currentView === 'home') {
      fetchDashboardStats();
    }
  }, [currentView]);

  const fetchDashboardStats = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get('http://localhost:5000/api/stats/dashboard', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderCurrentView = () => {
    switch (currentView) {
      case 'add-person':
        return <AddPerson onBack={() => onNavigate('home')} />;
      case 'train-model':
        return <TrainModel onBack={() => onNavigate('home')} />;
      case 'image-recognition':
        return <ImageRecognition onBack={() => onNavigate('home')} />;
      case 'webcam-recognition':
        return <WebcamRecognition onBack={() => onNavigate('home')} />;
      case 'video-recognition':
        return <VideoRecognition onBack={() => onNavigate('home')} />;
      case 'reports':
        return <ReportList 
          onBack={() => onNavigate('home')} 
          onViewReport={(id) => console.log('View report:', id)}
          onCreateReport={() => onNavigate('create-report')}
        />;
      case 'create-report':
        return <CreateReport 
          onBack={() => onNavigate('reports')} 
          onReportCreated={() => onNavigate('reports')}
        />;
      case 'home':
      default:
        return (
          <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              {/* Header */}
              <div className="text-center mb-12">
                <h1 className="text-4xl font-bold text-gray-900 mb-4">
                  Welcome to Face Recognition System
                </h1>
                <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                  Advanced face recognition with authentication, reporting, and real-time notifications
                </p>
              </div>

              {/* Statistics */}
              <Stats stats={stats} loading={loading} />

              {/* Quick Actions */}
              <QuickActions onNavigate={onNavigate} />

              {/* Features Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-12">
                {[
                  {
                    title: "Add New Person",
                    description: "Add persons with name, age, location and 20-30 images",
                    icon: "👤",
                    color: "from-blue-500 to-blue-600",
                    view: "add-person"
                  },
                  {
                    title: "Train AI Model",
                    description: "Train the face recognition model with all persons",
                    icon: "🤖",
                    color: "from-green-500 to-green-600",
                    view: "train-model"
                  },
                  {
                    title: "Image Recognition",
                    description: "Recognize faces from uploaded images",
                    icon: "🖼️",
                    color: "from-purple-500 to-purple-600",
                    view: "image-recognition"
                  },
                  {
                    title: "Webcam Recognition",
                    description: "Real-time face recognition using webcam",
                    icon: "📷",
                    color: "from-orange-500 to-orange-600",
                    view: "webcam-recognition"
                  },
                  {
                    title: "Video Recognition",
                    description: "Recognize faces in video files",
                    icon: "🎬",
                    color: "from-red-500 to-red-600",
                    view: "video-recognition"
                  },
                  {
                    title: "Reports & Alerts",
                    description: "Manage missing/found person reports and get notifications",
                    icon: "📋",
                    color: "from-indigo-500 to-indigo-600",
                    view: "reports"
                  }
                ].map((feature, index) => (
                  <div
                    key={index}
                    onClick={() => onNavigate(feature.view)}
                    className="bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 cursor-pointer overflow-hidden"
                  >
                    <div className={`bg-gradient-to-r ${feature.color} p-6 text-white`}>
                      <div className="text-4xl mb-4">{feature.icon}</div>
                      <h3 className="text-xl font-bold">{feature.title}</h3>
                    </div>
                    <div className="p-6">
                      <p className="text-gray-600 mb-4">{feature.description}</p>
                      <button className="w-full bg-gray-100 hover:bg-gray-200 text-gray-800 font-semibold py-2 px-4 rounded-lg transition-colors">
                        Get Started
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
    }
  };

  return renderCurrentView();
};

export default Home;