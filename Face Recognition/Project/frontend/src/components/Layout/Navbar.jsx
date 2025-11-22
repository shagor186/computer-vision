import React, { useState, useEffect } from 'react';
import NotificationBell from '../Notifications/NotificationBell';

const Navbar = ({ user, onLogout, onMenuToggle, currentView, onNavigate }) => {
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    fetchUnreadCount();
  }, []);

  const fetchUnreadCount = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:5000/api/notifications?unread_only=true&per_page=1', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setUnreadCount(data.unread_count);
      }
    } catch (error) {
      console.error('Error fetching notification count:', error);
    }
  };

  return (
    <nav className="bg-white shadow-lg border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side - Menu button and title */}
          <div className="flex items-center">
            <button
              onClick={onMenuToggle}
              className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 md:hidden"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <div className="hidden md:flex items-center space-x-8 ml-4">
              <button
                onClick={() => onNavigate('home')}
                className={`font-medium ${
                  currentView === 'home' ? 'text-blue-600' : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => onNavigate('reports')}
                className={`font-medium ${
                  currentView === 'reports' ? 'text-blue-600' : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                Reports
              </button>
              <button
                onClick={() => onNavigate('image-recognition')}
                className={`font-medium ${
                  currentView.startsWith('image') || currentView.startsWith('webcam') || currentView.startsWith('video') 
                    ? 'text-blue-600' 
                    : 'text-gray-700 hover:text-blue-600'
                }`}
              >
                Recognition
              </button>
            </div>
          </div>

          {/* Right side - Notifications and user info */}
          <div className="flex items-center space-x-4">
            <NotificationBell 
              unreadCount={unreadCount} 
              onNotificationClick={() => onNavigate('notifications')}
            />
            
            <div className="hidden md:flex items-center space-x-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                {user?.email?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-medium text-gray-700">
                  {user?.email || 'User'}
                </span>
                <span className="text-xs text-gray-500">Administrator</span>
              </div>
            </div>

            <button
              onClick={onLogout}
              className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors text-sm font-medium"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;