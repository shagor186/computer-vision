import React from 'react';

const Sidebar = ({ isOpen, onClose, currentView, onNavigate }) => {
  const menuItems = [
    {
      title: 'Dashboard',
      icon: '📊',
      view: 'home',
      color: 'text-blue-600'
    },
    {
      title: 'Face Recognition',
      icon: '🤖',
      items: [
        { name: 'Add Person', view: 'add-person', icon: '👤' },
        { name: 'Train Model', view: 'train-model', icon: '⚙️' },
        { name: 'Image Recognition', view: 'image-recognition', icon: '🖼️' },
        { name: 'Webcam Recognition', view: 'webcam-recognition', icon: '📷' },
        { name: 'Video Recognition', view: 'video-recognition', icon: '🎬' },
      ]
    },
    {
      title: 'Reports',
      icon: '📋',
      items: [
        { name: 'All Reports', view: 'reports', icon: '📄' },
        { name: 'Create Report', view: 'create-report', icon: '➕' },
      ]
    },
    {
      title: 'People Database',
      icon: '👥',
      view: 'people-list',
      color: 'text-green-600'
    },
    {
      title: 'Notifications',
      icon: '🔔',
      view: 'notifications',
      color: 'text-orange-600'
    },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 flex z-40 md:hidden"
          onClick={onClose}
        >
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75"></div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 flex flex-col z-50 bg-white w-64 shadow-xl transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 md:static md:inset-0
      `}>
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-4 bg-blue-600 text-white">
          <h1 className="text-lg font-bold">Face Recognition System</h1>
          <button 
            onClick={onClose}
            className="md:hidden text-white"
          >
            ✕
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-4 space-y-2 overflow-y-auto">
          {menuItems.map((item, index) => (
            <div key={index}>
              {item.items ? (
                <div className="mb-4">
                  <div className="flex items-center px-2 py-2 text-gray-700 font-medium">
                    <span className="mr-3">{item.icon}</span>
                    {item.title}
                  </div>
                  <div className="ml-6 space-y-1">
                    {item.items.map((subItem, subIndex) => (
                      <button
                        key={subIndex}
                        onClick={() => onNavigate(subItem.view)}
                        className={`flex items-center w-full px-3 py-2 text-sm rounded-lg transition-colors ${
                          currentView === subItem.view
                            ? 'bg-blue-100 text-blue-700'
                            : 'text-gray-600 hover:bg-gray-100'
                        }`}
                      >
                        <span className="mr-2">{subItem.icon}</span>
                        {subItem.name}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => onNavigate(item.view)}
                  className={`flex items-center w-full px-3 py-3 text-sm rounded-lg transition-colors ${
                    currentView === item.view
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  <span className={`mr-3 ${item.color || 'text-gray-500'}`}>
                    {item.icon}
                  </span>
                  {item.title}
                </button>
              )}
            </div>
          ))}
        </nav>

        {/* User Info */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center">
            <div className="shrink-0">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                {localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).email.charAt(0).toUpperCase() : 'U'}
              </div>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium text-gray-700">
                {localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).email : 'User'}
              </p>
              <p className="text-xs text-gray-500">Administrator</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;