import React from 'react';

const QuickActions = ({ onNavigate }) => {
  const quickActions = [
    {
      title: 'Add Person',
      description: 'Add new person to database',
      icon: '👤',
      color: 'bg-blue-500',
      action: () => onNavigate('add-person')
    },
    {
      title: 'Image Recognition',
      description: 'Recognize face from image',
      icon: '🖼️',
      color: 'bg-green-500',
      action: () => onNavigate('image-recognition')
    },
    {
      title: 'Create Report',
      description: 'Report missing/found person',
      icon: '📋',
      color: 'bg-purple-500',
      action: () => onNavigate('create-report')
    },
    {
      title: 'View Reports',
      description: 'Check all reports',
      icon: '📊',
      color: 'bg-orange-500',
      action: () => onNavigate('reports')
    }
  ];

  return (
    <div className="bg-white rounded-2xl shadow-lg p-8 mb-8">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {quickActions.map((action, index) => (
          <button
            key={index}
            onClick={action.action}
            className="bg-gray-50 hover:bg-gray-100 rounded-xl p-6 text-left transition-colors group"
          >
            <div className={`${action.color} rounded-lg w-12 h-12 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
              <span className="text-2xl text-white">{action.icon}</span>
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">{action.title}</h3>
            <p className="text-sm text-gray-600">{action.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
};

export default QuickActions;