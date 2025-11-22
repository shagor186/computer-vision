import React from 'react';

const Stats = ({ stats, loading }) => {
  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white rounded-2xl shadow-lg p-6 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
            <div className="h-8 bg-gray-200 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  const statCards = [
    {
      title: 'Total Reports',
      value: stats.reports.total,
      change: '+5%',
      icon: '📋',
      color: 'blue',
      description: 'Active: ' + stats.reports.active
    },
    {
      title: 'People in DB',
      value: stats.people.total,
      change: '+2',
      icon: '👥',
      color: 'green',
      description: 'Persons registered'
    },
    {
      title: 'Recognitions Today',
      value: stats.recognition.today,
      change: '+12%',
      icon: '🔍',
      color: 'purple',
      description: 'Total: ' + stats.recognition.total
    },
    {
      title: 'Unread Notifications',
      value: stats.notifications.unread,
      change: '+3',
      icon: '🔔',
      color: 'orange',
      description: 'Require attention'
    }
  ];

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-500',
      green: 'bg-green-500',
      purple: 'bg-purple-500',
      orange: 'bg-orange-500'
    };
    return colors[color] || 'bg-blue-500';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {statCards.map((stat, index) => (
        <div key={index} className="bg-white rounded-2xl shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">{stat.title}</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
              <p className="text-xs text-gray-500 mt-1">{stat.description}</p>
            </div>
            <div className={`${getColorClasses(stat.color)} rounded-lg p-3`}>
              <span className="text-2xl text-white">{stat.icon}</span>
            </div>
          </div>
          <div className="mt-4">
            <span className={`text-sm font-medium ${
              stat.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
            }`}>
              {stat.change} from last week
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Stats;