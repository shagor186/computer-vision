import React, { useState, useEffect } from 'react';
import Signin from './components/Auth/Signin';
import Signup from './components/Auth/Signup';
import ForgotPassword from './components/Auth/ForgotPassword';
import Home from './components/Dashboard/Home';

function App() {
  const [currentView, setCurrentView] = useState('signin');
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check if user is already logged in
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        const savedUser = localStorage.getItem('user');
        const token = localStorage.getItem('token');

        if (!savedUser || !token) {
          setLoading(false);
          return;
        }

        // Verify token with backend
        const response = await fetch('http://127.0.0.1:5000/api/verify-token', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const userData = JSON.parse(savedUser);
          setUser(userData);
          setCurrentView('home');
        } else {
          // Token is invalid, clear storage
          localStorage.removeItem('user');
          localStorage.removeItem('token');
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        localStorage.removeItem('user');
        localStorage.removeItem('token');
      } finally {
        setLoading(false);
      }
    };

    checkAuthStatus();
  }, []);

  const handleLogin = (userData, token) => {
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
    localStorage.setItem('token', token);
    setCurrentView('home');
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
    setCurrentView('signin');
  };

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-xl p-8 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Checking authentication...</p>
        </div>
      </div>
    );
  }

  const renderView = () => {
    switch (currentView) {
      case 'signin':
        return (
          <Signin 
            switchToSignup={() => setCurrentView('signup')}
            switchToForgotPassword={() => setCurrentView('forgot')}
            onLogin={handleLogin}
          />
        );
      case 'signup':
        return <Signup 
          switchToLogin={() => setCurrentView('signin')}
          onLogin={handleLogin}
        />;
      case 'forgot':
        return <ForgotPassword switchToLogin={() => setCurrentView('signin')} />;
      case 'home':
        return <Home user={user} onLogout={handleLogout} />;
      default:
        return <Signin switchToSignup={() => setCurrentView('signup')} />;
    }
  };

  return (
    <div className="App">
      {renderView()}
    </div>
  );
}

export default App;





