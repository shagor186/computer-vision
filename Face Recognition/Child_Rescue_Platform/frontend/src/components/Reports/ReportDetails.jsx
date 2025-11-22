import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ReportDetails = ({ reportId, onBack }) => {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (reportId) {
      fetchReportDetails();
    }
  }, [reportId]);

  const fetchReportDetails = async () => {
    try {
      setLoading(true);
      setError('');
      const token = localStorage.getItem('token');
      const response = await axios.get(`http://127.0.0.1:5000/api/reports/${reportId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      setReport(response.data.report);
    } catch (error) {
      console.error('Error fetching report details:', error);
      setError('Failed to fetch report details');
    } finally {
      setLoading(false);
    }
  };

  const updateReportStatus = async (newStatus) => {
    try {
      setError('');
      const token = localStorage.getItem('token');
      await axios.put(`http://127.0.0.1:5000/api/reports/${reportId}/update-status`, {
        status: newStatus
      }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      // Refresh report details
      fetchReportDetails();
    } catch (error) {
      console.error('Error updating report status:', error);
      setError('Failed to update report status');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading report details...</p>
        </div>
      </div>
    );
  }

  if (error && !report) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={onBack}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Back to Reports
          </button>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
          <p className="text-red-600 mb-4">Report not found</p>
          <button
            onClick={onBack}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            Back to Reports
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <button
              onClick={onBack}
              className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-2 transition-colors"
            >
              ← Back to Reports
            </button>
            <h1 className="text-3xl font-bold text-gray-900">Report Details</h1>
            <p className="text-gray-600">Report #{report.report_number}</p>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-lg p-8">
          {/* Report Header */}
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">{report.title}</h2>
              <div className="flex items-center space-x-4">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  report.report_type === 'missing' 
                    ? 'bg-red-100 text-red-800' 
                    : 'bg-orange-100 text-orange-800'
                }`}>
                  {report.report_type === 'missing' ? 'Missing Person' : 'Found Person'}
                </span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  report.status === 'active' 
                    ? 'bg-green-100 text-green-800' 
                    : report.status === 'resolved'
                    ? 'bg-blue-100 text-blue-800'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {report.status ? report.status.charAt(0).toUpperCase() + report.status.slice(1) : 'Unknown'}
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Created</p>
              <p className="font-medium text-gray-900">
                {report.created_at ? new Date(report.created_at).toLocaleDateString() : 'Unknown date'}
              </p>
            </div>
          </div>

          {/* Person Information */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-1">Person Name</h3>
              <p className="font-medium text-gray-900">{report.person_name || 'Not specified'}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-1">Age</h3>
              <p className="font-medium text-gray-900">{report.person_age || 'Not specified'}</p>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-1">Location</h3>
              <p className="font-medium text-gray-900">{report.person_location || 'Not specified'}</p>
            </div>
          </div>

          {/* Description */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-500 mb-2">Description</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-gray-700 whitespace-pre-wrap">{report.description || 'No description provided'}</p>
            </div>
          </div>

          {/* Recognition Matches */}
          {report.recognition_matches !== undefined && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-500 mb-2">Recognition Matches</h3>
              <div className="bg-blue-50 rounded-lg p-4">
                <p className="text-blue-700">
                  This person has been recognized {report.recognition_matches} times in the system.
                </p>
                {report.recognition_matches > 0 && (
                  <p className="text-blue-600 text-sm mt-1">
                    You will receive notifications when this person is recognized again.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Status Actions */}
          <div className="border-t pt-6">
            <h3 className="text-sm font-medium text-gray-500 mb-3">Update Status</h3>
            <div className="flex flex-wrap gap-3">
              {report.status !== 'active' && (
                <button
                  onClick={() => updateReportStatus('active')}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Mark as Active
                </button>
              )}
              {report.status !== 'resolved' && (
                <button
                  onClick={() => updateReportStatus('resolved')}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Mark as Resolved
                </button>
              )}
              {report.status !== 'closed' && (
                <button
                  onClick={() => updateReportStatus('closed')}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Close Report
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportDetails;