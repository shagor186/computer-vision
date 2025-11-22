import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const ReportList = ({ onBack, onViewReport, onCreateReport }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [pagination, setPagination] = useState({
    currentPage: 1,
    totalPages: 1,
    totalReports: 0
  });
  const [filters, setFilters] = useState({
    type: '',
    status: ''
  });

  // Fetch reports function with useCallback to prevent unnecessary re-renders
  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError('');
    
    try {
      const token = localStorage.getItem('token');
      
      if (!token) {
        setError('Authentication token not found. Please login again.');
        setLoading(false);
        return;
      }

      const params = new URLSearchParams({
        page: pagination.currentPage,
        per_page: 10,
        ...filters
      });

      const response = await axios.get(`http://127.0.0.1:5000/api/reports/list?${params}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        },
        timeout: 10000 // 10 seconds timeout
      });

      if (response.data && Array.isArray(response.data.reports)) {
        setReports(response.data.reports);
        setPagination(prev => ({
          ...prev,
          totalPages: response.data.pages || 1,
          totalReports: response.data.total || 0
        }));
      } else {
        throw new Error('Invalid response format from server');
      }

    } catch (error) {
      handleFetchError(error);
    } finally {
      setLoading(false);
    }
  }, [pagination.currentPage, filters]); // Proper dependencies

  // Handle fetch errors
  const handleFetchError = (error) => {
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // Server responded with error status
        switch (error.response.status) {
          case 401:
            setError('Authentication failed. Please login again.');
            break;
          case 403:
            setError('You do not have permission to view reports.');
            break;
          case 404:
            setError('Reports endpoint not found.');
            break;
          case 500:
            setError('Server error. Please try again later.');
            break;
          default:
            setError(error.response.data?.error || `Error: ${error.response.status}`);
        }
      } else if (error.request) {
        // No response received
        setError('Cannot connect to server. Please check if the backend is running.');
      } else {
        setError(`Request error: ${error.message}`);
      }
    } else {
      setError(`Unexpected error: ${error.message}`);
    }
    setReports([]); // Clear reports on error
  };

  // useEffect with proper dependency array
  useEffect(() => {
    fetchReports();
  }, [fetchReports]); // Now fetchReports is stable due to useCallback

  // Handle filter changes
  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
    // Reset to first page when filters change
    setPagination(prev => ({ ...prev, currentPage: 1 }));
  };

  // Handle pagination
  const handlePageChange = (newPage) => {
    setPagination(prev => ({ ...prev, currentPage: newPage }));
  };

  // Refresh reports
  const handleRefresh = () => {
    fetchReports();
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      active: { 
        color: 'bg-green-100 text-green-800 border border-green-200', 
        label: 'Active',
        icon: '🟢'
      },
      resolved: { 
        color: 'bg-blue-100 text-blue-800 border border-blue-200', 
        label: 'Resolved',
        icon: '✅'
      },
      closed: { 
        color: 'bg-gray-100 text-gray-800 border border-gray-200', 
        label: 'Closed',
        icon: '🔒'
      }
    };
    
    const config = statusConfig[status] || statusConfig.active;
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${config.color}`}>
        <span>{config.icon}</span>
        <span>{config.label}</span>
      </span>
    );
  };

  const getTypeBadge = (type) => {
    const typeConfig = {
      missing: {
        color: 'bg-red-100 text-red-800 border border-red-200',
        label: 'Missing',
        icon: '🔍'
      },
      found: {
        color: 'bg-orange-100 text-orange-800 border border-orange-200',
        label: 'Found',
        icon: '👤'
      }
    };
    
    const config = typeConfig[type] || typeConfig.missing;
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${config.color}`}>
        <span>{config.icon}</span>
        <span>{config.label}</span>
      </span>
    );
  };

  // Loading state
  if (loading && reports.length === 0) {
    return (
      <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading reports...</p>
          <p className="text-sm text-gray-500 mt-2">Please wait while we fetch your reports</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-8 gap-4">
          <div className="flex-1">
            <button
              onClick={onBack}
              className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-3 transition-colors font-medium"
            >
              ← Back to Dashboard
            </button>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Reports Management</h1>
            <p className="text-gray-600 max-w-2xl">
              Manage your missing and found person reports. Track status and receive notifications when persons are recognized.
            </p>
          </div>
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="px-4 py-3 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white font-semibold rounded-lg transition-colors flex items-center justify-center"
            >
              <svg className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
            <button
              onClick={onCreateReport}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors flex items-center justify-center"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
              </svg>
              New Report
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-xl p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <svg className="w-5 h-5 text-red-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div>
                  <h3 className="text-red-800 font-semibold">Error Loading Reports</h3>
                  <p className="text-red-700 text-sm mt-1">{error}</p>
                </div>
              </div>
              <button
                onClick={handleRefresh}
                className="text-red-700 hover:text-red-800 font-medium text-sm"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Filters</h2>
            <div className="text-sm text-gray-600">
              Showing {reports.length} of {pagination.totalReports} reports
            </div>
          </div>
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Report Type
              </label>
              <select
                value={filters.type}
                onChange={(e) => handleFilterChange('type', e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                <option value="">All Report Types</option>
                <option value="missing">Missing Person</option>
                <option value="found">Found Person</option>
              </select>
            </div>
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Status
              </label>
              <select
                value={filters.status}
                onChange={(e) => handleFilterChange('status', e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              >
                <option value="">All Status</option>
                <option value="active">Active</option>
                <option value="resolved">Resolved</option>
                <option value="closed">Closed</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={() => {
                  setFilters({ type: '', status: '' });
                  setPagination(prev => ({ ...prev, currentPage: 1 }));
                }}
                className="px-4 py-2.5 text-gray-600 hover:text-gray-800 font-medium text-sm transition-colors"
              >
                Clear Filters
              </button>
            </div>
          </div>
        </div>

        {/* Reports List */}
        <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
          {reports.length === 0 && !loading ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p className="text-gray-500 text-lg mb-2">No reports found</p>
              <p className="text-gray-400 text-sm mb-6 max-w-md mx-auto">
                {filters.type || filters.status 
                  ? 'Try adjusting your filters to see more results.' 
                  : 'Get started by creating your first missing or found person report.'
                }
              </p>
              <button
                onClick={onCreateReport}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors inline-flex items-center"
              >
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
                Create Your First Report
              </button>
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        Report Details
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        Person
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        Date Created
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {reports.map((report) => (
                      <tr 
                        key={report.id} 
                        className="hover:bg-gray-50 transition-colors cursor-pointer"
                        onClick={() => onViewReport && onViewReport(report.id)}
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-start space-x-3">
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-2 mb-2">
                                {getTypeBadge(report.report_type)}
                                <span className="font-mono text-sm font-medium text-gray-900 bg-gray-100 px-2 py-1 rounded">
                                  {report.report_number}
                                </span>
                              </div>
                              <div className="text-sm font-semibold text-gray-900 line-clamp-2">
                                {report.title}
                              </div>
                              {report.description && (
                                <p className="text-sm text-gray-500 mt-1 line-clamp-1">
                                  {report.description}
                                </p>
                              )}
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="text-sm font-medium text-gray-900">
                            {report.person_name}
                          </div>
                          {report.person_age && (
                            <div className="text-sm text-gray-500">
                              Age: {report.person_age}
                            </div>
                          )}
                          {report.person_location && (
                            <div className="text-sm text-gray-500 line-clamp-1">
                              📍 {report.person_location}
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          {getStatusBadge(report.status)}
                        </td>
                        <td className="px-6 py-4">
                          <div className="text-sm text-gray-900">
                            {new Date(report.created_at).toLocaleDateString()}
                          </div>
                          <div className="text-xs text-gray-400">
                            {new Date(report.created_at).toLocaleTimeString()}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onViewReport && onViewReport(report.id);
                            }}
                            className="text-blue-600 hover:text-blue-700 font-medium text-sm inline-flex items-center transition-colors"
                          >
                            View Details
                            <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                            </svg>
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {pagination.totalPages > 1 && (
                <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
                  <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                    <div className="text-sm text-gray-700">
                      Page {pagination.currentPage} of {pagination.totalPages} •{' '}
                      {pagination.totalReports} total reports
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handlePageChange(pagination.currentPage - 1)}
                        disabled={pagination.currentPage === 1}
                        className="px-4 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors flex items-center"
                      >
                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
                        </svg>
                        Previous
                      </button>
                      
                      {/* Page Numbers */}
                      <div className="flex space-x-1">
                        {[...Array(Math.min(5, pagination.totalPages))].map((_, index) => {
                          const pageNum = Math.max(1, Math.min(
                            pagination.currentPage - 2,
                            pagination.totalPages - 4
                          )) + index;
                          
                          if (pageNum > 0 && pageNum <= pagination.totalPages) {
                            return (
                              <button
                                key={pageNum}
                                onClick={() => handlePageChange(pageNum)}
                                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                                  pagination.currentPage === pageNum
                                    ? 'bg-blue-600 text-white'
                                    : 'text-gray-700 hover:bg-gray-100'
                                }`}
                              >
                                {pageNum}
                              </button>
                            );
                          }
                          return null;
                        })}
                      </div>

                      <button
                        onClick={() => handlePageChange(pagination.currentPage + 1)}
                        disabled={pagination.currentPage === pagination.totalPages}
                        className="px-4 py-2 border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors flex items-center"
                      >
                        Next
                        <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ReportList;