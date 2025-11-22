import React, { useState } from 'react';
import axios from 'axios';

const AddPerson = () => {
  const [personName, setPersonName] = useState('');
  const [images, setImages] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleImageChange = (e) => {
    const files = Array.from(e.target.files);
    setImages(files);
    
    // Create preview URLs
    const newPreviews = files.map(file => URL.createObjectURL(file));
    setPreviews(newPreviews);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!personName.trim()) {
      setMessage('Please enter person name');
      return;
    }
    
    if (images.length === 0) {
      setMessage('Please select at least one image');
      return;
    }

    setLoading(true);
    setMessage('');

    const formData = new FormData();
    formData.append('personName', personName.trim());
    images.forEach(image => {
      formData.append('images', image);
    });

    try {
      const response = await axios.post('http://localhost:5000/api/add-person', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setMessage(`✅ ${response.data.message}`);
      setPersonName('');
      setImages([]);
      setPreviews([]);
      
      // Clear file input
      document.getElementById('imageInput').value = '';
      
    } catch (error) {
      setMessage(`❌ Error: ${error.response?.data?.message || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center">
            Add New Person
          </h1>

          {message && (
            <div className={`p-4 rounded-lg mb-6 ${
              message.includes('✅') ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'
            }`}>
              {message}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Person Name Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Person Name *
              </label>
              <input
                type="text"
                value={personName}
                onChange={(e) => setPersonName(e.target.value)}
                placeholder="Enter person's name (e.g., John Doe)"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>

            {/* Image Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Face Images *
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition">
                <input
                  id="imageInput"
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleImageChange}
                  className="hidden"
                />
                <label htmlFor="imageInput" className="cursor-pointer">
                  <svg className="w-12 h-12 text-gray-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="text-gray-600">
                    Click to upload images or drag and drop
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    PNG, JPG, JPEG up to 10MB each
                  </p>
                </label>
              </div>
              {images.length > 0 && (
                <p className="text-sm text-gray-600 mt-2">
                  Selected {images.length} image(s)
                </p>
              )}
            </div>

            {/* Image Previews */}
            {previews.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Image Previews:</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {previews.map((preview, index) => (
                    <div key={index} className="relative">
                      <img
                        src={preview}
                        alt={`Preview ${index + 1}`}
                        className="w-full h-24 object-cover rounded-lg"
                      />
                      <div className="absolute top-1 right-1 bg-black bg-opacity-50 text-white text-xs px-1 rounded">
                        {index + 1}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:ring-4 focus:ring-blue-200 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Adding Person...
                </div>
              ) : (
                'Add Person to Dataset'
              )}
            </button>
          </form>

          {/* Instructions */}
          <div className="mt-8 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">📝 Instructions:</h3>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Use clear, front-facing face images</li>
              <li>• Good lighting conditions recommended</li>
              <li>• Upload 5-10 images per person for best results</li>
              <li>• Avoid blurry or dark images</li>
              <li>• Images should show face clearly without obstructions</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AddPerson;