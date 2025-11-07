import React, { useState } from 'react'
import axiosClient from '../utils/axiosClient'

const Report = () => {
  // Form state
  const [form, setForm] = useState({
    name: '',
    age: '',
    gender: '',
    location: ''
  })

  // Multiple images state
  const [files, setFiles] = useState([])

  // Handle text input change
  const handleChange = e => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  // Handle file input change
  const handleFiles = e => {
    setFiles(e.target.files)
  }

  // Handle form submit
  const handleSubmit = e => {
    e.preventDefault() // 🔹 Prevent page reload

    if (!form.name || !form.age || !form.gender || !form.location) {
      alert("All fields are required")
      return
    }

    if (files.length === 0) {
      alert("Please select at least one image")
      return
    }

    const formData = new FormData()
    formData.append("name", form.name)
    formData.append("age", form.age)
    formData.append("gender", form.gender)
    formData.append("location", form.location)

    for (let i = 0; i < files.length; i++) {
      formData.append("images", files[i])
    }

    axiosClient.post("/report/add", formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    .then(res => {
      alert(res.data.msg)
      setForm({ name: '', age: '', gender: '', location: '' })
      setFiles([])
    })
    .catch(err => {
      console.error(err)
      alert("Error adding person")
    })
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded shadow-md max-w-lg mx-auto mt-6">
      <h2 className="text-2xl font-bold mb-4">Add Missing Person</h2>

      <input
        type="text"
        name="name"
        placeholder="Name"
        value={form.name}
        onChange={handleChange}
        className="border p-2 mb-3 w-full rounded"
        required
      />

      <input
        type="number"
        name="age"
        placeholder="Age"
        value={form.age}
        onChange={handleChange}
        className="border p-2 mb-3 w-full rounded"
        required
      />

      <input
        type="text"
        name="gender"
        placeholder="Gender"
        value={form.gender}
        onChange={handleChange}
        className="border p-2 mb-3 w-full rounded"
        required
      />

      <input
        type="text"
        name="location"
        placeholder="Location"
        value={form.location}
        onChange={handleChange}
        className="border p-2 mb-3 w-full rounded"
        required
      />

      <input
        type="file"
        multiple
        accept="image/*"
        onChange={handleFiles}
        className="border p-2 mb-3 w-full rounded"
        required
      />

      <button
        type="submit"
        className="bg-blue-500 text-white p-2 rounded w-full hover:bg-blue-600 transition"
      >
        Add Person
      </button>
    </form>
  )
}

export default Report
