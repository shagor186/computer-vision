import axios from "axios";

const BASE_URL = "http://127.0.0.1:5500"; // Flask backend

export function submitPersonData(formData) {
  return axios.post(`${BASE_URL}/submit`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  })
  .then(res => res.data)
  .catch(err => { throw err; });
}
