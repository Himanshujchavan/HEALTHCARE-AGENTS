import API from "./api"

// Register with form data
export const registerUser = async (data) => {
  const res = await API.post("/auth/register", data)
  return res.data
}

// Login with form data
export const loginUser = async (data) => {
  const formData = new URLSearchParams()
  formData.append("username", data.username)
  formData.append("password", data.password)
  const res = await API.post("/auth/login", formData)
  return res.data
}