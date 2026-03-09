import React, { useState } from "react";
import "./signup.css";
import TypingProfile from "./TypingProfile";

const Signup = ({ setPage }) => {
  const [step, setStep] = useState(1);

  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleContinue = () => {
    // Check if all fields are filled
    if (
      !formData.username ||
      !formData.email ||
      !formData.password ||
      !formData.confirmPassword
    ) {
      alert("Please fill all fields");
      return;
    }

    // Trim whitespace and check again
    if (
      formData.username.trim() === "" ||
      formData.email.trim() === "" ||
      formData.password.trim() === "" ||
      formData.confirmPassword.trim() === ""
    ) {
      alert("Please fill all fields properly");
      return;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.email)) {
      alert("Please enter a valid email address");
      return;
    }

    // Check username length and format
    if (formData.username.length < 3) {
      alert("Username must be at least 3 characters");
      return;
    }
    const usernameRegex = /^[a-zA-Z0-9]+$/;
    if (!usernameRegex.test(formData.username)) {
      alert("Username must contain only letters and numbers");
      return;
    }

    // Check password rules
    if (formData.password.length < 8) {
      alert("Password must be at least 8 characters");
      return;
    }
    if (!/\d/.test(formData.password)) {
      alert("Password must contain at least one number");
      return;
    }
    if (!/[A-Z]/.test(formData.password)) {
      alert("Password must contain at least one uppercase letter");
      return;
    }
    if (!/[!@#$%^&*()_+?<>,"'-=.\/]/.test(formData.password)) {
      alert("Password must contain at least one special character");
      return;
    }

    // Check if passwords match
    if (formData.password !== formData.confirmPassword) {
      alert("Passwords do not match");
      return;
    }

    // Everything is valid, move to step 2
    setStep(2);
  };

  return (
    <>
      {step === 1 && (
        <div className="signup-wrapper">
          <div className="signup-box">
            <h2>Create Account</h2>
            <p className="subtitle">
              Create your account before setting up typing identity
            </p>

            <input
              type="text"
              name="username"
              placeholder="Username"
              value={formData.username}
              onChange={handleChange}
            />

            <input
              type="email"
              name="email"
              placeholder="Email"
              value={formData.email}
              onChange={handleChange}
            />

            <input
              type="password"
              name="password"
              placeholder="Password"
              value={formData.password}
              onChange={handleChange}
            />

            <input
              type="password"
              name="confirmPassword"
              placeholder="Confirm Password"
              value={formData.confirmPassword}
              onChange={handleChange}
            />

            <button onClick={handleContinue}>Continue</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <TypingProfile
          username={formData.username}
          email={formData.email}
          password={formData.password}
          setPage={setPage}
        />
      )}
    </>
  );
};

export default Signup;