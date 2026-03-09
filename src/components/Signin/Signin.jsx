import React, { useState } from "react";
import "./signin.css";

const Signin = ({ setPage, setUserId, onLoginSuccess }) => {
  const [loginMethod, setLoginMethod] = useState("typing");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const handleUsernameSubmit = (e) => {
    e.preventDefault();
    if (!username.trim()) {
      alert("Please enter your username");
      return;
    }
    
    if (username.trim().toLowerCase() === "admin") {
      setLoginMethod("password");
      return;
    }
    
    const isAlnum = /^[a-zA-Z0-9]+$/.test(username.trim());
    if (username.trim().length < 3 || !isAlnum) {
      alert("Invalid username format");
      return;
    }
    
    // redirect to biometrics page instead of flipping states in this component
    setUserId(username);
    setPage("biometrics");
  };

  const handlePasswordLogin = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setLoading(true);

    const isAlnum = /^[a-zA-Z0-9]+$/.test(username.trim());
    const isEmail = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(username.trim());
    if (!isAlnum && !isEmail) {
      setErrorMsg("Invalid username or email format");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5001/api/login-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username, password }),
      });

      const data = await response.json();

      if (response.ok && data.access_granted) {
        if (onLoginSuccess) {
          onLoginSuccess(data.user_id, data.role || 'user');
        } else {
          setPage("home");
        }
      } else {
        setErrorMsg("Invalid username or password");
      }
    } catch (err) {
      console.error("Login error:", err);
      setErrorMsg("Error connecting to server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="signin-wrapper">
      <div className="signin-box">
        {loginMethod === "password" ? (
          <>
            <h2>Sign In (Password)</h2>
            <p style={{marginBottom: '20px', color: '#666'}}>Enter credentials</p>

            {errorMsg && <div className="error-msg">{errorMsg}</div>}

            <form onSubmit={handlePasswordLogin}>
              <input
                type="text"
                className="default-input"
                placeholder="Username or Email"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
              <input
                type="password"
                className="default-input"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              
              <button 
                type="submit" 
                className="default-btn"
                disabled={loading}
              >
                {loading ? "Authenticating..." : "Login"}
              </button>
            </form>

            <p className="password-link" onClick={() => setLoginMethod("typing")}>
              Login with typing pattern instead?
            </p>
          </>
        ) : (
          <>
            <h2>Verify your identity</h2>
            <p style={{marginBottom: '20px', color: '#666'}}>Enter your username to begin verification</p>

            <form onSubmit={handleUsernameSubmit}>
              <input
                type="text"
                name="username"
                className="default-input"
                placeholder="Enter Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoFocus
                required
              />
              <button type="submit" className="default-btn">Continue</button>
            </form>

            <p className="password-link" onClick={() => setLoginMethod("password")}>
              Login with password instead?
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export default Signin;
