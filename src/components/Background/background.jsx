import React from "react";
import "./background.css";
import homeImage from "../../assets/home4.jpg";

const Background = ({ setPage }) => {
  return (
    <div className="cyber-theme">
      <section className="hero" style={{ backgroundImage: `url(${homeImage})` }}>
        <div className="hero-overlay">
          <h1 className="glow-text">Behavioral Biometrics Security</h1>
          <p>Advanced Authentication Using Keystroke Dynamics & AI Analysis</p>

          <button className="cyber-btn" onClick={() => setPage("signin")}>
            ACCESS SYSTEM
          </button>
        </div>
      </section>

      <section className="section">
        <h2 className="glow-text">About The System</h2>
        <p>
          Our system verifies users not just by passwords, but by analyzing their typing rhythm,
          speed, and behavioral patterns in real-time.
        </p>
      </section>

      <section className="section dark">
        <h2 className="glow-text">Security Modules</h2>

        <div className="features">
          <div className="card">
            <h3>Keystroke Dynamics</h3>
            <p>Measures typing speed & dwell time.</p>
          </div>

          <div className="card">
            <h3>AI Matching</h3>
            <p>Compares behavioral patterns instantly.</p>
          </div>

          <div className="card">
            <h3>Intrusion Detection</h3>
            <p>Detects abnormal user behavior.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Background;
