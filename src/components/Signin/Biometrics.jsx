import React from "react";
import MLTypingAPP from "./MLTypingAPP";
import "./signin.css";

export default function Biometrics({ setPage, userId, onLoginSuccess }) {
  if (!userId) {
     return <div style={{color: 'white', textAlign: 'center', marginTop: '100px'}}>No user provided. Go back and login.</div>;
  }

  return (
    <div className="signin-wrapper" style={{alignItems: 'flex-start', paddingTop: '15vh'}}>
      <MLTypingAPP username={userId} onLoginSuccess={onLoginSuccess} />
    </div>
  );
}
