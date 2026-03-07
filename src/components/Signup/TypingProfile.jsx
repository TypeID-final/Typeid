import React, { useState, useRef } from "react";
import "./signup.css";
import { freshState, handleKeyDownEvent, handleKeyUpEvent, buildFeatures } from "../../utils/TypingBiometrics";

const TOTAL_ATTEMPTS = 5;
const TARGET_SENTENCE = "the quick brown fox jumps over the lazy dog";
const sentences = Array(TOTAL_ATTEMPTS).fill(TARGET_SENTENCE);

const TypingProfile = ({ username, email, password, setPage }) => {
  const [attempt, setAttempt] = useState(1);
  const [text, setText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const currentSentence = sentences[attempt - 1];

  const ks = useRef(freshState());

  const handleKeyDown = (e) => {
    if (isSubmitting) return;
    handleKeyDownEvent(e, ks.current);
  };

  const handleKeyUp = (e) => {
    if (isSubmitting) return;
    handleKeyUpEvent(e, ks.current);
  };

  const normalizeSentence = (value) => value.toLowerCase().trim().replace(/\s+/g, " ");

  const handleSubmit = async () => {
    if (isSubmitting) return;

    if (normalizeSentence(text) !== normalizeSentence(currentSentence)) {
      alert("Sentence doesn't match. Please type it exactly.");
      return;
    }

    const features = buildFeatures(ks.current, text);

    if (ks.current.keyCount < 10) {
      alert("Not enough typing data. Please type the full sentence.");
      return;
    }

    console.log("Calculated features:", features);

    setIsSubmitting(true);

    try {
      const response = await fetch("http://127.0.0.1:5001/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: username,
          email: email,
          role: "student",
          password: password,
          keystroke_features: features,
          sample_text: currentSentence,
          attempt: attempt,
        }),
      });

      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }

      if (response.ok) {
        if (attempt < TOTAL_ATTEMPTS) {
          alert(`Attempt ${attempt}/${TOTAL_ATTEMPTS} completed!`);
          setAttempt((prev) => prev + 1);
          setText("");
          ks.current = freshState();
          setIsSubmitting(false);
        } else {
          alert("Registration completed successfully!");
          setPage("signin");
        }
      } else {
        const msg = data.message || data.error || "Registration failed";
        alert(`Error: ${msg}`);
        setIsSubmitting(false);
      }
    } catch (err) {
      console.error("Registration error:", err);
      alert("Backend connection failed. Make sure Flask is running on port 5000.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="signup-wrapper">
      <div className="signup-box">
        <h2>Create Typing Identity</h2>
        <p className="subtitle">Type the sentence below exactly as shown</p>

        <div className="sentence-box">{currentSentence}</div>

        <p className="attempt">
          Sentence {attempt} of {TOTAL_ATTEMPTS}
          <span style={{ marginLeft: "20px", color: "#666", fontSize: "14px" }}>
            ({ks.current.keyCount} keys recorded)
          </span>
        </p>

        <input
          type="text"
          placeholder="Type the sentence here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          disabled={isSubmitting}
          autoFocus
        />

        <button onClick={handleSubmit} disabled={isSubmitting}>
          {isSubmitting ? "Saving..." : attempt < TOTAL_ATTEMPTS ? "Next Sentence" : "Complete Registration"}
        </button>
      </div>
    </div>
  );
};

export default TypingProfile;
