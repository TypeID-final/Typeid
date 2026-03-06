import React, { useState, useRef, useEffect, useCallback } from "react";
import "./signin.css";
import { freshState, handleKeyDownEvent, handleKeyUpEvent, buildFeatures } from "../../utils/TypingBiometrics";

const TARGET_SENTENCE = "the quick brown fox jumps over the lazy dog";

// Helper components mapped from App.js:
function MetricBox({val,lbl}) {
  const display = Number.isFinite(val) ? val : String(val ?? "--");
  return (
    <div className="metric-box">
      <div className="metric-val">{display}</div>
      <div className="metric-lbl">{lbl}</div>
    </div>
  );
}

function ResultPanel({ result, username, authSuccess, onContinueClick }) {
  const maxA = result.max_appearances || 21;
  const isMatch = authSuccess;
  
  return (
    <div className="ml-card result-panel">
      <div className="section-label"> 03 / IDENTIFICATION RESULT</div>
      <div className="winner-box" style={!isMatch ? {borderColor: 'var(--warn)', background: 'rgba(255, 71, 87, 0.05)'} : {}}>
        <div className="winner-label">{isMatch ? "SUCCESSFUL IDENTIFICATION" : "AUTHENTICATION FAILED"}</div>
        <div className="winner-name" style={!isMatch ? {color: 'var(--warn)', textShadow: '0 0 30px rgba(255, 71, 87, 0.5)'} : {}}>
          {result.winner?.toUpperCase() || result.predicted_user?.toUpperCase() || 'UNKNOWN'}
        </div>
        <div className="winner-votes">
          Expected user: {username.toUpperCase()}<br/>
          {result.winner_appearances ? `Appeared in ${result.winner_appearances} of ${maxA} prediction slots.` : `Confidence score: ${(result.confidence || result.confidence_pct || 0).toFixed(1)}%`}
        </div>
      </div>
      
      {result.total_models && (
        <>
          <div className="voting-explainer">
            <div className="explainer-title">HOW THE WINNER WAS CHOSEN</div>
            <div className="explainer-body">
              Each of the {result.total_models} models independently predicted its own Top-3 using different features and different training data. The user appearing most across all {maxA} slots wins.
            </div>
          </div>
          <div className="section-label" style={{marginTop:24,marginBottom:8}}>EACH MODEL'S INDEPENDENT TOP-3</div>
          <div className="model-grid">
            {Object.entries(result.per_model_top3 || {}).map(([mname,top3])=>(
              <div className="model-card" key={mname}>
                <div className="model-name">{mname}</div>
                {top3.map((item,i)=>(
                  <div className={`rank-row ${item.user===result.winner?"rank-row-winner":""}`} key={i}>
                    <span className="rank-num">#{i+1}</span>
                    <span className="rank-user">{item.user}</span>
                    <span className="rank-conf">{item.confidence}%</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="section-label" style={{marginTop:24,marginBottom:8}}>VOTE COUNT</div>
          {Object.entries(result.appearance_counts || {}).map(([user,count])=>(
            <div className="vote-row" key={user}>
              <div className={`vote-name ${user===result.winner?"vote-name-winner":""}`}>
                {user===result.winner?" ":""}{user}
              </div>
              <div className="vote-bar-wrap">
                <div className="vote-bar-fill" style={{width:`${(count/maxA)*100}%`}}/>
              </div>
              <div className="vote-count-label">{count}/{maxA}</div>
            </div>
          ))}
        </>
      )}

      {authSuccess && (
        <button className="ml-btn success" onClick={onContinueClick}>
          CONTINUE TO DASHBOARD
        </button>
      )}
      {!authSuccess && (
        <button className="ml-btn reset" style={{width: '100%', marginTop: '20px'}} onClick={onContinueClick}>
          TRY AGAIN
        </button>
      )}
    </div>
  );
}

const MLTypingAPP = ({ username, onLoginSuccess }) => {
  // UI states for ML App
  const [inputVal,   setInputVal]   = useState("");
  const [complete,   setComplete]   = useState(false);
  const [errorFlash, setErrorFlash] = useState(false);
  const [metrics,    setMetrics]    = useState({ dwell:0,flight:0,wpm:0,entropy:0,count:0 });
  const [status,     setStatus]     = useState({ text:"Waiting for input...", type:"" });
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [authResData, setAuthResData] = useState(null);
  
  const textareaRef = useRef(null);
  const progress    = Math.min(100,(inputVal.length/TARGET_SENTENCE.length)*100);

  const ks = useRef(freshState());

  const refreshMetrics = useCallback(() => {
    const s = ks.current;
    if (!s.startTime) return;
    const elapsed = (performance.now() - s.startTime) / 1000;
    const words = inputVal.trim().split(/\s+/).filter(Boolean).length;
    const wpmRaw = elapsed > 0 ? (words / elapsed) * 60 : 0;
    
    // safe compute for entropy since it needs array
    let ent = 0;
    if (s.dwellTimes.length || s.flightTimes.length) {
      const lst = [...s.dwellTimes, ...s.flightTimes];
      if (lst.length > 1) {
        const mn=Math.min(...lst), mx=Math.max(...lst);
        if (mx!==mn) {
          const bins=10, w=(mx-mn)/bins;
          const hist=new Array(bins).fill(0);
          lst.forEach(v=>{ const b=Math.min(bins-1,Math.floor((v-mn)/w)); hist[b]++; });
          ent = -hist.reduce((sum,c)=>{ if(!c) return sum; const p=c/lst.length; return sum+p*Math.log2(p); },0);
        }
      }
    }

    const avg = (a) => a.length ? a.reduce((x,y)=>x+y,0)/a.length : 0;

    setMetrics({
      dwell:  Math.round(avg(s.dwellTimes)) || 0,
      flight: Math.round(avg(s.flightTimes)) || 0,
      wpm:    Math.round(wpmRaw) || 0,
      entropy: ent.toFixed(2),
      count:  s.keyCount,
    });
  }, [inputVal]);

  const handleKeyDown = useCallback((e) => {
    if (loading || result) return;
    handleKeyDownEvent(e, ks.current);
    refreshMetrics();
  }, [refreshMetrics, loading, result]);

  const handleKeyUp = useCallback((e) => {
    if (loading || result) return;
    handleKeyUpEvent(e, ks.current);
    refreshMetrics();
  }, [refreshMetrics, loading, result]);

  const handleChange = useCallback((e) => {
    const val = e.target.value;
    setInputVal(val);
    if (val === TARGET_SENTENCE) {
      setComplete(true);
      setStatus({text:"Complete. Press IDENTIFY USER.",type:"ok"});
    } else {
      setComplete(false);
      if (val !== "" && !TARGET_SENTENCE.startsWith(val)) {
        setErrorFlash(true);
        setTimeout(() => setErrorFlash(false), 300);
      }
      if (status.type === "ok") setStatus({text:"Waiting for input...",type:""});
    }
  }, [status.type]);

  const handleTypingSubmit = async () => {
    if (loading || !complete) return;
    
    const features = buildFeatures(ks.current, inputVal);
    
    if (ks.current.keyCount < 10) {
      alert("Not enough typing data. Please type the full sentence.");
      return;
    }

    setLoading(true);
    setStatus({text:"Analyzing keystroke pattern via ML...",type:"loading"});
    
    try {
      const response = await fetch("http://127.0.0.1:5001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features)
      });

      const data = await response.json();
      setAuthResData(data);
      
      const mlDetails = data; // the 5001 predict endpoint returns the raw dictionary directly
      
      // We check if the predicted winner matches the username they entered 
      if (response.ok && data.winner.toLowerCase() === username.toLowerCase()) {
        setResult(mlDetails);
        setStatus({text:"Identification complete. Access Granted.",type:"ok"});
      } else {
        setResult(mlDetails);
        setStatus({text:" Authentication Failed: Pattern Mismatch ",type:"error"});
      }
    } catch (err) {
      console.error('Login error:', err);
      setStatus({text:" Backend connection failed. Make sure Flask is running.",type:"error"});
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    ks.current = freshState();
    setInputVal(""); setComplete(false); setErrorFlash(false);
    setMetrics({dwell:0,flight:0,wpm:0,entropy:0,count:0});
    setStatus({text:"Waiting for input...",type:""});
    setLoading(false); setResult(null); setAuthResData(null);
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  const handleContinueDashboard = () => {
    if (authResData && authResData.winner.toLowerCase() === username.toLowerCase()) {
      if (typeof onLoginSuccess === 'function') {
        onLoginSuccess(username, 'user');
      }
    } else {
      resetAll();
    }
  };

  useEffect(() => { 
    setTimeout(() => textareaRef.current?.focus(), 100);
  }, []);

  // ML TYPING APP UI
  const inputClass=["typing-input",complete?"complete":"",errorFlash?"error-flash":""].filter(Boolean).join(" ");
  const statusClass=["status-bar",status.type].filter(Boolean).join(" ");

  return (
    <div className="ml-typing-app">
      <div className="ml-card">
        <div className="section-label"> 01 / TYPE THE SENTENCE BELOW (all lowercase)</div>
        <div className="sentence-display">{TARGET_SENTENCE}</div>

        <textarea
          ref={textareaRef}
          className={inputClass}
          rows={2}
          value={inputVal}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          disabled={loading || !!result}
          autoComplete="off" autoCorrect="off" autoCapitalize="off" spellCheck={false}
          placeholder="Start typing here..."
        />
        <div className="progress-bar-wrap">
          <div className="progress-bar-fill" style={{width:`${progress}%`}}/>
        </div>

        <div style={{marginTop:18}}>
          <div className="section-label"> 02 / REAL-TIME BIOMETRICS</div>
          <div className="metrics-grid">
            <MetricBox val={`${metrics.dwell}ms`}  lbl="Dwell Mean"  />
            <MetricBox val={`${metrics.flight}ms`} lbl="Flight Mean" />
            <MetricBox val={metrics.wpm}            lbl="WPM"         />
            <MetricBox val={metrics.entropy}        lbl="Entropy"     />
            <MetricBox val={metrics.count}          lbl="Key Count"   />
          </div>
        </div>

        {!result && (
          <div>
            <button className="ml-btn" disabled={!complete||loading} onClick={handleTypingSubmit}>
              {loading?<><span className="spinner"/>ANALYZING...</>:" IDENTIFY USER"}
            </button>
            <button className="ml-btn reset" onClick={resetAll} disabled={loading}> RESET</button>
          </div>
        )}
        <div className={statusClass}>
          {status.type==="loading" && <span className="spinner"/>}
          {status.text}
        </div>
      </div>

      {result && (
        <ResultPanel 
          result={result} 
          username={username}
          authSuccess={authResData && authResData.winner.toLowerCase() === username.toLowerCase()}
          onContinueClick={handleContinueDashboard} 
        />
      )}
    </div>
  );
};

export default MLTypingAPP;
