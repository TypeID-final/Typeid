import { useState } from "react";
import { BrowserRouter as Router } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Background from "./components/Background/background";
import Signup from "./components/Signup/Signup";
import Signin from "./components/Signin/Signin";
import Biometrics from "./components/Signin/Biometrics";
import Dashboard from "./pages/AdminDashboard";
import UserDashboard from "./pages/UserDashboard";
import Home from "./Home/Home";

const App = () => {
  const [page, setPage] = useState("home");
  const [userId, setUserId] = useState(null);
  const [role, setRole] = useState(null);

  const handleLoginSuccess = (uid, urole) => {
    setUserId(uid);
    setRole(urole);
    if (urole === 'admin') {
      setPage("dashboard");
    } else {
      setPage("user-dashboard");
    }
  };

  return (
    <Router>
      <div className="min-h-screen w-full bg-[#0f172a]">
        {page !== "dashboard" && <Navbar setPage={setPage} />}

        <main className="w-full">
          {page === "home" && <Home setPage={setPage} />}
          {page === "background" && <Background setPage={setPage} />}
          {page === "signup" && <Signup setPage={setPage} />}
          
          {page === "signin" && (
            <Signin 
              setPage={setPage} 
              setUserId={setUserId}
              onLoginSuccess={handleLoginSuccess}
            />
          )}

          {page === "biometrics" && (
            <Biometrics 
              setPage={setPage} 
              userId={userId} 
              onLoginSuccess={handleLoginSuccess}
            />
          )}

          {page === "dashboard" && <Dashboard />}
          {page === "user-dashboard" && <UserDashboard username={userId} />}
        </main>
      </div>
    </Router>
  );
};

export default App;
