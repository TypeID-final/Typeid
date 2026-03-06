import React, { useState, useEffect } from "react";
import "./AdminDashboard.css";

const AdminDashboard = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("Dashboard");

  const fetchData = async () => {
    try {
      setLoading(true);
      const res = await fetch("http://127.0.0.1:5000/api/dashboard/admin?role=admin");
      if (!res.ok) throw new Error("Failed to fetch dashboard data");
      const d = await res.json();
      if (d.success) {
        setData(d);
      } else {
        throw new Error(d.message || "Unknown error");
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleDeleteUser = async (userId, username) => {
    if (!window.confirm(`Are you sure you want to permanently delete user '${username}'?`)) return;
    
    try {
      const res = await fetch(`http://127.0.0.1:5000/api/admin/users/${userId}`, {
        method: 'DELETE'
      });
      const result = await res.json();
      if (result.success) {
        alert(result.message);
        fetchData(); // Refresh the data
      } else {
        alert("Failed to delete user: " + result.message);
      }
    } catch (err) {
      console.error(err);
      alert("Error deleting user.");
    }
  };

  if (loading) return <div className="admin-dashboard" style={{color: 'white', padding: '50px'}}>Loading admin data...</div>;
  if (error) return <div className="admin-dashboard" style={{color: 'red', padding: '50px'}}>Error: {error}</div>;

  return (
    <div className="admin-dashboard">
      <aside className="admin-sidebar">
        <h2 className="admin-sidebar-title">Admin Panel</h2>
        <ul className="admin-menu">
          <li className={activeTab === 'Dashboard' ? 'active' : ''} onClick={() => setActiveTab('Dashboard')}>Dashboard</li>
          <li className={activeTab === 'Users' ? 'active' : ''} onClick={() => setActiveTab('Users')}>Users</li>
          <li className={activeTab === 'Auth Logs' ? 'active' : ''} onClick={() => setActiveTab('Auth Logs')}>Auth Logs</li>
          <li className={activeTab === 'Typing Analytics' ? 'active' : ''} onClick={() => setActiveTab('Typing Analytics')}>Typing Analytics</li>
          <li className={activeTab === 'Risk Analysis' ? 'active' : ''} onClick={() => setActiveTab('Risk Analysis')}>Risk Analysis</li>
        </ul>
      </aside>

      <main className="admin-main">
        <h1 className="admin-title">Behavioural Biometrics Dashboard</h1>

        {activeTab === 'Dashboard' && (
          <>
            <div className="stats-grid">
              <div className="stat-card">
                <p className="stat-label">Total Users</p>
                <h2>{data?.total_users || 0}</h2>
              </div>
              <div className="stat-card">
                <p className="stat-label">Biometric Profiles</p>
                <h2>{data?.total_profiles || 0}</h2>
              </div>
              <div className="stat-card">
                <p className="stat-label">Auth Success Rate</p>
                <h2>{data?.activity_logs?.length > 0 ? 
                  Math.round((data.activity_logs.filter(l => l.status === 'success').length / data.activity_logs.length) * 100) + '%' 
                  : 'N/A'}</h2>
              </div>
            </div>

            <div className="table-card" style={{marginTop: '30px'}}>
              <h2 className="table-title">Recent Authentication Attempts</h2>
              <table className="admin-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>User</th>
                    <th>Method</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.activity_logs?.slice(0, 5).map((log, idx) => (
                    <tr key={idx}>
                      <td>{new Date(log.login_time).toLocaleString()}</td>
                      <td>{log.username || 'Unknown'}</td>
                      <td>{log.login_method}</td>
                      <td className={log.status === 'success' ? 'status-success' : 'status-blocked'}>
                        {log.status}
                      </td>
                    </tr>
                  ))}
                  {(!data?.activity_logs || data.activity_logs.length === 0) && (
                    <tr><td colSpan="4" style={{textAlign: 'center', padding: '20px'}}>No recent activity</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}

        {activeTab === 'Users' && (
          <div className="table-card">
            <h2 className="table-title">All Registered Users</h2>
            <table className="admin-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Username</th>
                  <th>Email</th>
                  <th>Joined</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data?.recent_users?.map((user) => (
                  <tr key={user.user_id}>
                    <td>{user.user_id}</td>
                    <td>{user.name}</td>
                    <td>{user.email || 'N/A'}</td>
                    <td>{new Date(user.created_at).toLocaleDateString()}</td>
                    <td>
                      {user.name.toLowerCase() !== 'admin' && (
                        <button 
                          onClick={() => handleDeleteUser(user.user_id, user.name)}
                          style={{
                            background: '#ff4757', 
                            color: 'white', 
                            border: 'none', 
                            padding: '5px 10px', 
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '12px'
                          }}
                        >
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'Auth Logs' && (
          <div className="table-card">
            <h2 className="table-title">Complete Authentication Log History</h2>
            <table className="admin-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>User</th>
                  <th>Method</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {data?.activity_logs?.map((log, idx) => (
                  <tr key={idx}>
                    <td>{new Date(log.login_time).toLocaleString()}</td>
                    <td>{log.username || 'Unknown'}</td>
                    <td>{log.login_method}</td>
                    <td className={log.status === 'success' ? 'status-success' : 'status-blocked'}>
                      {log.status}
                    </td>
                  </tr>
                ))}
                {(!data?.activity_logs || data.activity_logs.length === 0) && (
                  <tr><td colSpan="4" style={{textAlign: 'center', padding: '20px'}}>No recent activity</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'Typing Analytics' && (
          <div className="table-card">
            <h2 className="table-title">Typing Biometric Overview</h2>
            <div style={{ padding: '20px', backgroundColor: '#2a2d3e', borderRadius: '8px', marginTop: '20px' }}>
              <h3 style={{ marginBottom: '15px', color: '#8a92b2' }}>Model Information</h3>
              <p style={{ marginBottom: '10px' }}><strong>Algorithm:</strong> Random Forest Classifier & Statistical Z-Score Matching</p>
              <p style={{ marginBottom: '10px' }}><strong>Tracked Features:</strong> 33 (Including Flight Times, Dwell Times, and Digraphs)</p>
              <p style={{ marginBottom: '10px' }}><strong>Confidence Threshold:</strong> 30%</p>
              <p><strong>Total Registered Profiles:</strong> {data?.total_profiles || 0}</p>
            </div>
            
            <h3 style={{ marginTop: '30px', marginBottom: '15px', color: '#8a92b2' }}>Highest Contributing Profiles</h3>
            <table className="admin-table">
              <thead>
                <tr>
                  <th>User ID</th>
                  <th>Username</th>
                  <th>Total Biometric Samples Logged</th>
                </tr>
              </thead>
              <tbody>
                {data?.top_users?.map((user, idx) => (
                  <tr key={idx}>
                    <td>{user.user_id}</td>
                    <td>{user.username}</td>
                    <td>{user.profile_count} Samples</td>
                  </tr>
                ))}
                {(!data?.top_users || data.top_users.length === 0) && (
                  <tr><td colSpan="3" style={{textAlign: 'center', padding: '20px'}}>No profiles registered yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {activeTab === 'Risk Analysis' && (
          <div className="table-card">
            <h2 className="table-title">Security & Risk Assessment</h2>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
              <div style={{ padding: '20px', backgroundColor: 'rgba(255, 71, 87, 0.1)', border: '1px solid #ff4757', borderRadius: '8px' }}>
                <h3 style={{ color: '#ff4757', marginBottom: '15px' }}>Blocked Authentication Attempts</h3>
                <h1 style={{ fontSize: '3rem', color: 'white' }}>
                  {data?.activity_logs?.filter(l => l.status === 'blocked').length || 0}
                </h1>
                <p style={{ color: '#8a92b2', marginTop: '10px' }}>
                  Unauthorized attempts stopped by the AI model in the recent log history.
                </p>
              </div>
              
              <div style={{ padding: '20px', backgroundColor: 'rgba(46, 213, 115, 0.1)', border: '1px solid #2ed573', borderRadius: '8px' }}>
                <h3 style={{ color: '#2ed573', marginBottom: '15px' }}>Successful Authentications</h3>
                <h1 style={{ fontSize: '3rem', color: 'white' }}>
                  {data?.activity_logs?.filter(l => l.status === 'success').length || 0}
                </h1>
                <p style={{ color: '#8a92b2', marginTop: '10px' }}>
                  Legitimate logins verified by statistical typing matching.
                </p>
              </div>
            </div>

            <h3 style={{ marginTop: '30px', marginBottom: '15px', color: '#ff4757' }}>Recent Blocked Attempts (High Risk)</h3>
            <table className="admin-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Attempted User</th>
                  <th>Method</th>
                </tr>
              </thead>
              <tbody>
                {data?.activity_logs?.filter(l => l.status === 'blocked').slice(0, 5).map((log, idx) => (
                  <tr key={idx} style={{ backgroundColor: 'rgba(255, 71, 87, 0.05)' }}>
                    <td>{new Date(log.login_time).toLocaleString()}</td>
                    <td><strong style={{ color: '#ff4757' }}>{log.username || 'Unknown'}</strong></td>
                    <td>{log.login_method}</td>
                  </tr>
                ))}
                {(!data?.activity_logs || data.activity_logs.filter(l => l.status === 'blocked').length === 0) && (
                  <tr><td colSpan="3" style={{textAlign: 'center', padding: '20px'}}>No blocked attempts recorded</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}

      </main>
    </div>
  );
};

export default AdminDashboard;
