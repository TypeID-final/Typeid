import React from 'react';

const UserDashboard = ({ username }) => {
  return (
    <div style={{
      minHeight: 'calc(100vh - 80px)', // account for navbar
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: 'center',
      backgroundColor: '#0f172a',
      color: 'white'
    }}>
      <div style={{ 
        backgroundColor: 'rgba(255, 255, 255, 0.05)', 
        padding: '3rem 5rem', 
        borderRadius: '12px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        textAlign: 'center',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.5)'
      }}>
        <h1 style={{ fontSize: '3rem', marginBottom: '10px' }}>
          Welcome, <span style={{ color: '#00ff9f' }}>{(username || 'User').toUpperCase()}</span>!
        </h1>
        <p style={{ fontSize: '1.2rem', color: '#c8e6f0' }}>You have successfully verified your behavioral identity.</p>
      </div>
    </div>
  );
};

export default UserDashboard;
