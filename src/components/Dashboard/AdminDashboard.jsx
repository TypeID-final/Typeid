const AdminDashboard = () => {
  return (
    <div className="min-h-screen flex bg-slate-900 text-white">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-800 p-6">
        <h2 className="text-xl font-bold mb-6">Admin Panel</h2>
        <ul className="space-y-3 text-slate-300">
          <li className="hover:text-white cursor-pointer">Dashboard</li>
          <li className="hover:text-white cursor-pointer">Users</li>
          <li className="hover:text-white cursor-pointer">Auth Logs</li>
          <li className="hover:text-white cursor-pointer">Typing Analytics</li>
          <li className="hover:text-white cursor-pointer">Risk Analysis</li>
        </ul>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-8">
        <h1 className="text-3xl font-bold mb-8">Behavioural Biometrics Dashboard</h1>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          <div className="bg-slate-800 p-6 rounded-xl">
            <p className="text-slate-400 text-sm">Auth Success Rate</p>
            <h2 className="text-3xl font-bold mt-2">92%</h2>
          </div>

          <div className="bg-slate-800 p-6 rounded-xl">
            <p className="text-slate-400 text-sm">Failed Attempts</p>
            <h2 className="text-3xl font-bold mt-2">18</h2>
          </div>

          <div className="bg-slate-800 p-6 rounded-xl">
            <p className="text-slate-400 text-sm">Avg Risk Score</p>
            <h2 className="text-3xl font-bold mt-2">Medium</h2>
          </div>
        </div>

        {/* Table */}
        <div className="bg-slate-800 rounded-xl p-6">
          <h2 className="text-xl font-semibold mb-4">Recent Authentication Attempts</h2>

          <table className="w-full text-left text-sm">
            <thead className="text-slate-400 border-b border-slate-700">
              <tr>
                <th className="py-2">User</th>
                <th>Risk</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>

            <tbody>
              <tr className="border-b border-slate-700">
                <td className="py-2">user_01</td>
                <td>Low</td>
                <td className="text-green-400">Success</td>
                <td>10:42 AM</td>
              </tr>

              <tr className="border-b border-slate-700">
                <td className="py-2">user_02</td>
                <td>High</td>
                <td className="text-red-400">Blocked</td>
                <td>10:39 AM</td>
              </tr>

              <tr>
                <td className="py-2">user_03</td>
                <td>Medium</td>
                <td className="text-yellow-400">Review</td>
                <td>10:30 AM</td>
              </tr>
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
};

export default AdminDashboard;
