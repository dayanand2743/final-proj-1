import axios from "axios";
import React, { useEffect, useState } from "react";

function App() {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:5000/history")
      .then((response) => setLogs(response.data))
      .catch((error) => console.error(error));
  }, []);

  const handlePredict = async () => {
    const sampleData = { features: [1.2, 3.4, 5.6, 7.8] }; // Example input
    const response = await axios.post("http://127.0.0.1:5000/predict", sampleData);
    alert(`Prediction: ${response.data.prediction ? "Anomaly" : "Normal"}`);
  };

  return (
    <div>
      <h1>Network Intrusion Detection</h1>
      <button onClick={handlePredict}>Test Prediction</button>
      <h2>Past Detections</h2>
      <ul>
        {logs.map((log, index) => (
          <li key={index}>{log[1]} → {log[2] === "1" ? "Anomaly" : "Normal"}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;
