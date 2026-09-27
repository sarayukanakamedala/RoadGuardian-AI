import { useState } from "react";
import {
  ShieldCheck,
  Upload,
  Video,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Loader2,
  FileVideo,
  Zap,
  Ambulance,
  Siren,
  MapPin,
  Phone,
  Radio,
  Clock3,
} from "lucide-react";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
} from "react-leaflet";

import L from "leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";


// ---------------------------------------------------------
// Backend
// ---------------------------------------------------------

const API_URL = "http://localhost:8000";


// ---------------------------------------------------------
// DEMO ACCIDENT LOCATION
// ---------------------------------------------------------

const DEMO_LOCATION = {
  latitude: 16.5062,
  longitude: 80.6480,
  name: "Vijayawada, Andhra Pradesh",
};


// ---------------------------------------------------------
// Custom map marker
// ---------------------------------------------------------

const accidentIcon = L.divIcon({
  className: "custom-accident-marker",
  html: `
    <div class="accident-marker">
      <div class="marker-pulse"></div>
      <div class="marker-pin">!</div>
    </div>
  `,
  iconSize: [42, 42],
  iconAnchor: [21, 21],
});


// ---------------------------------------------------------
// App
// ---------------------------------------------------------

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [responseStarted, setResponseStarted] = useState(false);
  const [responseLoading, setResponseLoading] = useState(false);


  // -------------------------------------------------------
  // File selection
  // -------------------------------------------------------

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setSelectedFile(file);
    setResult(null);
    setError("");
    setResponseStarted(false);
  };


  // -------------------------------------------------------
  // AI prediction
  // -------------------------------------------------------

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select a video first.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError("");
    setResponseStarted(false);

    const formData = new FormData();

    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData =
          await response.json().catch(() => null);

        throw new Error(
          errorData?.detail ||
          "Prediction request failed."
        );
      }

      const data = await response.json();

      setResult(data);

    } catch (err) {
      console.error(err);

      setError(
        err.message ||
        "Could not connect to the RoadGuardian backend."
      );

    } finally {
      setLoading(false);
    }
  };


  // -------------------------------------------------------
  // Emergency response simulation
  // -------------------------------------------------------

  const handleEmergencyResponse = () => {
    setResponseLoading(true);

    setTimeout(() => {
      setResponseLoading(false);
      setResponseStarted(true);
    }, 1800);
  };


  const isAccident =
    result?.prediction === "ACCIDENT";


  // -------------------------------------------------------
  // UI
  // -------------------------------------------------------

  return (
    <div className="app">

      {/* ===================================================
          HEADER
      =================================================== */}

      <header className="header">

        <div className="brand">

          <div className="brand-icon">
            <ShieldCheck size={28} />
          </div>

          <div>
            <h1>RoadGuardian AI</h1>

            <p>
              AI-Based Road Accident Detection & Response
            </p>
          </div>

        </div>


        <div className="system-status">

          <span className="status-dot"></span>

          System Online

        </div>

      </header>


      {/* ===================================================
          MAIN
      =================================================== */}

      <main className="main-content">

        {/* =================================================
            HERO
        ================================================= */}

        <section className="hero">

          <div>

            <span className="eyebrow">

              <Zap size={15} />

              AI-POWERED ROAD SAFETY

            </span>


            <h2>

              Detect road accidents

              <br />

              <span>
                before it's too late.
              </span>

            </h2>


            <p>

              Upload a road video and let RoadGuardian AI
              analyze the scene using ResNet-18 spatial
              features and GRU temporal analysis.

            </p>

          </div>


          <div className="hero-icon">

            <Activity size={64} />

          </div>

        </section>


        {/* =================================================
            DETECTION DASHBOARD
        ================================================= */}

        <div className="dashboard-grid">


          {/* =================================================
              VIDEO UPLOAD
          ================================================= */}

          <section className="card upload-card">

            <div className="card-header">

              <div>

                <h3>
                  Video Detection
                </h3>

                <p>
                  Upload a road surveillance or dashcam video.
                </p>

              </div>

              <Video size={24} />

            </div>


            <label className="upload-area">

              <input
                type="file"
                accept=".mp4,.avi,.mov,.mkv,video/*"
                onChange={handleFileChange}
              />


              <Upload size={42} />


              <strong>

                {selectedFile
                  ? selectedFile.name
                  : "Choose a video to analyze"}

              </strong>


              <span>

                {selectedFile
                  ? `${(
                    selectedFile.size /
                    (1024 * 1024)
                  ).toFixed(2)} MB`
                  : "MP4, AVI, MOV or MKV"}

              </span>

            </label>


            <button
              className="predict-button"
              onClick={handlePredict}
              disabled={!selectedFile || loading}
            >

              {loading ? (
                <>
                  <Loader2
                    className="spin"
                    size={20}
                  />

                  Analyzing Video...
                </>
              ) : (
                <>
                  <Activity size={20} />

                  Analyze Video
                </>
              )}

            </button>


            {error && (

              <div className="error-message">

                <AlertTriangle size={18} />

                {error}

              </div>

            )}

          </section>


          {/* =================================================
              DETECTION RESULT
          ================================================= */}

          <section className="card result-card">

            <div className="card-header">

              <div>

                <h3>
                  Detection Result
                </h3>

                <p>
                  AI analysis from the RoadGuardian model.
                </p>

              </div>

              <ShieldCheck size={24} />

            </div>


            {!result && !loading && (

              <div className="empty-result">

                <FileVideo size={52} />

                <h4>
                  Waiting for analysis
                </h4>

                <p>
                  Upload a video and click{" "}
                  <strong>
                    Analyze Video
                  </strong>{" "}
                  to receive a prediction.
                </p>

              </div>

            )}


            {loading && (

              <div className="empty-result">

                <Loader2
                  className="spin large"
                  size={52}
                />

                <h4>
                  Analyzing...
                </h4>

                <p>
                  RoadGuardian is processing the
                  video frames.
                </p>

              </div>

            )}


            {result && (

              <div className="result-content">

                <div
                  className={`prediction-banner ${isAccident
                      ? "danger"
                      : "safe"
                    }`}
                >

                  {isAccident ? (
                    <AlertTriangle size={42} />
                  ) : (
                    <CheckCircle2 size={42} />
                  )}


                  <div>

                    <span>
                      DETECTION
                    </span>

                    <strong>

                      {isAccident
                        ? "ACCIDENT DETECTED"
                        : "NORMAL ROAD"}

                    </strong>

                  </div>

                </div>


                <div className="probability-grid">

                  <div className="probability-box">

                    <span>
                      Accident Probability
                    </span>

                    <strong>

                      {(
                        result.accident_probability *
                        100
                      ).toFixed(2)}
                      %

                    </strong>

                  </div>


                  <div className="probability-box">

                    <span>
                      Normal Probability
                    </span>

                    <strong>

                      {(
                        result.normal_probability *
                        100
                      ).toFixed(2)}
                      %

                    </strong>

                  </div>

                </div>


                <div className="result-details">

                  <div>

                    <span>
                      Video
                    </span>

                    <strong>
                      {result.filename}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Threshold
                    </span>

                    <strong>
                      {result.decision_threshold}
                    </strong>

                  </div>


                  <div>

                    <span>
                      Device
                    </span>

                    <strong>
                      {result.device}
                    </strong>

                  </div>

                </div>

              </div>

            )}

          </section>

        </div>


        {/* =================================================
            EMERGENCY RESPONSE
        ================================================= */}

        {result && isAccident && (

          <section className="emergency-section">


            {/* Header */}

            <div className="emergency-header">

              <div className="emergency-title">

                <div className="emergency-icon">

                  <AlertTriangle size={28} />

                </div>


                <div>

                  <span>
                    EMERGENCY RESPONSE
                  </span>

                  <h3>
                    Potential accident detected
                  </h3>

                </div>

              </div>


              <div className="response-status">

                {responseStarted ? (
                  <>
                    <span
                      className="response-dot active"
                    />

                    Response Active
                  </>
                ) : (
                  <>
                    <span
                      className="response-dot"
                    />

                    Awaiting Response
                  </>
                )}

              </div>

            </div>


            <p className="emergency-description">

              RoadGuardian has detected a
              high-confidence accident event.
              The following emergency response
              actions can be simulated from this
              dashboard.

            </p>


            {/* =================================================
                LOCATION MAP
            ================================================= */}

            <div className="location-map-section">

              <div className="location-map-header">

                <div>

                  <span>
                    DETECTION LOCATION
                  </span>

                  <h4>
                    <MapPin size={18} />
                    {DEMO_LOCATION.name}
                  </h4>

                </div>


                <div className="coordinates">

                  <span>
                    LATITUDE
                  </span>

                  <strong>
                    {DEMO_LOCATION.latitude.toFixed(4)}
                  </strong>

                  <span>
                    LONGITUDE
                  </span>

                  <strong>
                    {DEMO_LOCATION.longitude.toFixed(4)}
                  </strong>

                </div>

              </div>


              <div className="map-container">

                <MapContainer
                  center={[
                    DEMO_LOCATION.latitude,
                    DEMO_LOCATION.longitude,
                  ]}
                  zoom={13}
                  scrollWheelZoom={true}
                  className="accident-map"
                >

                  <TileLayer
                    attribution='&copy; OpenStreetMap contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />


                  <Marker
                    position={[
                      DEMO_LOCATION.latitude,
                      DEMO_LOCATION.longitude,
                    ]}
                    icon={accidentIcon}
                  >

                    <Popup>

                      <strong>
                        🚨 Accident Detected
                      </strong>

                      <br />

                      RoadGuardian AI

                      <br />

                      Confidence:{" "}
                      {(
                        result.accident_probability *
                        100
                      ).toFixed(2)}
                      %

                    </Popup>

                  </Marker>

                </MapContainer>

              </div>


              <div className="map-demo-note">

                <MapPin size={16} />

                <span>
                  Demo location for project
                  simulation. Real GPS integration
                  can replace this coordinate.
                </span>

              </div>

            </div>


            {/* =================================================
                EMERGENCY SERVICES
            ================================================= */}

            <div className="emergency-grid">


              <div className="emergency-item">

                <div className="emergency-item-icon ambulance">

                  <Ambulance size={23} />

                </div>


                <div>

                  <span>
                    MEDICAL RESPONSE
                  </span>

                  <strong>
                    Ambulance
                  </strong>

                  <small>
                    Emergency service: 108
                  </small>

                </div>

              </div>


              <div className="emergency-item">

                <div className="emergency-item-icon police">

                  <Siren size={23} />

                </div>


                <div>

                  <span>
                    LAW ENFORCEMENT
                  </span>

                  <strong>
                    Police
                  </strong>

                  <small>
                    Emergency service: 112
                  </small>

                </div>

              </div>


              <div className="emergency-item">

                <div className="emergency-item-icon monitoring">

                  <Radio size={23} />

                </div>


                <div>

                  <span>
                    INCIDENT MONITORING
                  </span>

                  <strong>
                    AI Monitoring
                  </strong>

                  <small>
                    Continuous event tracking
                  </small>

                </div>

              </div>


              <div className="emergency-item">

                <div className="emergency-item-icon location">

                  <MapPin size={23} />

                </div>


                <div>

                  <span>
                    INCIDENT LOCATION
                  </span>

                  <strong>
                    GPS Available
                  </strong>

                  <small>
                    Demo coordinates active
                  </small>

                </div>

              </div>

            </div>


            {/* =================================================
                RESPONSE ACTION
            ================================================= */}

            {!responseStarted ? (

              <div className="response-action">

                <div className="response-note">

                  <Clock3 size={18} />

                  <span>
                    This is a simulation for
                    demonstration. No real emergency
                    service will be contacted.
                  </span>

                </div>


                <button
                  className="emergency-button"
                  onClick={handleEmergencyResponse}
                  disabled={responseLoading}
                >

                  {responseLoading ? (
                    <>
                      <Loader2
                        className="spin"
                        size={20}
                      />

                      Dispatching Response...
                    </>
                  ) : (
                    <>
                      <Phone size={20} />

                      Initiate Emergency Response
                    </>
                  )}

                </button>

              </div>

            ) : (

              <div className="response-success">

                <div className="success-icon">

                  <CheckCircle2 size={28} />

                </div>


                <div>

                  <span>
                    RESPONSE SIMULATION ACTIVE
                  </span>

                  <h4>
                    Emergency response dispatched
                  </h4>

                  <p>
                    Ambulance and police response
                    actions have been simulated
                    successfully.
                  </p>

                </div>


                <div className="dispatch-badge">

                  <CheckCircle2 size={16} />

                  DISPATCHED

                </div>

              </div>

            )}

          </section>

        )}

      </main>


      {/* =====================================================
          FOOTER
      ===================================================== */}

      <footer>

        <span>
          RoadGuardian AI
        </span>

        <span>
          ResNet-18 + GRU
        </span>

        <span>
          AI Road Safety System
        </span>

      </footer>

    </div>
  );
}

export default App;