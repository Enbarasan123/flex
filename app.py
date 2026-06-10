from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import numpy as np
import joblib
import os
import smtplib

# =========================
# FLASK APP
# =========================

app = Flask(__name__)
CORS(app, origins="*")

# =========================
# MAIL CONFIGURATION
# FIX: Use port 587 + STARTTLS (Gmail standard) instead of SSL 465
# Also added MAIL_DEBUG and timeout to catch errors
# =========================

app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True       # ← FIXED: was False
app.config['MAIL_USE_SSL']        = False      # ← FIXED: was True (conflicts with TLS)
app.config['MAIL_USERNAME']       = 'enbarasan24a@gmail.com'
app.config['MAIL_PASSWORD']       = 'amsxgehluetnygun'   # App Password (no spaces)
app.config['MAIL_DEFAULT_SENDER'] = ('Flex Fitness App', 'enbarasan24a@gmail.com')
app.config['MAIL_MAX_EMAILS']     = None
app.config['MAIL_ASCII_ATTACHMENTS'] = False

mail = Mail(app)


# =========================
# MONGODB CONNECTION
# =========================

client          = MongoClient("mongodb://localhost:27017/")
db              = client["employee_hub"]
users           = db["users"]
user_activities = db["user_activities"]


# =========================
# LOAD ML MODEL
# =========================

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'model', 'calorie_model.joblib')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'gender_encoder.joblib')

calorie_model  = None
gender_encoder = None

try:
    calorie_model  = joblib.load(MODEL_PATH)
    gender_encoder = joblib.load(ENCODER_PATH)
    print("✅ ML Model loaded successfully")
except Exception as e:
    print(f"⚠️  ML Model not loaded: {e}")


# =========================
# HELPER
# =========================

def serialize(doc):
    if doc is None:
        return None
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    for key, value in doc.items():
        if isinstance(value, datetime):
            doc[key] = value.isoformat()
    return doc


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return "✅ Flex Fitness Backend Running!"


# =========================
# TEST MAIL  (GET /test-mail)
# =========================

@app.route("/test-mail")
def test_mail():
    try:
        msg = Message(
            subject    = "Flex — Test Email",
            recipients = ["enbarasan24a@gmail.com"],
            body       = "Flask-Mail is working correctly via STARTTLS (port 587)."
        )
        mail.send(msg)
        return jsonify({"success": True, "message": "Test mail sent successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# =========================
# CALORIE PREDICTION  ← ML
# =========================

@app.route("/predict/calories", methods=["POST"])
def predict_calories():
    if calorie_model is None:
        return jsonify({
            "success": False,
            "message": "ML model not loaded. Run train_model.py first."
        }), 500

    data = request.get_json()

    required = ["gender", "age", "height", "weight", "duration", "heart_rate", "body_temp"]
    missing  = [f for f in required if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing:
        return jsonify({"success": False, "message": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        gender_str = str(data["gender"]).lower().strip()
        if gender_str not in ["male", "female"]:
            return jsonify({"success": False, "message": "Gender must be 'male' or 'female'"}), 400

        gender_enc = gender_encoder.transform([gender_str])[0]

        age        = float(data["age"])
        height     = float(data["height"])
        weight     = float(data["weight"])
        duration   = float(data["duration"])
        heart_rate = float(data["heart_rate"])
        body_temp  = float(data["body_temp"])

        if not (10  <= age        <= 120): return jsonify({"success": False, "message": "Age must be 10–120"}), 400
        if not (100 <= height     <= 250): return jsonify({"success": False, "message": "Height must be 100–250 cm"}), 400
        if not (30  <= weight     <= 300): return jsonify({"success": False, "message": "Weight must be 30–300 kg"}), 400
        if not (1   <= duration   <= 300): return jsonify({"success": False, "message": "Duration must be 1–300 min"}), 400
        if not (40  <= heart_rate <= 220): return jsonify({"success": False, "message": "Heart rate must be 40–220 bpm"}), 400
        if not (35  <= body_temp  <= 43 ): return jsonify({"success": False, "message": "Body temp must be 35–43 °C"}), 400

        features   = np.array([[gender_enc, age, height, weight, duration, heart_rate, body_temp]])
        prediction = calorie_model.predict(features)[0]
        calories   = max(1.0, round(float(prediction), 1))

        return jsonify({"success": True, "calories": calories, "message": f"Predicted {calories} kcal burned"})

    except ValueError as e:
        return jsonify({"success": False, "message": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": f"Prediction error: {str(e)}"}), 500


# =========================
# SIGNUP
# =========================

@app.route("/signup", methods=["POST"])
def signup():
    data     = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"success": False, "message": "All fields are required"}), 400

    if users.find_one({"email": email}):
        return jsonify({"success": False, "message": "User already exists"}), 409

    users.insert_one({"fullName": name, "email": email, "password": password, "createdAt": datetime.utcnow()})
    return jsonify({"success": True, "message": "Signup successful"})


# =========================
# LOGIN
# =========================

@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"success": False, "message": "Email and Password required"}), 400

    user = users.find_one({"email": email, "password": password})
    if user:
        return jsonify({"success": True, "name": user["fullName"], "email": user["email"], "message": "Login successful"})
    return jsonify({"success": False, "message": "Invalid email or password"}), 401


# =========================
# CONTACT  ← FIXED
# Now sends real email via Flask-Mail + returns AI-style response
# =========================

@app.route("/contact", methods=["POST"])
def contact():
    data    = request.get_json()
    name    = data.get("name", "").strip()
    email   = data.get("email", "").strip()
    subject = data.get("subject", "General Inquiry").strip()
    message = data.get("message", "").strip()

    if not name or not email or not message:
        return jsonify({"success": False, "message": "Name, email, and message are required"}), 400

    # ── Send notification to admin ──
    mail_sent = False
    mail_error = ""
    try:
        admin_msg = Message(
            subject    = f"[Flex Contact] {subject}",
            recipients = ["enbarasan24a@gmail.com"],
            reply_to   = email
        )
        admin_msg.html = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:0 auto;background:#0f0f1a;color:#f0f0f0;border-radius:16px;overflow:hidden;">
          <div style="background:linear-gradient(135deg,#ff8c00,#ff4500);padding:24px 28px;">
            <h2 style="margin:0;color:white;font-size:1.3rem;">💬 New Contact Message — Flex App</h2>
          </div>
          <div style="padding:28px;">
            <p><strong>From:</strong> {name} &lt;{email}&gt;</p>
            <p><strong>Subject:</strong> {subject}</p>
            <hr style="border-color:#333;margin:18px 0;">
            <p style="line-height:1.7;white-space:pre-wrap;">{message}</p>
          </div>
          <div style="padding:16px 28px;background:#1a1a2e;font-size:12px;color:#666;">
            Sent via Flex Fitness App Contact Form
          </div>
        </div>
        """
        mail.send(admin_msg)
        mail_sent = True
    except Exception as e:
        mail_error = str(e)
        print(f"⚠️  Mail error: {e}")

    # ── Send confirmation to user ──
    try:
        user_msg = Message(
            subject    = "We received your message — Flex Fitness",
            recipients = [email]
        )
        user_msg.html = f"""
        <div style="font-family:sans-serif;max-width:520px;margin:0 auto;background:#0f0f1a;color:#f0f0f0;border-radius:16px;overflow:hidden;">
          <div style="background:linear-gradient(135deg,#ff8c00,#ff4500);padding:24px 28px;">
            <h2 style="margin:0;color:white;">Hi {name}, we got your message! 💪</h2>
          </div>
          <div style="padding:28px;">
            <p>Thank you for reaching out to Flex. Our support team has received your message and will respond within 24 hours.</p>
            <div style="background:#1a1a2e;border-left:3px solid #ff8c00;border-radius:8px;padding:16px;margin:18px 0;">
              <strong>Your message:</strong><br>
              <span style="color:#aaa;font-size:13px;">{message[:200]}{'...' if len(message)>200 else ''}</span>
            </div>
            <p style="color:#aaa;font-size:13px;">— The Flex Team</p>
          </div>
        </div>
        """
        mail.send(user_msg)
    except Exception as e:
        print(f"⚠️  Confirmation mail error: {e}")

    if mail_sent:
        return jsonify({"success": True, "message": "Message sent successfully! Check your inbox for confirmation."})
    else:
        # Still return success but note the mail issue
        return jsonify({
            "success": True,
            "message": "Message received! (Email delivery pending — check SMTP config)",
            "mail_error": mail_error
        })


# =========================
# LOG ACTIVITY  (supports GPS coordinates)
# =========================

@app.route("/activity/log", methods=["POST"])
def log_activity():
    data          = request.get_json()
    user_email    = data.get("userEmail")
    exercise_type = data.get("exerciseType")

    if not user_email or not exercise_type:
        return jsonify({"success": False, "message": "userEmail and exerciseType are required"}), 400

    if not users.find_one({"email": user_email}):
        return jsonify({"success": False, "message": "User not found"}), 404

    activity = {
        "userEmail"    : user_email,
        "exerciseType" : exercise_type,
        "level"        : data.get("level", ""),
        "distance"     : data.get("distance", ""),
        "steps"        : data.get("steps", ""),
        "calories"     : data.get("calories", ""),
        "burnDetails"  : data.get("burnDetails", {}),
        # ── NEW: GPS route data ──
        "gpsRoute"     : data.get("gpsRoute", []),       # list of {lat,lng,timestamp}
        "startCoords"  : data.get("startCoords", None),  # {lat, lng}
        "endCoords"    : data.get("endCoords", None),    # {lat, lng}
        "avgSpeed"     : data.get("avgSpeed", 0),
        "maxSpeed"     : data.get("maxSpeed", 0),
        "elevationGain": data.get("elevationGain", 0),
        "activityDate" : datetime.utcnow()
    }
    result = user_activities.insert_one(activity)
    return jsonify({"success": True, "message": "Activity logged successfully", "activityId": str(result.inserted_id)})


# =========================
# GET ALL ACTIVITIES
# =========================

@app.route("/activity/<email>", methods=["GET"])
def get_activities(email):
    if not users.find_one({"email": email}):
        return jsonify({"success": False, "message": "User not found"}), 404

    query         = {"userEmail": email}
    exercise_type = request.args.get("exerciseType")
    if exercise_type:
        query["exerciseType"] = exercise_type

    limit  = request.args.get("limit", default=0, type=int)
    cursor = user_activities.find(query).sort("activityDate", -1)
    if limit > 0:
        cursor = cursor.limit(limit)

    activities = [serialize(doc) for doc in cursor]
    return jsonify({"success": True, "count": len(activities), "activities": activities})


# =========================
# GET SINGLE ACTIVITY
# =========================

@app.route("/activity/detail/<activity_id>", methods=["GET"])
def get_activity_detail(activity_id):
    try:
        oid = ObjectId(activity_id)
    except Exception:
        return jsonify({"success": False, "message": "Invalid activity ID"}), 400

    activity = user_activities.find_one({"_id": oid})
    if not activity:
        return jsonify({"success": False, "message": "Activity not found"}), 404

    return jsonify({"success": True, "activity": serialize(activity)})


# =========================
# UPDATE ACTIVITY
# =========================

@app.route("/activity/update/<activity_id>", methods=["PUT"])
def update_activity(activity_id):
    try:
        oid = ObjectId(activity_id)
    except Exception:
        return jsonify({"success": False, "message": "Invalid activity ID"}), 400

    data          = request.get_json()
    allowed_fields= ["exerciseType", "level", "distance", "steps", "calories", "avgSpeed", "maxSpeed"]
    update_data   = {k: v for k, v in data.items() if k in allowed_fields}

    if "burnDetails" in data:
        existing = user_activities.find_one({"_id": oid})
        if existing:
            merged = {**existing.get("burnDetails", {}), **data["burnDetails"]}
            update_data["burnDetails"] = merged

    if not update_data:
        return jsonify({"success": False, "message": "No valid fields to update"}), 400

    result = user_activities.update_one({"_id": oid}, {"$set": update_data})
    if result.matched_count == 0:
        return jsonify({"success": False, "message": "Activity not found"}), 404

    return jsonify({"success": True, "message": "Activity updated successfully"})


# =========================
# DELETE ACTIVITY
# =========================

@app.route("/activity/delete/<activity_id>", methods=["DELETE"])
def delete_activity(activity_id):
    try:
        oid = ObjectId(activity_id)
    except Exception:
        return jsonify({"success": False, "message": "Invalid activity ID"}), 400

    result = user_activities.delete_one({"_id": oid})
    if result.deleted_count == 0:
        return jsonify({"success": False, "message": "Activity not found"}), 404

    return jsonify({"success": True, "message": "Activity deleted successfully"})


# =========================
# ACTIVITY SUMMARY
# =========================

@app.route("/activity/summary/<email>", methods=["GET"])
def activity_summary(email):
    if not users.find_one({"email": email}):
        return jsonify({"success": False, "message": "User not found"}), 404

    total    = user_activities.count_documents({"userEmail": email})
    pipeline = [
        {"$match": {"userEmail": email}},
        {"$group": {"_id": "$exerciseType", "count": {"$sum": 1}}}
    ]
    breakdown = {doc["_id"]: doc["count"] for doc in user_activities.aggregate(pipeline)}
    latest    = user_activities.find_one({"userEmail": email}, sort=[("activityDate", -1)])

    return jsonify({
        "success": True,
        "summary": {
            "totalSessions"  : total,
            "byExerciseType" : breakdown,
            "latestActivity" : serialize(latest)
        }
    })


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(debug=True, port=5000)