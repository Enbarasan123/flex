# Flex Fitness Backend

A scalable Flask-based REST API powering the Flex Fitness platform. The backend provides user authentication, AI-powered calorie prediction, workout activity tracking, GPS route logging, email notifications, and MongoDB integration.

## Overview

Flex Fitness Backend is designed to help users monitor fitness activities, track workout performance, and estimate calorie burn using Machine Learning. The system exposes RESTful APIs that can be integrated with web and mobile fitness applications.

## Features

### Authentication
- User Registration
- User Login
- User Validation

### AI-Powered Calorie Prediction
- Machine Learning calorie estimation
- Gender-aware prediction model
- Real-time calorie burn calculation

### Activity Tracking
- Exercise logging
- Distance tracking
- Step tracking
- Calorie tracking
- Workout history

### GPS Route Tracking
- Start location tracking
- End location tracking
- Route storage
- Average speed calculation
- Maximum speed calculation
- Elevation gain monitoring

### Analytics
- Activity summaries
- Exercise type breakdown
- Recent activity insights

### Communication
- Contact form API
- Automated email notifications
- User confirmation emails

---

## Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Backend Development |
| Flask | REST API Framework |
| MongoDB | Database |
| NumPy | Data Processing |
| Joblib | Model Serialization |
| Flask-Mail | Email Services |
| Flask-CORS | Cross-Origin Support |
| Machine Learning | Calorie Prediction |

---

## Project Structure

```bash
Flex-Fitness-Backend/
│
├── model/
│   ├── calorie_model.joblib
│   └── gender_encoder.joblib
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/flex-fitness-backend.git

cd flex-fitness-backend
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## MongoDB Configuration

Start MongoDB server:

```bash
mongod
```

Default Database:

```text
employee_hub
```

Collections:

```text
users
user_activities
```

---

## Run Application

```bash
python app.py
```

Server runs on:

```text
http://localhost:5000
```

---

## API Endpoints

### Authentication

| Method | Endpoint |
|----------|----------|
| POST | /signup |
| POST | /login |

### Calorie Prediction

| Method | Endpoint |
|----------|----------|
| POST | /predict/calories |

### Contact

| Method | Endpoint |
|----------|----------|
| POST | /contact |
| GET | /test-mail |

### Activities

| Method | Endpoint |
|----------|----------|
| POST | /activity/log |
| GET | /activity/<email> |
| GET | /activity/detail/<id> |
| PUT | /activity/update/<id> |
| DELETE | /activity/delete/<id> |
| GET | /activity/summary/<email> |

---

## Sample Calorie Prediction Request

```json
{
  "gender": "male",
  "age": 24,
  "height": 175,
  "weight": 70,
  "duration": 45,
  "heart_rate": 120,
  "body_temp": 37.5
}
```

### Response

```json
{
  "success": true,
  "calories": 320.5
}

## Security Improvements Planned

- JWT Authentication
- Password Hashing
- Role-Based Access Control
- API Rate Limiting
- Environment Variables for Secrets
## Future Enhancements

- AI Workout Recommendation System
- Personalized Diet Suggestions
- Fitness Dashboard
- Real-Time Tracking
- Wearable Device Integration
- OpenAI Fitness Assistant
- Social Fitness Challenges

---
## Author

**Enbarasan**

Data Analytics & AI Enthusiast


This project is licensed under the MIT License.
