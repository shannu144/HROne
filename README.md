# Employee Attendance & Analytics API

A FastAPI + MongoDB service implementing employee attendance tracking, regularization, and analytics reporting.

## Requirements & Setup
- Python 3.11+
- MongoDB 6.0+ (local instance or MongoDB Atlas)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Configuration
The application reads database configuration from environment variables (`.env` supported):
- `MONGO_URI`: MongoDB connection string (default: `mongodb://localhost:27017`)
- `MONGO_DB`: Database name (default: `attendance_db`)

### Seed Sample Data (Optional)
```bash
python sample_seed.py
```

### Run the Application
```bash
uvicorn app.main:app --port 8000
```

## Features & Implementation Notes
- **All application logic in `app/main.py`** as required.
- **Full OpenAPI Contract Compliance**: Exact paths, parameter naming, response structures, and HTTP status codes (200, 201, 404, 409, 422, 503).
- **Business Rules (R1 - R10)**:
  - IST timezone handling with overnight shift date assignment.
  - 10-minute late grace period logic (R2).
  - 30-minute minimum overtime threshold (R3).
  - Half-up rounding for work hours and attendance percentages (R4, R8).
  - Half-day tracking (R5).
  - Working days evaluation (Mon-Fri) with mid-month joiner handling (R7, R9).
- **Concurrency & Races**:
  - Race-safe punch-ins backed by unique database compound index `(emp_code, date)`.
  - Atomic punch-outs via `find_one_and_update`.
  - Optimistic concurrency control on manual regularizations (`PATCH`) to prevent lost audit trails.
- **MongoDB Aggregations & Explain**:
  - Analytics (`employee_monthly`, `department_summary`, `late_leaderboard`, `department_trend`) executed directly inside MongoDB pipelines.
  - Standard competition ranking (`$rank` in `$setWindowFields`) handling ties and limit cutoffs.
  - Calendar gap filling via `$densify` and 7-day moving window average via `$setWindowFields`.
  - All queries backed by indexes with `IXSCAN` plans verified via `/admin/explain/{endpoint}`.
