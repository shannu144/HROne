"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import calendar
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))


def init_db():
    # Employees indexes
    db.employees.create_index([("emp_code", 1)], unique=True)
    db.employees.create_index([("department", 1), ("emp_code", 1)])
    db.employees.create_index([("department", 1), ("joined_on", 1)])
    db.employees.create_index([("joined_on", 1), ("department", 1)])

    # Attendance logs indexes
    db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
    db.attendance_logs.create_index([("date", -1), ("emp_code", 1)])
    db.attendance_logs.create_index([("emp_code", 1), ("punch_in", -1)])
    db.attendance_logs.create_index([("date", 1), ("late_minutes", 1), ("emp_code", 1)])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# Initialize indexes on startup
init_db()

app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers & Rules (R1 - R8)
# --------------------------------------------------------------------------- #
def round_half_up(val: Optional[float | Decimal], decimals: int = 2) -> Optional[float]:
    if val is None:
        return None
    d = Decimal(str(val))
    target = Decimal("10") ** -decimals
    return float(d.quantize(target, rounding=ROUND_HALF_UP))


def to_epoch_ms(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return int(val.timestamp() * 1000)
    return None


def get_attendance_date(punch_in_ms: int, shift_start: str, shift_end: str) -> str:
    """R1: Determine attendance date in IST, accounting for overnight shifts."""
    dt_utc = datetime.fromtimestamp(punch_in_ms / 1000, tz=timezone.utc)
    dt_ist = dt_utc.astimezone(IST)

    sh_h, sh_m = map(int, shift_start.split(":"))
    eh_h, eh_m = map(int, shift_end.split(":"))

    is_overnight = (eh_h, eh_m) <= (sh_h, sh_m)
    if is_overnight:
        punch_time = (dt_ist.hour, dt_ist.minute, dt_ist.second)
        shift_end_time = (eh_h, eh_m, 0)
        if punch_time < shift_end_time:
            return (dt_ist.date() - timedelta(days=1)).isoformat()

    return dt_ist.date().isoformat()


def compute_late_minutes(punch_in_ms: int, shift_start: str, attendance_date_str: str) -> int:
    """R2: Grace period 10 mins. Late if strictly > 10:00 after shift_start."""
    dt_utc = datetime.fromtimestamp(punch_in_ms / 1000, tz=timezone.utc)
    dt_ist = dt_utc.astimezone(IST)

    sh_h, sh_m = map(int, shift_start.split(":"))
    att_date = datetime.fromisoformat(attendance_date_str).date()
    shift_start_ist = datetime(att_date.year, att_date.month, att_date.day, sh_h, sh_m, 0, tzinfo=IST)

    diff_seconds = (dt_ist - shift_start_ist).total_seconds()
    if diff_seconds > 600:
        return int(diff_seconds // 60)
    return 0


def compute_overtime(punch_out_ms: int, shift_start: str, shift_end: str, attendance_date_str: str) -> int:
    """R3: Whole minutes between shift_end and punch_out, only if >= 30, else 0."""
    dt_utc = datetime.fromtimestamp(punch_out_ms / 1000, tz=timezone.utc)
    dt_ist = dt_utc.astimezone(IST)

    sh_h, sh_m = map(int, shift_start.split(":"))
    eh_h, eh_m = map(int, shift_end.split(":"))

    att_date = datetime.fromisoformat(attendance_date_str).date()
    is_overnight = (eh_h, eh_m) <= (sh_h, sh_m)
    if is_overnight:
        shift_end_date = att_date + timedelta(days=1)
    else:
        shift_end_date = att_date

    shift_end_ist = datetime(shift_end_date.year, shift_end_date.month, shift_end_date.day, eh_h, eh_m, 0, tzinfo=IST)
    diff_seconds = (dt_ist - shift_end_ist).total_seconds()
    minutes = int(diff_seconds // 60)
    return minutes if minutes >= 30 else 0


def compute_work_hours(punch_in_ms: int, punch_out_ms: int) -> float:
    """R4: (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals half-up."""
    diff_seconds = (punch_out_ms - punch_in_ms) // 1000
    hours = (Decimal(diff_seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(hours)


# --------------------------------------------------------------------------- #
# Schemas & Validation
# --------------------------------------------------------------------------- #
EpochMillis = Annotated[int, Field(ge=100000000000, le=4102444800000, strict=True)]


class Status(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class PresenceStatus(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class EmployeeCreate(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")

    @model_validator(mode="after")
    def validate_shift_and_date(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        try:
            date.fromisoformat(self.joined_on)
        except ValueError:
            raise ValueError("Invalid joined_on date")
        return self


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMillis] = None
    status: PresenceStatus = PresenceStatus.PRESENT


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMillis] = None


class RegularizeRequest(BaseModel):
    status: Optional[Status] = None
    punch_in: Optional[EpochMillis] = None
    punch_out: Optional[EpochMillis] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


def serialize_employee_doc(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_epoch_ms(doc.get("created_at")),
    }


def serialize_history_entry(entry: dict) -> dict:
    changes = {}
    for k, v in entry.get("changes", {}).items():
        if k in ("punch_in", "punch_out"):
            changes[k] = {
                "from": to_epoch_ms(v.get("from")),
                "to": to_epoch_ms(v.get("to")),
            }
        else:
            changes[k] = {
                "from": v.get("from"),
                "to": v.get("to"),
            }
    return {
        "at": to_epoch_ms(entry.get("at")),
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": changes,
    }


def serialize_attendance_doc(doc: dict) -> dict:
    history = [serialize_history_entry(h) for h in doc.get("history", [])]
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_epoch_ms(doc.get("punch_in")),
        "punch_out": to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    doc = body.model_dump()
    doc["created_at"] = created_at
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee_doc(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    q = {}
    if department:
        q["department"] = department
    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.employees.find(q).sort([("emp_code", 1)]).skip(skip).limit(page_size)
    items = [serialize_employee_doc(doc) for doc in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")

    if body.punched_at is not None:
        raw_ms = (body.punched_at // 1000) * 1000
    else:
        now_dt = datetime.now(timezone.utc).replace(microsecond=0)
        raw_ms = int(now_dt.timestamp() * 1000)

    punch_in_dt = datetime.fromtimestamp(raw_ms / 1000, tz=timezone.utc)
    attendance_date = get_attendance_date(raw_ms, emp["shift_start"], emp["shift_end"])
    late_minutes = compute_late_minutes(raw_ms, emp["shift_start"], attendance_date)

    doc = {
        "emp_code": body.emp_code,
        "date": attendance_date,
        "status": body.status.value,
        "punch_in": punch_in_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")

    return serialize_attendance_doc(doc)


@app.post("/attendance/punch-out", status_code=200)
def punch_out(body: PunchOutRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")

    if body.punched_at is not None:
        punch_out_ms = (body.punched_at // 1000) * 1000
    else:
        now_dt = datetime.now(timezone.utc).replace(microsecond=0)
        punch_out_ms = int(now_dt.timestamp() * 1000)

    punch_out_dt = datetime.fromtimestamp(punch_out_ms / 1000, tz=timezone.utc)

    # Find the most recent record whose punch_in <= punched_at
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punch_out_dt}},
        sort=[("punch_in", -1)],
    )
    if not record:
        raise HTTPException(404, "No punch-in found")

    if record.get("punch_out") is not None:
        raise HTTPException(409, "Record already punched out")

    punch_in_ms = to_epoch_ms(record["punch_in"])
    if punch_out_ms <= punch_in_ms:
        raise HTTPException(422, "punched_at must be after punch_in")
    if (punch_out_ms - punch_in_ms) > 24 * 3600 * 1000:
        raise HTTPException(422, "punched_at cannot be more than 24 hours after punch_in")

    work_hours = compute_work_hours(punch_in_ms, punch_out_ms)
    overtime_minutes = compute_overtime(punch_out_ms, emp["shift_start"], emp["shift_end"], record["date"])
    half_day = bool(work_hours < 4.50)

    # Atomic update to avoid race conditions
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hours,
                "overtime_minutes": overtime_minutes,
                "half_day": half_day,
            }
        },
        return_document=True,
    )
    if not updated:
        raise HTTPException(409, "Record already punched out")

    return serialize_attendance_doc(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from cannot be greater than date_to")

    q = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status.value

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.attendance_logs.find(q).sort([("date", -1), ("emp_code", 1)]).skip(skip).limit(page_size)
    items = [serialize_attendance_doc(doc) for doc in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date_str}")
def regularize_attendance(emp_code: str, date_str: str, body: RegularizeRequest):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")

    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date_str})
    if not rec:
        raise HTTPException(404, "Attendance record not found")

    old_status = rec.get("status")
    old_punch_in = rec.get("punch_in")
    old_punch_out = rec.get("punch_out")
    old_work_hours = rec.get("work_hours")
    old_late_minutes = rec.get("late_minutes", 0)
    old_overtime_minutes = rec.get("overtime_minutes", 0)
    old_half_day = rec.get("half_day", False)
    old_history = rec.get("history", [])

    new_status = body.status.value if body.status is not None else old_status

    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "Cannot supply punch times when status is ABSENT or LEAVE")
        new_punch_in = None
        new_punch_out = None
        new_work_hours = None
        new_late_minutes = 0
        new_overtime_minutes = 0
        new_half_day = False
    else:
        # Presence status
        if body.punch_in is not None:
            new_punch_in_ms = (body.punch_in // 1000) * 1000
            new_punch_in = datetime.fromtimestamp(new_punch_in_ms / 1000, tz=timezone.utc)
        else:
            new_punch_in = old_punch_in
            new_punch_in_ms = to_epoch_ms(old_punch_in)

        if new_punch_in is None:
            raise HTTPException(422, "A presence status requires a punch_in")

        # Verify punch_in stays on record's attendance date per R1
        calc_date = get_attendance_date(new_punch_in_ms, emp["shift_start"], emp["shift_end"])
        if calc_date != date_str:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")

        if body.punch_out is not None:
            new_punch_out_ms = (body.punch_out // 1000) * 1000
            new_punch_out = datetime.fromtimestamp(new_punch_out_ms / 1000, tz=timezone.utc)
        else:
            new_punch_out = old_punch_out
            new_punch_out_ms = to_epoch_ms(old_punch_out)

        if new_punch_out is not None:
            if new_punch_out_ms <= new_punch_in_ms:
                raise HTTPException(422, "punch_out must be after punch_in")
            if (new_punch_out_ms - new_punch_in_ms) > 24 * 3600 * 1000:
                raise HTTPException(422, "punch_out cannot be more than 24 hours after punch_in")

            new_work_hours = compute_work_hours(new_punch_in_ms, new_punch_out_ms)
            new_overtime_minutes = compute_overtime(
                new_punch_out_ms, emp["shift_start"], emp["shift_end"], date_str
            )
            new_half_day = bool(new_work_hours < 4.50)
        else:
            new_work_hours = None
            new_overtime_minutes = 0
            new_half_day = False

        new_late_minutes = compute_late_minutes(new_punch_in_ms, emp["shift_start"], date_str)

    # Detect actual changes
    changes = {}
    if old_status != new_status:
        changes["status"] = {"from": old_status, "to": new_status}
    if to_epoch_ms(old_punch_in) != to_epoch_ms(new_punch_in):
        changes["punch_in"] = {"from": old_punch_in, "to": new_punch_in}
    if to_epoch_ms(old_punch_out) != to_epoch_ms(new_punch_out):
        changes["punch_out"] = {"from": old_punch_out, "to": new_punch_out}
    if old_work_hours != new_work_hours:
        changes["work_hours"] = {"from": old_work_hours, "to": new_work_hours}
    if old_late_minutes != new_late_minutes:
        changes["late_minutes"] = {"from": old_late_minutes, "to": new_late_minutes}
    if old_overtime_minutes != new_overtime_minutes:
        changes["overtime_minutes"] = {"from": old_overtime_minutes, "to": new_overtime_minutes}
    if old_half_day != new_half_day:
        changes["half_day"] = {"from": old_half_day, "to": new_half_day}

    if not changes:
        raise HTTPException(422, "A request that changes nothing is rejected")

    history_entry = {
        "at": datetime.now(timezone.utc).replace(microsecond=0),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    # Optimistic concurrency control on history length
    res = db.attendance_logs.update_one(
        {
            "_id": rec["_id"],
            "history": {"$size": len(old_history)},
        },
        {
            "$set": {
                "status": new_status,
                "punch_in": new_punch_in,
                "punch_out": new_punch_out,
                "work_hours": new_work_hours,
                "late_minutes": new_late_minutes,
                "overtime_minutes": new_overtime_minutes,
                "half_day": new_half_day,
            },
            "$push": {"history": history_entry},
        },
    )
    if res.matched_count == 0:
        raise HTTPException(409, "Concurrent regularizations conflict")

    updated_rec = db.attendance_logs.find_one({"_id": rec["_id"]})
    return serialize_attendance_doc(updated_rec)


# --------------------------------------------------------------------------- #
# Analytics Pipelines
# --------------------------------------------------------------------------- #
def get_month_range(month_str: str) -> tuple[date, date, int]:
    try:
        y, m = map(int, month_str.split("-"))
        _, last_day = calendar.monthrange(y, m)
        return date(y, m, 1), date(y, m, last_day), last_day
    except Exception:
        raise HTTPException(422, "Invalid month format")


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str,
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")

    m_start, m_end, last_day = get_month_range(month)
    joined_date = date.fromisoformat(emp["joined_on"])

    # Count working days (Mon-Fri) from max(m_start, joined_date) to m_end
    start_eval = max(m_start, joined_date)
    working_days = 0
    if start_eval <= m_end:
        cur = start_eval
        while cur <= m_end:
            if cur.weekday() < 5:
                working_days += 1
            cur += timedelta(days=1)

    start_date_str = m_start.isoformat()
    end_date_str = m_end.isoformat()

    pipeline = [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": start_date_str, "$lte": end_date_str},
            }
        },
        {
            "$addFields": {
                "day_of_week": {
                    "$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]},
                                ]
                            },
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "total_overtime_minutes": {
                    "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                },
            }
        },
    ]

    agg_res = list(db.attendance_logs.aggregate(pipeline))
    if agg_res:
        row = agg_res[0]
        present_days = float(row.get("present_days", 0.0))
        leave_days = int(row.get("leave_days", 0))
        late_count = int(row.get("late_count", 0))
        total_late_minutes = int(row.get("total_late_minutes", 0))
        total_overtime_minutes = int(row.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late_minutes = 0
        total_overtime_minutes = 0

    attendance_pct = None
    if working_days > 0:
        attendance_pct = round_half_up(
            (Decimal(str(present_days)) / Decimal(str(working_days))) * Decimal(100), 2
        )

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


def build_department_summary_pipeline(month: str, department: Optional[str] = None) -> list:
    m_start, m_end, _ = get_month_range(month)
    start_date_str = m_start.isoformat()
    end_date_str = m_end.isoformat()

    match_stage = {"joined_on": {"$lte": end_date_str}}
    if department:
        match_stage["department"] = department

    pipeline = [
        {"$match": match_stage},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$gte": ["$date", start_date_str]},
                                    {"$lte": ["$date", end_date_str]},
                                ]
                            }
                        }
                    },
                    {
                        "$addFields": {
                            "day_of_week": {
                                "$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                            }
                        }
                    },
                ],
                "as": "logs",
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "all_logs": {"$push": "$logs"},
            }
        },
        {
            "$project": {
                "department": "$_id",
                "headcount": 1,
                "logs": {
                    "$reduce": {
                        "input": "$all_logs",
                        "initialValue": [],
                        "in": {"$concatArrays": ["$$value", "$$this"]},
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": 1,
                "headcount": 1,
                "present_days": {
                    "$sum": {
                        "$map": {
                            "input": "$logs",
                            "as": "l",
                            "in": {
                                "$cond": [
                                    {
                                        "$and": [
                                            {"$in": ["$$l.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                            {"$in": ["$$l.day_of_week", [2, 3, 4, 5, 6]]},
                                        ]
                                    },
                                    {"$cond": [{"$eq": ["$$l.half_day", True]}, 0.5, 1.0]},
                                    0.0,
                                ]
                            },
                        }
                    }
                },
                "valid_work_hours": {
                    "$filter": {
                        "input": "$logs",
                        "as": "l",
                        "cond": {
                            "$and": [
                                {"$in": ["$$l.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                {"$ne": ["$$l.work_hours", None]},
                            ]
                        },
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": ["$$l.late_minutes", 0]},
                        }
                    }
                },
                "total_late_minutes": {
                    "$sum": {
                        "$map": {
                            "input": "$logs",
                            "as": "l",
                            "in": {"$ifNull": ["$$l.late_minutes", 0]},
                        }
                    }
                },
                "leave_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "LEAVE"]},
                        }
                    }
                },
                "on_duty_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "ON_DUTY"]},
                        }
                    }
                },
            }
        },
        {
            "$project": {
                "department": 1,
                "headcount": 1,
                "present_days": 1,
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": [{"$size": "$valid_work_hours"}, 0]},
                        {"$avg": "$valid_work_hours.work_hours"},
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": 1}},
    ]
    return pipeline


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    pipeline = build_department_summary_pipeline(month, department)
    docs = list(db.employees.aggregate(pipeline))
    items = []
    for d in docs:
        avg_wh = d.get("avg_work_hours")
        items.append({
            "department": d["department"],
            "headcount": int(d["headcount"]),
            "present_days": float(d.get("present_days", 0.0)),
            "avg_work_hours": round_half_up(avg_wh, 2) if avg_wh is not None else None,
            "late_count": int(d.get("late_count", 0)),
            "total_late_minutes": int(d.get("total_late_minutes", 0)),
            "leave_count": int(d.get("leave_count", 0)),
            "on_duty_count": int(d.get("on_duty_count", 0)),
        })
    return {"month": month, "items": items}


def build_leaderboard_pipeline(month: str, limit: int = 10, department: Optional[str] = None) -> list:
    m_start, m_end, _ = get_month_range(month)
    pipeline = [
        {
            "$match": {
                "date": {"$gte": m_start.isoformat(), "$lte": m_end.isoformat()},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {
                    "rank": {"$rank": {}},
                },
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {
            "$sort": {
                "total_late_minutes": -1,
                "_id": 1,
            }
        },
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ])
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = None,
):
    pipeline = build_leaderboard_pipeline(month, limit, department)
    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


def build_department_trend_pipeline(department: str, from_date: str, to_date: str) -> list:
    f_dt = date.fromisoformat(from_date)
    t_dt = date.fromisoformat(to_date)
    if t_dt < f_dt:
        raise HTTPException(422, "to cannot be earlier than from")
    if (t_dt - f_dt).days + 1 > 92:
        raise HTTPException(422, "Range cannot exceed 92 days")

    dept_emp_cursor = db.employees.find({"department": department}, {"emp_code": 1})
    emp_codes = [e["emp_code"] for e in dept_emp_cursor]

    from_dt_utc = datetime(f_dt.year, f_dt.month, f_dt.day, tzinfo=timezone.utc)
    to_dt_utc = datetime(t_dt.year, t_dt.month, t_dt.day, tzinfo=timezone.utc) + timedelta(days=1)

    pipeline = [
        {
            "$match": {
                "emp_code": {"$in": emp_codes},
                "date": {"$gte": from_date, "$lte": to_date},
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date_str": "$_id",
                "as_date": {"$dateFromString": {"dateString": "$_id"}},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$densify": {
                "field": "as_date",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [from_dt_utc, to_dt_utc],
                },
            }
        },
        {
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$as_date"}},
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
            }
        },
        {"$sort": {"date": 1}},
        {
            "$lookup": {
                "from": "employees",
                "let": {"d": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$d"]},
                                ]
                            }
                        }
                    },
                    {"$count": "count"},
                ],
                "as": "hc",
            }
        },
        {
            "$addFields": {
                "headcount": {"$ifNull": [{"$first": "$hc.count"}, 0]},
                "day_of_week": {
                    "$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                },
            }
        },
        {
            "$addFields": {
                "is_working_day": {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]}
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                {"$eq": ["$is_working_day", True]},
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {
                            "$round": [
                                {"$divide": ["$present_count", "$headcount"]},
                                4,
                            ]
                        },
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {
                            "documents": [-6, "current"],
                        },
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": 1,
            }
        },
    ]
    return pipeline


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: str = Query(alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    if not db.employees.find_one({"department": department}):
        raise HTTPException(404, "Department not found")

    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    raw_items = list(db.attendance_logs.aggregate(pipeline))

    # Apply R8 4-decimal half-up rounding
    items = []
    for row in raw_items:
        items.append({
            "date": row["date"],
            "is_working_day": bool(row["is_working_day"]),
            "headcount": int(row["headcount"]),
            "present_count": float(row.get("present_count", 0.0)),
            "late_count": int(row.get("late_count", 0)),
            "attendance_rate": round_half_up(row.get("attendance_rate"), 4),
            "moving_avg_7d": round_half_up(row.get("moving_avg_7d"), 4),
        })

    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Admin Explain Endpoint
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_date: Optional[str] = Query(default=None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(default=None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from > date_to")
        q = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status.value
        skip = (page - 1) * page_size
        explain = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": q,
                "sort": {"date": -1, "emp_code": 1},
                "skip": skip,
                "limit": page_size,
            },
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(422, "emp_code and month are required for employee_monthly")
        m_start, m_end, _ = get_month_range(month)
        pipeline = [
            {
                "$match": {
                    "emp_code": emp_code,
                    "date": {"$gte": m_start.isoformat(), "$lte": m_end.isoformat()},
                }
            },
            {
                "$addFields": {
                    "day_of_week": {
                        "$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "present_days": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                        {"$in": ["$day_of_week", [2, 3, 4, 5, 6]]},
                                    ]
                                },
                                {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                                0.0,
                            ]
                        }
                    },
                    "leave_days": {
                        "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                    },
                    "late_count": {
                        "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                    },
                    "total_late_minutes": {
                        "$sum": {"$ifNull": ["$late_minutes", 0]}
                    },
                    "total_overtime_minutes": {
                        "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                    },
                }
            },
        ]
        explain = db.command(
            "explain",
            {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(422, "month is required for department_summary")
        pipeline = build_department_summary_pipeline(month, department)
        explain = db.command(
            "explain",
            {"aggregate": "employees", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "employees", "explain": explain}

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(422, "month is required for late_leaderboard")
        pipeline = build_leaderboard_pipeline(month, limit, department)
        explain = db.command(
            "explain",
            {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(422, "department, from, and to are required for department_trend")
        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        explain = db.command(
            "explain",
            {"aggregate": "attendance_logs", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    else:
        raise HTTPException(422, f"Unknown endpoint '{endpoint}'")


# --------------------------------------------------------------------------- #
# Interactive Executive Dashboard (UI)
# --------------------------------------------------------------------------- #
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HROne Intelligence | Attendance Analytics Console</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script>tailwind.config={theme:{extend:{fontFamily:{sans:['"Plus Jakarta Sans"','sans-serif']}}}}</script>
  <style>
    body{font-family:'Plus Jakarta Sans',sans-serif;background:#080c16;color:#f1f5f9}
    .glass{background:rgba(15,23,42,.80);backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.07);transition:border-color .2s}
    .glass:hover{border-color:rgba(99,102,241,.28)}
    .tab-active{color:#818cf8!important;background:rgba(99,102,241,.12)!important;border-color:rgba(99,102,241,.35)!important}
    ::-webkit-scrollbar{width:5px;height:5px}
    ::-webkit-scrollbar-track{background:#0f172a}
    ::-webkit-scrollbar-thumb{background:#334155;border-radius:4px}
    @keyframes pulsing{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
    .pdot{width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;animation:pulsing 2s ease-in-out infinite}
  </style>
</head>
<body class="min-h-screen flex flex-col">

<!-- HEADER -->
<header class="sticky top-0 z-50 border-b border-slate-800 bg-slate-900/90 backdrop-blur">
  <div class="max-w-screen-xl mx-auto px-5 h-16 flex items-center justify-between gap-4">
    <div class="flex items-center gap-3 shrink-0">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-600/30">
        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
      </div>
      <div>
        <span class="text-lg font-extrabold bg-gradient-to-r from-white via-indigo-200 to-indigo-400 bg-clip-text text-transparent">HROne Intelligence</span>
        <span class="ml-2 text-[11px] font-semibold px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">v2.0</span>
      </div>
    </div>
    <div class="hidden md:flex items-center gap-3">
      <div class="flex items-center gap-2 text-xs bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-3 py-1.5 rounded-full">
        <span class="pdot"></span><span>MongoDB Connected</span>
      </div>
      <div class="flex items-center gap-2 text-xs bg-slate-800 text-slate-300 px-3 py-1.5 rounded-full border border-slate-700">
        <svg class="w-3.5 h-3.5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        <span id="clk" class="font-mono text-[11px]">--:--:-- IST</span>
      </div>
    </div>
    <div class="flex items-center gap-2 shrink-0">
      <a href="/docs" target="_blank" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:bg-slate-700 flex items-center gap-1.5 transition">Swagger UI <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/></svg></a>
      <button onclick="go()" class="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold shadow shadow-indigo-600/30 flex items-center gap-1.5 transition">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
        Refresh All
      </button>
    </div>
  </div>
</header>

<main class="max-w-screen-xl mx-auto px-5 py-6 flex-1 w-full">

  <!-- TABS -->
  <div class="flex items-center gap-1.5 border-b border-slate-800 mb-6 pb-1 overflow-x-auto text-xs">
    <button onclick="tab('ov')"  id="t-ov"  class="tab-btn tab-active whitespace-nowrap px-3.5 py-2 rounded-lg border border-transparent font-medium flex items-center gap-1.5">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/></svg>
      Executive Overview
    </button>
    <button onclick="tab('pu')" id="t-pu" class="tab-btn text-slate-400 hover:text-white hover:bg-slate-800/60 whitespace-nowrap px-3.5 py-2 rounded-lg border border-transparent font-medium flex items-center gap-1.5">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
      Live Punch Station
    </button>
    <button onclick="tab('at')" id="t-at" class="tab-btn text-slate-400 hover:text-white hover:bg-slate-800/60 whitespace-nowrap px-3.5 py-2 rounded-lg border border-transparent font-medium flex items-center gap-1.5">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
      Attendance & Audit
    </button>
    <button onclick="tab('an')" id="t-an" class="tab-btn text-slate-400 hover:text-white hover:bg-slate-800/60 whitespace-nowrap px-3.5 py-2 rounded-lg border border-transparent font-medium flex items-center gap-1.5">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"/></svg>
      Trend &amp; Leaderboard
    </button>
    <button onclick="tab('mo')" id="t-mo" class="tab-btn text-slate-400 hover:text-white hover:bg-slate-800/60 whitespace-nowrap px-3.5 py-2 rounded-lg border border-transparent font-medium flex items-center gap-1.5">
      <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
      Employee Drilldown
    </button>
    <button onclick="tab('ex')" id="t-ex" class="tab-btn text-slate-400 hover:text-white hover:bg-slate-800/60 whitespace-nowrap px-3.5 py-2 rounded-lg border border-transparent font-medium flex items-center gap-1.5">
      <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
      Query Inspector
    </button>
  </div>

  <!-- TAB: OVERVIEW -->
  <div id="p-ov" class="space-y-6">
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="glass rounded-2xl p-5">
        <div class="flex items-center justify-between mb-4">
          <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Headcount</span>
          <div class="w-8 h-8 rounded-lg bg-indigo-500/10 flex items-center justify-center"><svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"/></svg></div>
        </div>
        <span class="text-3xl font-bold text-white" id="K1">--</span>
        <p class="text-[11px] text-slate-500 mt-1">Rule R9 evaluated</p>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="flex items-center justify-between mb-4">
          <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Present Days</span>
          <div class="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center"><svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></div>
        </div>
        <span class="text-3xl font-bold text-white" id="K2">--</span>
        <p class="text-[11px] text-slate-500 mt-1">Half-day = 0.5 (R5)</p>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="flex items-center justify-between mb-4">
          <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Late Incidents</span>
          <div class="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center"><svg class="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg></div>
        </div>
        <span class="text-3xl font-bold text-white" id="K3">--</span>
        <p class="text-[11px] text-slate-500 mt-1">&gt;10 min grace (R2)</p>
      </div>
      <div class="glass rounded-2xl p-5">
        <div class="flex items-center justify-between mb-4">
          <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Avg Work Hours</span>
          <div class="w-8 h-8 rounded-lg bg-sky-500/10 flex items-center justify-center"><svg class="w-4 h-4 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg></div>
        </div>
        <span class="text-3xl font-bold text-white" id="K4">--</span>
        <p class="text-[11px] text-slate-500 mt-1">Half-up rounded (R4, R8)</p>
      </div>
    </div>
    <div>
      <div class="flex items-center justify-between mb-3">
        <h2 class="text-sm font-bold text-white">Department Summaries — <span class="text-indigo-400 font-mono" id="dML">2026-07</span></h2>
        <code class="text-[11px] bg-slate-800 text-indigo-300 px-2 py-0.5 rounded">GET /analytics/departments/summary</code>
      </div>
      <div id="dG" class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="glass rounded-xl p-8 text-center text-slate-500 text-sm col-span-3">Loading…</div>
      </div>
    </div>
  </div>

  <!-- TAB: PUNCH -->
  <div id="p-pu" class="hidden space-y-5">
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <div class="glass rounded-2xl p-6 space-y-4">
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded-lg bg-indigo-500/15 flex items-center justify-center"><svg class="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1"/></svg></div>
          <div><h3 class="text-sm font-bold text-white">Punch-In Console</h3><p class="text-[11px] text-slate-400">Race-safe via unique index (R1, R2)</p></div>
        </div>
        <div class="space-y-3 text-xs">
          <div><label class="block text-slate-400 mb-1 font-medium">Employee</label><select id="piE" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-indigo-500"></select></div>
          <div><label class="block text-slate-400 mb-1 font-medium">Status</label><select id="piS" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-indigo-500"><option value="PRESENT">PRESENT (On-site)</option><option value="WFH">WFH</option><option value="ON_DUTY">ON_DUTY</option></select></div>
          <div><label class="block text-slate-400 mb-1 font-medium">Custom Instant (optional)</label><input type="datetime-local" id="piT" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs"><span class="text-[11px] text-slate-500 mt-1 block">Empty = server time now</span></div>
          <button onclick="doPi()" class="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition shadow shadow-indigo-600/30">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> Record Punch-In
          </button>
        </div>
      </div>
      <div class="glass rounded-2xl p-6 space-y-4">
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded-lg bg-emerald-500/15 flex items-center justify-center"><svg class="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"/></svg></div>
          <div><h3 class="text-sm font-bold text-white">Punch-Out Console</h3><p class="text-[11px] text-slate-400">Atomic find_one_and_update (R3–R5)</p></div>
        </div>
        <div class="space-y-3 text-xs">
          <div><label class="block text-slate-400 mb-1 font-medium">Employee</label><select id="poE" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs focus:outline-none focus:border-emerald-500"></select></div>
          <div><label class="block text-slate-400 mb-1 font-medium">Custom Instant (optional)</label><input type="datetime-local" id="poT" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-xs"><span class="text-[11px] text-slate-500 mt-1 block">Empty = server time now</span></div>
          <button onclick="doPo()" class="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-white font-semibold text-xs flex items-center justify-center gap-2 transition shadow shadow-emerald-600/30">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> Record Punch-Out
          </button>
        </div>
      </div>
      <div class="glass rounded-2xl p-6 flex flex-col">
        <div class="flex items-center justify-between mb-3"><span class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">API Response</span><span id="pBadge" class="text-[11px] px-2 py-0.5 rounded font-mono bg-slate-800 text-slate-400">idle</span></div>
        <pre id="pOut" class="flex-1 bg-slate-950 border border-slate-800 rounded-xl text-[11px] font-mono text-slate-300 p-3 overflow-auto leading-relaxed min-h-36">Awaiting punch event…</pre>
        <p class="text-[11px] text-slate-500 border-t border-slate-800 pt-3 mt-4">💡 Punch <b class="text-white">EMP0005</b> at ~01:30 IST to watch Rule R1 overnight shift binding!</p>
      </div>
    </div>
  </div>

  <!-- TAB: ATTENDANCE & AUDIT -->
  <div id="p-at" class="hidden space-y-4">
    <div class="glass p-4 rounded-xl flex flex-wrap items-center gap-3 text-xs">
      <input type="text" id="fE" placeholder="emp_code…" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white w-32 focus:outline-none focus:border-indigo-500">
      <select id="fSt" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500"><option value="">All Statuses</option><option value="PRESENT">PRESENT</option><option value="WFH">WFH</option><option value="ON_DUTY">ON_DUTY</option><option value="ABSENT">ABSENT</option><option value="LEAVE">LEAVE</option></select>
      <input type="date" id="fFr" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500">
      <input type="date" id="fTo" class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-indigo-500">
      <button onclick="loadRec()" class="bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-2 rounded-lg font-semibold transition">Apply</button>
      <div class="ml-auto text-slate-400">Sort: <code class="text-indigo-300">date↓ emp_code↑</code></div>
    </div>
    <div class="glass rounded-2xl overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-slate-900/80 border-b border-slate-800 text-slate-400 uppercase text-[11px] tracking-wider">
            <tr>
              <th class="py-3 px-4">Employee</th><th class="py-3 px-4">Date</th><th class="py-3 px-4">Status</th>
              <th class="py-3 px-4">In (IST)</th><th class="py-3 px-4">Out (IST)</th><th class="py-3 px-4">Hours</th>
              <th class="py-3 px-4">Late</th><th class="py-3 px-4">OT</th><th class="py-3 px-4">½</th><th class="py-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody id="rTb" class="divide-y divide-slate-800/50 font-mono">
            <tr><td colspan="10" class="py-10 text-center text-slate-500 font-sans">Loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB: TREND & LEADERBOARD -->
  <div id="p-an" class="hidden">
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5">
      <div class="lg:col-span-2 glass rounded-2xl p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="text-sm font-bold text-white flex items-center gap-2">Dept Attendance Trend<span class="text-[11px] px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">7-Day MA</span></h3>
            <p class="text-[11px] text-slate-400">$densify gap-fill · $setWindowFields sliding average</p>
          </div>
          <select id="tD" onchange="loadTrend()" class="bg-slate-800 border border-slate-700 text-xs px-2.5 py-1.5 rounded-lg text-white"><option value="Engineering">Engineering</option><option value="Sales">Sales</option><option value="Support">Support</option></select>
        </div>
        <div class="h-64"><canvas id="tC"></canvas></div>
      </div>
      <div class="glass rounded-2xl p-6 space-y-3">
        <div>
          <h3 class="text-sm font-bold text-white flex items-center gap-2">Late-Comers Ranking<span class="text-[11px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">1,2,2,4</span></h3>
          <p class="text-[11px] text-slate-400">$rank · limit after ranking · 2026-07</p>
        </div>
        <div id="lbL" class="space-y-2 max-h-72 overflow-y-auto pr-0.5"><div class="text-center py-8 text-xs text-slate-500">Loading…</div></div>
      </div>
    </div>
  </div>

  <!-- TAB: EMPLOYEE DRILLDOWN -->
  <div id="p-mo" class="hidden">
    <div class="glass rounded-2xl p-6 space-y-5">
      <div class="flex flex-wrap items-center justify-between gap-4">
        <div><h3 class="text-sm font-bold text-white">Employee Monthly Drilldown</h3><p class="text-[11px] text-slate-400">Working days from joined_on for mid-month joiners (R7). Half-up rounding (R8).</p></div>
        <div class="flex items-center gap-3">
          <select id="drE" onchange="loadMonthly()" class="bg-slate-800 border border-slate-700 text-xs px-3 py-2 rounded-lg text-white"></select>
          <input type="month" id="drM" value="2026-07" onchange="loadMonthly()" class="bg-slate-800 border border-slate-700 text-xs px-3 py-2 rounded-lg text-white">
        </div>
      </div>
      <div id="drC" class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div class="col-span-6 text-center py-10 text-xs text-slate-500">Select an employee to load metrics…</div>
      </div>
    </div>
  </div>

  <!-- TAB: QUERY INSPECTOR -->
  <div id="p-ex" class="hidden">
    <div class="glass rounded-2xl p-6 space-y-4">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 class="text-sm font-bold text-white flex items-center gap-2">MongoDB Query Plan Inspector<span class="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">Zero COLLSCAN</span></h3>
          <p class="text-[11px] text-slate-400">Every endpoint verified via executionStats — IXSCAN on 100k-record scale.</p>
        </div>
        <div class="flex items-center gap-2">
          <select id="epS" class="bg-slate-800 border border-slate-700 text-xs px-3 py-2 rounded-lg text-white"><option value="attendance_list">attendance_list</option><option value="employee_monthly">employee_monthly</option><option value="department_summary">department_summary</option><option value="late_leaderboard">late_leaderboard</option><option value="department_trend">department_trend</option></select>
          <button onclick="runEx()" class="bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-2 rounded-lg font-semibold transition">Run Explain</button>
        </div>
      </div>
      <div id="exSum" class="hidden grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs">
        <div><span class="text-slate-500 block">Collection</span><span id="eC" class="font-mono font-bold text-white">--</span></div>
        <div><span class="text-slate-500 block">Top Stage</span><span id="eS" class="font-mono font-bold text-emerald-400">--</span></div>
        <div><span class="text-slate-500 block">Verbosity</span><span class="font-mono text-slate-300">executionStats</span></div>
        <div><span class="text-slate-500 block">COLLSCAN?</span><span id="eCS" class="font-mono font-bold text-emerald-400">--</span></div>
      </div>
      <pre id="exOut" class="bg-slate-950 border border-slate-800 rounded-xl p-4 text-[11px] font-mono text-emerald-400 overflow-x-auto max-h-96">Select an endpoint and click Run Explain…</pre>
    </div>
  </div>
</main>

<!-- HISTORY MODAL -->
<div id="hM" class="fixed inset-0 bg-black/80 backdrop-blur z-50 hidden flex items-center justify-center p-4">
  <div class="bg-slate-900 border border-slate-700 rounded-2xl max-w-xl w-full p-6 shadow-2xl flex flex-col max-h-[80vh]">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
      <div><h3 class="text-sm font-bold text-white">Audit Trail — Regularization History</h3><p class="text-[11px] text-slate-400" id="hSub">--</p></div>
      <button onclick="closeH()" class="text-slate-400 hover:text-white text-xl">&times;</button>
    </div>
    <div id="hBdy" class="space-y-3 overflow-y-auto pr-1 text-xs flex-1"></div>
    <div class="pt-4 border-t border-slate-800 text-right mt-4"><button onclick="closeH()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold">Close</button></div>
  </div>
</div>

<!-- REGULARIZE MODAL -->
<div id="rM" class="fixed inset-0 bg-black/80 backdrop-blur z-50 hidden flex items-center justify-center p-4">
  <div class="bg-slate-900 border border-slate-700 rounded-2xl max-w-md w-full p-6 shadow-2xl">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
      <div><h3 class="text-sm font-bold text-white">Regularize Attendance</h3><p class="text-[11px] text-slate-400" id="rSub">--</p></div>
      <button onclick="closeR()" class="text-slate-400 hover:text-white text-xl">&times;</button>
    </div>
    <div class="space-y-3 text-xs">
      <div><label class="block text-slate-400 mb-1">New Status</label><select id="rSt" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white"><option value="PRESENT">PRESENT</option><option value="WFH">WFH</option><option value="ON_DUTY">ON_DUTY</option><option value="ABSENT">ABSENT</option><option value="LEAVE">LEAVE</option></select></div>
      <div><label class="block text-slate-400 mb-1">Punch-In (leave blank to keep)</label><input type="datetime-local" id="rPI" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white"></div>
      <div><label class="block text-slate-400 mb-1">Punch-Out (leave blank to keep)</label><input type="datetime-local" id="rPO" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white"></div>
      <div><label class="block text-slate-400 mb-1">Reason <span class="text-rose-400">*</span></label><input type="text" id="rRe" placeholder="min 5 chars…" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white"></div>
      <div><label class="block text-slate-400 mb-1">Regularized By <span class="text-rose-400">*</span></label><input type="text" id="rBy" placeholder="hr.admin" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white"></div>
    </div>
    <div class="flex items-center justify-end gap-2 pt-4 border-t border-slate-800 mt-4">
      <button onclick="closeR()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold">Cancel</button>
      <button onclick="subReg()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold">Save &amp; Recompute</button>
    </div>
  </div>
</div>

<footer class="border-t border-slate-800 py-4 text-center text-[11px] text-slate-500">
  HROne SE Assignment &bull; FastAPI 0.122 &bull; MongoDB 7.0 &bull; Business Rules R1–R10 implemented
</footer>

<script>
let chartInst=null, emps=[], regEmp=null, regDate=null;

setInterval(()=>{const d=new Date(),ist=new Date(d.toLocaleString('en-US',{timeZone:'Asia/Kolkata'}));document.getElementById('clk').textContent=ist.toLocaleTimeString('en-GB')+' IST';},1000);

function tab(id){
  document.querySelectorAll('[id^="p-"]').forEach(el=>el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(b=>{b.classList.remove('tab-active');b.classList.add('text-slate-400');});
  document.getElementById('p-'+id).classList.remove('hidden');
  const b=document.getElementById('t-'+id);
  if(b){b.classList.add('tab-active');b.classList.remove('text-slate-400');}
  if(id==='an')setTimeout(loadTrend,80);
}

async function go(){await Promise.all([loadEmps(),loadDept(),loadRec(),loadLB(),loadTrend()]);}

async function loadEmps(){
  const r=await fetch('/employees?page_size=100');
  const d=await r.json();
  emps=d.items||[];
  document.getElementById('K1').textContent=d.total??emps.length;
  ['piE','poE','drE'].forEach(id=>{
    const el=document.getElementById(id);
    if(el)el.innerHTML=emps.map(e=>`<option value="${e.emp_code}">${e.emp_code} – ${e.name} (${e.department})</option>`).join('');
  });
}

async function loadDept(month='2026-07'){
  document.getElementById('dML').textContent=month;
  const r=await fetch(`/analytics/departments/summary?month=${month}`);
  const d=await r.json();
  const items=d.items||[];
  let tp=0,tl=0,sw=0,cw=0;
  items.forEach(x=>{tp+=x.present_days;tl+=x.late_count;if(x.avg_work_hours!=null){sw+=x.avg_work_hours;cw++;}});
  document.getElementById('K2').textContent=tp.toFixed(1);
  document.getElementById('K3').textContent=tl;
  document.getElementById('K4').textContent=cw?(sw/cw).toFixed(2):'--';
  document.getElementById('dG').innerHTML=items.length?items.map(d=>`
    <div class="glass rounded-2xl p-5 space-y-3">
      <div class="flex items-center justify-between border-b border-slate-800 pb-2">
        <span class="font-bold text-white text-sm">${d.department}</span>
        <span class="text-[11px] font-mono px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300">${d.headcount} staff</span>
      </div>
      <div class="grid grid-cols-2 gap-2 text-xs">
        <div><span class="text-slate-500 block text-[11px]">Present Days</span><span class="font-mono text-emerald-400 font-bold">${d.present_days}</span></div>
        <div><span class="text-slate-500 block text-[11px]">Avg Hours</span><span class="font-mono text-white font-bold">${d.avg_work_hours!=null?d.avg_work_hours+'h':'—'}</span></div>
        <div><span class="text-slate-500 block text-[11px]">Late (total min)</span><span class="font-mono text-amber-400 font-bold">${d.late_count} (${d.total_late_minutes}m)</span></div>
        <div><span class="text-slate-500 block text-[11px]">Leave / On-Duty</span><span class="font-mono text-slate-300 font-bold">${d.leave_count} / ${d.on_duty_count}</span></div>
      </div>
    </div>`).join(''):'<div class="glass rounded-xl p-8 text-center text-slate-500 col-span-3">No data.</div>';
}

async function loadRec(){
  let url='/attendance?page=1&page_size=50';
  const e=document.getElementById('fE').value;
  const s=document.getElementById('fSt').value;
  const df=document.getElementById('fFr').value;
  const dt=document.getElementById('fTo').value;
  if(e)url+=`&emp_code=${encodeURIComponent(e)}`;
  if(s)url+=`&status=${s}`;
  if(df)url+=`&date_from=${df}`;
  if(dt)url+=`&date_to=${dt}`;
  const r=await fetch(url);
  const d=await r.json();
  const items=d.items||[];
  const sMap={PRESENT:'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',WFH:'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',ON_DUTY:'bg-sky-500/10 text-sky-400 border-sky-500/20',ABSENT:'bg-rose-500/10 text-rose-400 border-rose-500/20',LEAVE:'bg-amber-500/10 text-amber-400 border-amber-500/20'};
  const fm=ms=>ms?new Date(ms).toLocaleTimeString('en-GB',{timeZone:'Asia/Kolkata'}):'—';
  document.getElementById('rTb').innerHTML=items.length?items.map(r=>{
    const hc=(r.history||[]).length;
    const hd=encodeURIComponent(JSON.stringify(r.history||[]));
    return`<tr class="hover:bg-slate-800/40 transition">
      <td class="py-3 px-4 font-semibold text-white">${r.emp_code}</td>
      <td class="py-3 px-4 text-slate-300">${r.date}</td>
      <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[11px] font-sans border ${sMap[r.status]||'bg-slate-800 text-slate-400'}">${r.status}</span></td>
      <td class="py-3 px-4 text-slate-400 text-[11px]">${fm(r.punch_in)}</td>
      <td class="py-3 px-4 text-slate-400 text-[11px]">${fm(r.punch_out)}</td>
      <td class="py-3 px-4 text-white">${r.work_hours!=null?r.work_hours+'h':'—'}</td>
      <td class="py-3 px-4 ${r.late_minutes>0?'text-amber-400 font-bold':'text-slate-500'}">${r.late_minutes}m</td>
      <td class="py-3 px-4 ${r.overtime_minutes>0?'text-indigo-400 font-bold':'text-slate-500'}">${r.overtime_minutes}m</td>
      <td class="py-3 px-4">${r.half_day?'<span class="text-amber-400 font-bold text-[11px]">½</span>':'<span class="text-slate-600 text-[11px]">—</span>'}</td>
      <td class="py-3 px-4 font-sans text-right space-x-2">
        <button onclick="openR('${r.emp_code}','${r.date}','${r.status}')" class="text-indigo-400 hover:text-indigo-300 text-[11px] font-semibold">Edit</button>
        ${hc>0?`<button onclick="openH('${r.emp_code}','${r.date}',decodeURIComponent('${hd}'))" class="text-amber-400 hover:text-amber-300 text-[11px] font-semibold">Audit(${hc})</button>`:''}
      </td>
    </tr>`;}).join(''):'<tr><td colspan="10" class="py-10 text-center text-slate-500 font-sans text-xs">No records found.</td></tr>';
}

async function doPi(){
  const body={emp_code:document.getElementById('piE').value,status:document.getElementById('piS').value};
  const tv=document.getElementById('piT').value;
  if(tv)body.punched_at=new Date(tv).getTime();
  const r=await fetch('/attendance/punch-in',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  document.getElementById('pBadge').textContent=`HTTP ${r.status}`;
  document.getElementById('pBadge').className=r.ok?'text-[11px] px-2 py-0.5 rounded font-mono bg-emerald-500/20 text-emerald-400':'text-[11px] px-2 py-0.5 rounded font-mono bg-rose-500/20 text-rose-400';
  document.getElementById('pOut').textContent=JSON.stringify(d,null,2);
  go();
}

async function doPo(){
  const body={emp_code:document.getElementById('poE').value};
  const tv=document.getElementById('poT').value;
  if(tv)body.punched_at=new Date(tv).getTime();
  const r=await fetch('/attendance/punch-out',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  document.getElementById('pBadge').textContent=`HTTP ${r.status}`;
  document.getElementById('pBadge').className=r.ok?'text-[11px] px-2 py-0.5 rounded font-mono bg-emerald-500/20 text-emerald-400':'text-[11px] px-2 py-0.5 rounded font-mono bg-rose-500/20 text-rose-400';
  document.getElementById('pOut').textContent=JSON.stringify(d,null,2);
  go();
}

async function loadLB(){
  const r=await fetch('/analytics/leaderboard/late?month=2026-07&limit=10');
  const d=await r.json();
  const items=d.items||[];
  const medals=['🥇','🥈','🥉'];
  document.getElementById('lbL').innerHTML=items.length?items.map(it=>`
    <div class="flex items-center justify-between p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs">
      <div class="flex items-center gap-3">
        <span class="text-base w-5 text-center">${medals[it.rank-1]||'#'+it.rank}</span>
        <div><span class="font-bold text-white block">${it.name}</span><span class="text-[11px] text-slate-500">${it.emp_code} · ${it.department}</span></div>
      </div>
      <div class="text-right"><span class="font-mono text-amber-400 font-bold block">${it.total_late_minutes}m</span><span class="text-[11px] text-slate-500">${it.late_count} occurrence(s)</span></div>
    </div>`).join(''):'<div class="text-center py-8 text-xs text-slate-500">No late incidents this month.</div>';
}

async function loadTrend(){
  const dept=document.getElementById('tD').value;
  const r=await fetch(`/analytics/departments/${encodeURIComponent(dept)}/trend?from=2026-07-01&to=2026-07-15`);
  const d=await r.json();
  const items=d.items||[];
  const labels=items.map(i=>i.date.slice(5));
  const rates=items.map(i=>i.attendance_rate!==null?+(i.attendance_rate*100).toFixed(2):null);
  const mavg=items.map(i=>i.moving_avg_7d!==null?+(i.moving_avg_7d*100).toFixed(2):null);
  const ctx=document.getElementById('tC').getContext('2d');
  if(chartInst)chartInst.destroy();
  chartInst=new Chart(ctx,{type:'line',data:{labels,datasets:[
    {label:'Attendance Rate (%)',data:rates,borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,.08)',fill:true,tension:.35,borderWidth:2,pointRadius:4,pointHoverRadius:6,spanGaps:false},
    {label:'7-Day Moving Avg (%)',data:mavg,borderColor:'#10b981',borderDash:[5,4],borderWidth:2,pointRadius:0,fill:false,tension:.35}
  ]},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:'#94a3b8',font:{family:'Plus Jakarta Sans',size:11}}},tooltip:{backgroundColor:'#0f172a',borderColor:'#334155',borderWidth:1,titleColor:'#e2e8f0',bodyColor:'#94a3b8'}},scales:{x:{grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#64748b',font:{size:10}}},y:{min:0,max:100,grid:{color:'rgba(255,255,255,.04)'},ticks:{color:'#64748b',font:{size:10}}}}}});
}

async function loadMonthly(){
  const emp=document.getElementById('drE').value;
  const month=document.getElementById('drM').value;
  if(!emp||!month)return;
  const r=await fetch(`/analytics/employees/${emp}/monthly?month=${month}`);
  const d=await r.json();
  const cards=[
    {label:'Working Days (R7)',val:d.working_days,c:'text-white'},
    {label:'Present Days',val:d.present_days,c:'text-emerald-400'},
    {label:'Attendance %',val:d.attendance_pct!==null?d.attendance_pct+'%':'N/A',c:'text-indigo-400'},
    {label:'Leave Days',val:d.leave_days,c:'text-amber-400'},
    {label:'Late (×, min)',val:`${d.late_count}× ${d.total_late_minutes}m`,c:'text-rose-400'},
    {label:'Overtime (min)',val:d.total_overtime_minutes+'m',c:'text-sky-400'},
  ];
  document.getElementById('drC').innerHTML=cards.map(c=>`<div class="glass rounded-xl p-4 text-center"><span class="text-[11px] text-slate-500 block mb-1">${c.label}</span><span class="text-xl font-bold font-mono ${c.c}">${c.val}</span></div>`).join('');
}

async function runEx(){
  const ep=document.getElementById('epS').value;
  let url=`/admin/explain/${ep}`;
  if(ep==='employee_monthly')url+='?emp_code=EMP0001&month=2026-07';
  else if(ep==='department_summary'||ep==='late_leaderboard')url+='?month=2026-07';
  else if(ep==='department_trend')url+='?department=Engineering&from=2026-07-01&to=2026-07-10';
  const r=await fetch(url);
  const d=await r.json();
  const raw=JSON.stringify(d.explain,null,2);
  const bad=raw.includes('"COLLSCAN"');
  document.getElementById('exSum').classList.remove('hidden');
  document.getElementById('eC').textContent=d.collection;
  document.getElementById('eS').textContent=bad?'⚠️ COLLSCAN':'✓ IXSCAN';
  document.getElementById('eS').className='font-mono font-bold '+(bad?'text-rose-400':'text-emerald-400');
  document.getElementById('eCS').textContent=bad?'⚠️ COLLSCAN found':'✓ Clean';
  document.getElementById('eCS').className='font-mono font-bold '+(bad?'text-rose-400':'text-emerald-400');
  document.getElementById('exOut').textContent=raw;
}

function openH(emp,date,str){
  const hist=JSON.parse(str);
  document.getElementById('hSub').textContent=`${emp} · ${date}`;
  const fm=ms=>ms?new Date(ms).toLocaleTimeString('en-GB',{timeZone:'Asia/Kolkata'}):'null';
  document.getElementById('hBdy').innerHTML=hist.map((h,i)=>{
    const when=new Date(h.at).toLocaleString('en-GB',{timeZone:'Asia/Kolkata'});
    const changes=Object.entries(h.changes).map(([k,v])=>{
      const f=(k==='punch_in'||k==='punch_out')?fm:(x=>x);
      return`<div class="flex items-center gap-2 font-mono bg-slate-950 px-2 py-1 rounded text-[11px]"><span class="text-indigo-400 font-semibold">${k}:</span><span class="text-rose-400">${f(v.from)}</span><span class="text-slate-500">→</span><span class="text-emerald-400">${f(v.to)}</span></div>`;
    }).join('');
    return`<div class="p-3 rounded-xl bg-slate-800/70 border border-slate-700/60 space-y-2"><div class="flex items-center justify-between"><span class="text-slate-300 text-[11px] font-semibold">By: <b>${h.by}</b> · ${when} IST</span><span class="text-[11px] px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300">Entry #${i+1}</span></div><p class="text-slate-400 italic text-[11px]">"${h.reason}"</p><div class="space-y-1 pt-1">${changes}</div></div>`;
  }).join('');
  document.getElementById('hM').classList.remove('hidden');
}
function closeH(){document.getElementById('hM').classList.add('hidden');}

function openR(emp,date,st){
  regEmp=emp;regDate=date;
  document.getElementById('rSub').textContent=`${emp} · ${date}`;
  document.getElementById('rSt').value=st;
  ['rRe','rBy','rPI','rPO'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('rBy').value='hr.admin';
  document.getElementById('rM').classList.remove('hidden');
}
function closeR(){document.getElementById('rM').classList.add('hidden');}

async function subReg(){
  const reason=document.getElementById('rRe').value.trim();
  const rby=document.getElementById('rBy').value.trim();
  if(reason.length<5){alert('Reason must be ≥5 chars.');return;}
  if(!rby){alert('Regularized By required.');return;}
  const body={status:document.getElementById('rSt').value,reason,regularized_by:rby};
  const pi=document.getElementById('rPI').value;
  const po=document.getElementById('rPO').value;
  if(pi)body.punch_in=new Date(pi).getTime();
  if(po)body.punch_out=new Date(po).getTime();
  const r=await fetch(`/attendance/${regEmp}/${regDate}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(!r.ok){alert('Error: '+(d.detail||JSON.stringify(d)));return;}
  closeR();go();
}

window.addEventListener('DOMContentLoaded',go);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def get_dashboard():
    """Serve the HROne Intelligence executive dashboard UI."""
    return HTMLResponse(content=DASHBOARD_HTML)
