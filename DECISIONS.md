# DECISIONS.md

1. **Indexes.**
   On `employees`: `{"emp_code": 1}` (unique) guarantees uniqueness and fast lookup; `{"department": 1, "emp_code": 1}` supports sorted employee listing; `{"joined_on": 1, "department": 1}` powers index scans for department summary headcount.
   On `attendance_logs`: `{"emp_code": 1, "date": 1}` (unique) addresses records and eliminates punch-in races; `{"date": -1, "emp_code": 1}` supports paginated listing; `{"emp_code": 1, "punch_in": -1}` finds recent open punches; `{"date": 1, "late_minutes": 1, "emp_code": 1}` enables a covered scan for the leaderboard.
   *Rejected*: Indexing `status` alone, because its low cardinality offers poor selectivity.

2. **Punch-in race.**
   Both concurrent requests validate the payload and calculate the attendance date. Both issue `db.attendance_logs.insert_one(...)`. MongoDB's unique index on `(emp_code, date)` atomically allows exactly one insert (returning 201). The second fails with `DuplicateKeyError`, caught by our handler to return 409.

3. **Ties.**
   Our aggregation pipeline uses `$setWindowFields` with `$rank` (standard competition ranking: 1, 2, 2, 4), filtering by `rank <= limit`. If two employees tie at the cutoff rank, both share that rank and are returned (yielding > `limit` rows), sorted by `total_late_minutes` descending and `emp_code` ascending.

4. **Headcount.**
   The aggregation starts on `employees` (matching `joined_on <= end_date`), grouping by `department` with `headcount: {"$sum": 1}`. Logs are joined via `$lookup`. Employees with zero logs produce an empty array, contributing 0 to present days while preserving headcount.

5. **One thing to change for 100x data.**
   Pre-aggregate daily and monthly rollups incrementally into summary collections using MongoDB Change Streams or scheduled workers, eliminating on-the-fly cross-collection lookups and densification at multi-million scale.
