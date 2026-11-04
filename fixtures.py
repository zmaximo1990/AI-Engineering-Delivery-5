"""Load ``data/fixtures.csv`` into the core catalog database."""

import csv
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from db import reset_core_db, session_scope
from models import Asset, AssetCompetenceMap, Competence, Course, HumanCheck, TrainingCourse, User
from settings import settings

logger = logging.getLogger(__name__)

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}
_EMPTY_REVIEW = {"", "none", "null", "pending"}


def load_fixtures(path: Path | None = None) -> dict[str, int]:
    """Replace the core catalog with the rows in the fixtures CSV.

    The checkpointer database is left untouched.
    """
    csv_path = path or settings.fixtures_path
    grouped = _read_rows(csv_path)
    reset_core_db()
    with session_scope() as db:
        competences = _insert_competences(db, grouped["competence"])
        assets = _insert_assets(db, grouped["asset"])
        maps = _insert_maps(db, grouped["asset_competence_map"], assets, competences)
        users = _insert_users(db, grouped["user"])
        courses = _insert_courses(db, grouped["course"], competences)
        trainings = _insert_trainings(db, grouped["training_course"], users, courses)
    counts = {
        "competences": len(competences),
        "assets": len(assets),
        "asset_competence_map": maps,
        "users": len(users),
        "courses": len(courses),
        "training_courses": trainings,
    }
    logger.info("Loaded fixtures from %s: %s", csv_path, counts)
    return counts


def _read_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fixtures file not found: {path}")
    grouped: dict[str, list[dict[str, str]]] = {
        "competence": [],
        "asset": [],
        "asset_competence_map": [],
        "user": [],
        "course": [],
        "training_course": [],
    }
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "entity" not in reader.fieldnames:
            raise ValueError("fixtures.csv must include an 'entity' column")
        for line_number, row in enumerate(reader, start=2):
            entity = (row.get("entity") or "").strip()
            if entity == "":
                continue
            if entity not in grouped:
                allowed = ", ".join(grouped)
                raise ValueError(f"Line {line_number}: entity must be one of: {allowed}")
            grouped[entity].append({key: (value or "").strip() for key, value in row.items()})
    if not grouped["competence"] or not grouped["asset"]:
        raise ValueError("fixtures.csv needs at least one competence and one asset")
    return grouped


def _require(row: dict[str, str], field: str, entity: str) -> str:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"{entity} row is missing {field}")
    return value


def _parse_bool(value: str, field: str) -> bool:
    token = value.strip().casefold()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise ValueError(f"Invalid boolean for {field}: {value!r}")


def _parse_human_check(value: str) -> HumanCheck | None:
    token = value.strip().casefold()
    if token in _EMPTY_REVIEW:
        return None
    if token == HumanCheck.APPROVED.value:
        return HumanCheck.APPROVED
    if token == HumanCheck.REJECTED.value:
        return HumanCheck.REJECTED
    raise ValueError("human_check must be approved, rejected, or empty")


def _parse_progress(value: str) -> int:
    try:
        progress = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid progress: {value!r}") from exc
    if not 0 <= progress <= 100:
        raise ValueError(f"progress must be an integer from 0 to 100, got {progress}")
    return progress


def _insert_competences(db: Session, rows: list[dict[str, str]]) -> dict[str, Competence]:
    found: dict[str, Competence] = {}
    for row in rows:
        name = _require(row, "name", "competence")
        if name.casefold() in {item.casefold() for item in found}:
            raise ValueError(f"Duplicate competence: {name}")
        competence = Competence(name=name, description=_require(row, "description", "competence"))
        db.add(competence)
        found[name] = competence
    db.flush()
    return found


def _insert_assets(db: Session, rows: list[dict[str, str]]) -> dict[str, Asset]:
    found: dict[str, Asset] = {}
    for row in rows:
        name = _require(row, "name", "asset")
        if name.casefold() in {item.casefold() for item in found}:
            raise ValueError(f"Duplicate asset: {name}")
        asset = Asset(
            name=name,
            type=_require(row, "type", "asset"),
            transcription=row.get("transcription", ""),
            url=_require(row, "url", "asset"),
        )
        db.add(asset)
        found[name] = asset
    db.flush()
    return found


def _lookup[T](found: dict[str, T], name: str, kind: str) -> T:
    for key, value in found.items():
        if key.casefold() == name.casefold():
            return value
    known = ", ".join(found)
    raise ValueError(f"Unknown {kind} {name!r}. Known values: {known}")


def _insert_maps(
    db: Session,
    rows: list[dict[str, str]],
    assets: dict[str, Asset],
    competences: dict[str, Competence],
) -> int:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        asset_name = _require(row, "name", "asset_competence_map")
        competence_name = _require(row, "competence", "asset_competence_map")
        key = (asset_name.casefold(), competence_name.casefold())
        if key in seen:
            raise ValueError(f"Duplicate map for {asset_name} / {competence_name}")
        seen.add(key)
        asset = _lookup(assets, asset_name, "asset")
        competence = _lookup(competences, competence_name, "competence")
        if not isinstance(asset, Asset) or not isinstance(competence, Competence):
            raise ValueError(f"Map row resolved the wrong record for {asset_name} / {competence_name}")
        db.add(
            AssetCompetenceMap(
                asset_id=asset.id,
                competence_id=competence.id,
                ai_check=_parse_bool(row.get("ai_check", ""), "ai_check"),
                human_check=_parse_human_check(row.get("human_check", "")),
            )
        )
    return len(rows)


def _insert_users(db: Session, rows: list[dict[str, str]]) -> dict[str, User]:
    found: dict[str, User] = {}
    for row in rows:
        email = _require(row, "user_email", "user")
        if email.casefold() in {item.casefold() for item in found}:
            raise ValueError(f"Duplicate user email: {email}")
        user = User(name=_require(row, "name", "user"), email=email)
        db.add(user)
        found[email] = user
    db.flush()
    return found


def _insert_courses(
    db: Session,
    rows: list[dict[str, str]],
    competences: dict[str, Competence],
) -> dict[str, Course]:
    found: dict[str, Course] = {}
    for row in rows:
        name = _require(row, "name", "course")
        if name.casefold() in {item.casefold() for item in found}:
            raise ValueError(f"Duplicate course: {name}")
        competence_name = _require(row, "competence", "course")
        competence = _lookup(competences, competence_name, "competence")
        course = Course(
            name=name,
            icon=_require(row, "icon", "course"),
            competence_id=competence.id,
        )
        db.add(course)
        found[name] = course
    db.flush()
    return found


def _insert_trainings(
    db: Session,
    rows: list[dict[str, str]],
    users: dict[str, User],
    courses: dict[str, Course],
) -> int:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        course_name = _require(row, "name", "training_course")
        email = _require(row, "user_email", "training_course")
        key = (email.casefold(), course_name.casefold())
        if key in seen:
            raise ValueError(f"Duplicate enrollment for {email} / {course_name}")
        seen.add(key)
        user = _lookup(users, email, "user")
        course = _lookup(courses, course_name, "course")
        db.add(
            TrainingCourse(
                user_id=user.id,
                course_id=course.id,
                progress=_parse_progress(_require(row, "progress", "training_course")),
            )
        )
    return len(rows)
