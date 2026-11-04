"""Queries over the core catalog models.

Each query returns JSON text, including when the lookup fails, so the agent
can retry with a valid name or ask the user to clarify. The LangChain tools
that call these queries live in ``tools.py``.
"""

import json
import logging
import unicodedata
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, joinedload

from db import session_scope
from models import (
    Asset,
    AssetCompetenceMap,
    Competence,
    Course,
    HumanCheck,
    TrainingCourse,
    User,
)
from schemas import HumanCheckFilter
from settings import settings

logger = logging.getLogger(__name__)


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fold(value: str) -> str:
    """Case-fold and strip accents so Empatía and empatia match."""
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return without_marks.casefold().strip()


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _find_competence(db: Session, name: str) -> tuple[Competence | None, str | None]:
    rows = list(db.scalars(select(Competence).order_by(Competence.name)).all())
    folded = _fold(name)
    exact = [row for row in rows if _fold(row.name) == folded]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, _dump(
            {
                "error": f"Several competences match {name!r}.",
                "valid_names": [row.name for row in exact],
                "hint": "Retry with one exact name from valid_names, or ask the user which competence they mean.",
            }
        )
    partial = [row for row in rows if len(folded) >= 4 and folded in _fold(row.name)]
    if len(partial) == 1:
        return partial[0], None
    if len(partial) > 1:
        return None, _dump(
            {
                "error": f"Several competences match {name!r}.",
                "valid_names": [row.name for row in partial],
                "hint": "Retry with one exact name from valid_names, or ask the user which competence they mean.",
            }
        )
    return None, _dump(
        {
            "error": f"Competence {name!r} was not found.",
            "valid_names": [row.name for row in rows],
            "hint": "Retry with one exact name from valid_names, or ask the user which competence they mean.",
        }
    )


def _find_assets(db: Session, name: str) -> tuple[list[Asset], str | None]:
    rows = list(db.scalars(select(Asset).order_by(Asset.name)).all())
    folded = _fold(name)
    exact = [row for row in rows if _fold(row.name) == folded]
    if len(exact) == 1:
        return exact, None
    if len(exact) > 1:
        return [], _dump(
            {
                "error": f"Several assets match {name!r}.",
                "candidates": [row.name for row in exact],
                "hint": "Retry get_asset with one exact name from candidates.",
            }
        )
    partial = [row for row in rows if len(folded) >= 4 and folded in _fold(row.name)]
    if len(partial) == 1:
        return partial, None
    if len(partial) > 1:
        return [], _dump(
            {
                "error": f"Several assets match {name!r}.",
                "candidates": [row.name for row in partial],
                "hint": "Retry get_asset with one exact name from candidates.",
            }
        )
    return [], _dump(
        {
            "error": f"Asset {name!r} was not found.",
            "valid_names": [row.name for row in rows],
            "hint": "Retry with one exact name from valid_names, or ask the user to clarify.",
        }
    )


def _match_names[T](
    rows: list[T],
    query: str,
    label: str,
    name_of: Callable[[T], str],
) -> tuple[list[T], str | None]:
    """Return every row whose name matches, or an error when none do."""
    folded = _fold(query)
    exact = [row for row in rows if _fold(name_of(row)) == folded]
    if exact:
        return exact, None
    partial = [row for row in rows if len(folded) >= 4 and folded in _fold(name_of(row))]
    if partial:
        return partial, None
    return [], _dump(
        {
            "error": f"{label} {query!r} was not found.",
            "valid_names": [name_of(row) for row in rows],
            "hint": "Retry with one name from valid_names, or ask the user to clarify.",
        }
    )


def _email_matches(email: str, query: str) -> bool:
    folded_email = email.casefold()
    folded_query = query.strip().casefold()
    if folded_email == folded_query:
        return True
    return len(folded_query) >= 4 and folded_query in folded_email


def _review_value(human_check: HumanCheck | None) -> str | None:
    if human_check is None:
        return None
    return human_check.value


def _excerpt(transcription: str) -> str:
    text = transcription.strip()
    if not text:
        return ""
    limit = settings.transcript_excerpt_chars
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _known_emails(db: Session) -> list[str]:
    rows = db.scalars(select(User.email).order_by(User.email)).all()
    return list(rows)


def _user_payload(user: User) -> dict[str, object]:
    return {"name": user.name, "email": user.email}


def _course_payload(course: Course) -> dict[str, object]:
    return {
        "name": course.name,
        "icon": course.icon,
        "competence": course.competence.name,
    }


def _training_payload(row: TrainingCourse) -> dict[str, object]:
    return {
        "name": row.course.name,
        "icon": row.course.icon,
        "user_name": row.user.name,
        "user_email": row.user.email,
        "competence": row.course.competence.name,
        "progress": int(row.progress),
    }


def _training_statement():
    return (
        select(TrainingCourse)
        .join(TrainingCourse.user)
        .join(TrainingCourse.course)
        .options(
            joinedload(TrainingCourse.user),
            joinedload(TrainingCourse.course).joinedload(Course.competence),
        )
    )


def run(query: Callable[[], str]) -> str:
    try:
        return query()
    except OperationalError:
        logger.exception("Core catalog query failed")
        return _dump(
            {
                "error": "The core catalog is not initialized. Run python main.py --fixtures.",
                "hint": "Ask the user to load the fixtures, or stop if you cannot.",
            }
        )
    except Exception as exc:
        logger.exception("Tool query failed")
        return _dump(
            {
                "error": "The catalog query failed.",
                "detail": str(exc),
                "hint": "Retry once. If it fails again, tell the user the catalog could not be read.",
            }
        )


def get_competence(name: str) -> str:
    with session_scope() as db:
        competence, error = _find_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        return _dump(
            {
                "id": competence.id,
                "name": competence.name,
                "description": competence.description,
            }
        )


def list_assets_by_competence(name: str, human_check: HumanCheckFilter | None) -> str:
    with session_scope() as db:
        competence, error = _find_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})

        statement = (
            select(Asset, AssetCompetenceMap)
            .join(AssetCompetenceMap, AssetCompetenceMap.asset_id == Asset.id)
            .where(AssetCompetenceMap.competence_id == competence.id)
            .order_by(Asset.name)
        )
        if human_check is HumanCheckFilter.PENDING:
            statement = statement.where(AssetCompetenceMap.human_check.is_(None))
        elif human_check is not None:
            statement = statement.where(
                AssetCompetenceMap.human_check == HumanCheck(human_check.value)
            )

        assets: list[dict[str, object]] = []
        for asset, link in db.execute(statement):
            assets.append(
                {
                    "name": asset.name,
                    "type": asset.type,
                    "url": asset.url,
                    "ai_check": link.ai_check,
                    "human_check": _review_value(link.human_check),
                }
            )
        return _dump(
            {
                "competence": competence.name,
                "human_check": None if human_check is None else human_check.value,
                "order": "name",
                "assets": assets,
                "note": (
                    "Transcriptions are omitted. "
                    "The first item is the first asset for this filter. "
                    "Call get_asset with that exact name to read a transcription excerpt."
                ),
            }
        )


def get_asset(name: str) -> str:
    with session_scope() as db:
        matches, error = _find_assets(db, name)
        if error is not None:
            return error
        asset = matches[0]
        links = db.scalars(
            select(AssetCompetenceMap)
            .where(AssetCompetenceMap.asset_id == asset.id)
            .options(joinedload(AssetCompetenceMap.competence))
            .order_by(AssetCompetenceMap.id)
        ).all()
        checks = [
            {
                "competence": link.competence.name,
                "ai_check": link.ai_check,
                "human_check": _review_value(link.human_check),
            }
            for link in links
        ]
        excerpt = _excerpt(asset.transcription)
        incomplete = excerpt == "" or any(not check["ai_check"] for check in checks)
        payload: dict[str, object] = {
            "name": asset.name,
            "type": asset.type,
            "url": asset.url,
            "checks": checks,
            "transcription_excerpt": excerpt,
            "incomplete": incomplete,
        }
        if incomplete:
            payload["hint"] = (
                "This record is incomplete. Say what is missing "
                "(empty transcription and/or ai_check false) instead of inventing it."
            )
        return _dump(payload)


def list_users(name: str | None = None, email: str | None = None) -> str:
    """List users, optionally filtered by name and email. Several matches are all returned."""
    with session_scope() as db:
        rows = list(db.scalars(select(User).order_by(User.name)).all())
        selected = rows
        name_query = _blank(name)
        email_query = _blank(email)
        if name_query is not None:
            selected, error = _match_names(selected, name_query, "User", lambda user: user.name)
            if error is not None:
                return error
        if email_query is not None:
            selected = [user for user in selected if _email_matches(user.email, email_query)]
            if not selected:
                return _dump(
                    {
                        "error": f"No user matches email {email_query!r}.",
                        "known_emails": _known_emails(db),
                        "hint": "Retry with one email from known_emails, or ask the user which person they mean.",
                    }
                )
        return _dump(
            {
                "name": name_query,
                "email": email_query,
                "users": [_user_payload(user) for user in selected],
            }
        )


def list_courses(name: str | None = None) -> str:
    """List courses with icon and competence. A name filter returns every match, not one guess."""
    with session_scope() as db:
        rows = list(
            db.scalars(
                select(Course)
                .options(joinedload(Course.competence))
                .order_by(Course.name)
            ).all()
        )
        name_query = _blank(name)
        selected = rows
        if name_query is not None:
            selected, error = _match_names(selected, name_query, "Course", lambda course: course.name)
            if error is not None:
                return error
        return _dump(
            {
                "name": name_query,
                "courses": [_course_payload(course) for course in selected],
            }
        )


def list_trainings_by_user(email: str) -> str:
    with session_scope() as db:
        folded = email.strip().casefold()
        matched = list(
            db.scalars(
                _training_statement()
                .where(func.lower(User.email) == folded)
                .order_by(Course.name)
            )
            .unique()
            .all()
        )
        if not matched:
            return _dump(
                {
                    "error": f"No trainings are assigned to {email!r}.",
                    "known_emails": _known_emails(db),
                    "hint": "Retry with one email from known_emails, or ask the user which person they mean.",
                }
            )
        return _dump(
            {
                "email": matched[0].user.email,
                "trainings": [_training_payload(row) for row in matched],
            }
        )


def list_trainings_competence(name: str) -> str:
    with session_scope() as db:
        competence, error = _find_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        rows = list(
            db.scalars(
                _training_statement()
                .where(Course.competence_id == competence.id)
                .order_by(User.email, Course.name)
            )
            .unique()
            .all()
        )
        return _dump(
            {
                "competence": competence.name,
                "trainings": [_training_payload(row) for row in rows],
            }
        )


def list_trainings_by_user_and_competence(email: str, name: str) -> str:
    with session_scope() as db:
        competence, error = _find_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        folded = email.strip().casefold()
        matched = list(
            db.scalars(
                _training_statement()
                .where(Course.competence_id == competence.id)
                .where(func.lower(User.email) == folded)
                .order_by(Course.name)
            )
            .unique()
            .all()
        )
        if not matched:
            return _dump(
                {
                    "error": (
                        f"No trainings for {email!r} on competence {competence.name!r}."
                    ),
                    "competence": competence.name,
                    "known_emails": _known_emails(db),
                    "hint": "Retry with a known email, or ask the user to clarify the person.",
                }
            )
        return _dump(
            {
                "email": matched[0].user.email,
                "competence": competence.name,
                "trainings": [_training_payload(row) for row in matched],
            }
        )
