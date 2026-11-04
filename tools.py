"""LangChain tools over the catalog queries.

The model chooses a tool from its description. These functions do not route
the conversation. They open a session, call the repository, and project the
models to the JSON string the agent reads.
"""

import json
import logging
from collections.abc import Callable

from langchain_core.tools import tool
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

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
from repository import LookupStatus, NameLookup
from repository import asset_links as query_asset_links
from repository import assets as query_assets
from repository import assets_for_competence as query_assets_for_competence
from repository import competences as query_competences
from repository import courses as query_courses
from repository import filter_email as query_filter_email
from repository import known_emails as query_known_emails
from repository import match_names, resolve_name
from repository import trainings_for_competence as query_trainings_for_competence
from repository import trainings_for_user as query_trainings_for_user
from repository import (
    trainings_for_user_and_competence as query_trainings_for_user_and_competence,
)
from repository import users as query_users
from schemas import (
    AssetNameInput,
    CompetenceNameInput,
    HumanCheckFilter,
    ListAssetsInput,
    ListCoursesInput,
    ListUsersInput,
    UserAndCompetenceInput,
    UserEmailInput,
)
from settings import settings

logger = logging.getLogger(__name__)

_COMPETENCE_HINT = (
    "Retry with one exact name from valid_names, or ask the user which competence they mean."
)
_NAME_HINT = "Retry with one name from valid_names, or ask the user to clarify."
_EMAIL_HINT = "Retry with one email from known_emails, or ask the user which person they mean."


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


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


def run(query: Callable[[], str]) -> str:
    """Run one tool query and turn a catalog failure into JSON the agent can read."""
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


def _competence_miss(name: str, lookup: NameLookup[Competence]) -> str | None:
    if lookup.status is LookupStatus.FOUND:
        return None
    if lookup.status is LookupStatus.AMBIGUOUS:
        error = f"Several competences match {name!r}."
    else:
        error = f"Competence {name!r} was not found."
    return _dump(
        {
            "error": error,
            "valid_names": [row.name for row in lookup.options],
            "hint": _COMPETENCE_HINT,
        }
    )


def _resolve_competence(db: Session, name: str) -> tuple[Competence | None, str | None]:
    lookup = resolve_name(query_competences(db), name, lambda row: row.name)
    return lookup.match, _competence_miss(name, lookup)


def _name_miss(label: str, query: str, names: list[str]) -> str:
    return _dump(
        {
            "error": f"{label} {query!r} was not found.",
            "valid_names": names,
            "hint": _NAME_HINT,
        }
    )


def _review_filter(human_check: HumanCheckFilter | None) -> tuple[HumanCheck | None, bool]:
    if human_check is HumanCheckFilter.PENDING:
        return None, True
    if human_check is None:
        return None, False
    return HumanCheck(human_check.value), False


def _asset_payload(asset: Asset, link: AssetCompetenceMap) -> dict[str, object]:
    return {
        "name": asset.name,
        "type": asset.type,
        "url": asset.url,
        "ai_check": link.ai_check,
        "human_check": _review_value(link.human_check),
    }


def _query_get_competence(name: str) -> str:
    with session_scope() as db:
        competence, error = _resolve_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        return _dump(
            {
                "id": competence.id,
                "name": competence.name,
                "description": competence.description,
            }
        )


def _query_list_assets(name: str, human_check: HumanCheckFilter | None) -> str:
    with session_scope() as db:
        competence, error = _resolve_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        review, pending_only = _review_filter(human_check)
        rows = query_assets_for_competence(
            db,
            competence,
            human_check=review,
            pending_only=pending_only,
        )
        return _dump(
            {
                "competence": competence.name,
                "human_check": None if human_check is None else human_check.value,
                "order": "name",
                "assets": [_asset_payload(asset, link) for asset, link in rows],
                "note": (
                    "Transcriptions are omitted. "
                    "The first item is the first asset for this filter. "
                    "Call get_asset with that exact name to read a transcription excerpt."
                ),
            }
        )


def _query_get_asset(name: str) -> str:
    with session_scope() as db:
        lookup = resolve_name(query_assets(db), name, lambda row: row.name)
        if lookup.status is LookupStatus.AMBIGUOUS:
            return _dump(
                {
                    "error": f"Several assets match {name!r}.",
                    "candidates": [row.name for row in lookup.options],
                    "hint": "Retry get_asset with one exact name from candidates.",
                }
            )
        if lookup.status is not LookupStatus.FOUND or lookup.match is None:
            return _dump(
                {
                    "error": f"Asset {name!r} was not found.",
                    "valid_names": [row.name for row in lookup.options],
                    "hint": "Retry with one exact name from valid_names, or ask the user to clarify.",
                }
            )
        asset = lookup.match
        checks = [
            {
                "competence": link.competence.name,
                "ai_check": link.ai_check,
                "human_check": _review_value(link.human_check),
            }
            for link in query_asset_links(db, asset)
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


def _query_list_users(name: str | None, email: str | None) -> str:
    with session_scope() as db:
        selected = query_users(db)
        name_query = _blank(name)
        email_query = _blank(email)
        if name_query is not None:
            lookup = match_names(selected, name_query, lambda user: user.name)
            if lookup.status is LookupStatus.MISSING:
                return _name_miss("User", name_query, [user.name for user in lookup.options])
            selected = lookup.options
        if email_query is not None:
            selected = query_filter_email(selected, email_query)
            if not selected:
                return _dump(
                    {
                        "error": f"No user matches email {email_query!r}.",
                        "known_emails": query_known_emails(db),
                        "hint": _EMAIL_HINT,
                    }
                )
        return _dump(
            {
                "name": name_query,
                "email": email_query,
                "users": [_user_payload(user) for user in selected],
            }
        )


def _query_list_courses(name: str | None) -> str:
    with session_scope() as db:
        selected = query_courses(db)
        name_query = _blank(name)
        if name_query is not None:
            lookup = match_names(selected, name_query, lambda course: course.name)
            if lookup.status is LookupStatus.MISSING:
                return _name_miss("Course", name_query, [course.name for course in lookup.options])
            selected = lookup.options
        return _dump(
            {
                "name": name_query,
                "courses": [_course_payload(course) for course in selected],
            }
        )


def _query_trainings_by_user(email: str) -> str:
    with session_scope() as db:
        matched = query_trainings_for_user(db, email)
        if not matched:
            return _dump(
                {
                    "error": f"No trainings are assigned to {email!r}.",
                    "known_emails": query_known_emails(db),
                    "hint": _EMAIL_HINT,
                }
            )
        return _dump(
            {
                "email": matched[0].user.email,
                "trainings": [_training_payload(row) for row in matched],
            }
        )


def _query_trainings_competence(name: str) -> str:
    with session_scope() as db:
        competence, error = _resolve_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        rows = query_trainings_for_competence(db, competence)
        return _dump(
            {
                "competence": competence.name,
                "trainings": [_training_payload(row) for row in rows],
            }
        )


def _query_trainings_by_user_and_competence(email: str, name: str) -> str:
    with session_scope() as db:
        competence, error = _resolve_competence(db, name)
        if error is not None or competence is None:
            return error or _dump({"error": "Competence was not found."})
        matched = query_trainings_for_user_and_competence(db, email, competence)
        if not matched:
            return _dump(
                {
                    "error": (
                        f"No trainings for {email!r} on competence {competence.name!r}."
                    ),
                    "competence": competence.name,
                    "known_emails": query_known_emails(db),
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


@tool(
    "get_competence",
    args_schema=CompetenceNameInput,
    description=(
        "Look up one competence in the core catalog by name. "
        "Use it when the user names a competence or before you filter assets and trainings by it. "
        "Returns the competence id, the canonical name, and the description. "
        "Does not return assets or trainings. "
        "If the name does not exist, returns an error and the valid competence names: "
        "retry with one of those names, or ask the user to clarify."
    ),
)
def get_competence(name: str) -> str:
    """Return one competence id and description, or the valid names when it is missing."""
    logger.info("tool get_competence name=%s", name)
    return run(lambda: _query_get_competence(name))


@tool(
    "list_assets_by_competence",
    args_schema=ListAssetsInput,
    description=(
        "List catalog assets linked to a competence, with AI and human review status. "
        "Use it to answer which videos or audios belong to a competence and whether a human "
        "approved, rejected, or has not reviewed that link. "
        "Pass human_check=approved when the user asks for content a human already validated. "
        "Pass rejected or pending for those states. Omit human_check to return every state. "
        "Results are ordered by asset name and do not include the transcription. "
        "To quote or summarize what an asset says, call get_asset with the exact name of the "
        "first item, or of the item the user asked for. "
        "If the competence does not exist, returns the valid names so you can retry or ask."
    ),
)
def list_assets_by_competence(
    name: str,
    human_check: HumanCheckFilter | None = None,
) -> str:
    """Return assets and review flags for a competence, without the transcription."""
    logger.info("tool list_assets_by_competence name=%s human_check=%s", name, human_check)
    return run(lambda: _query_list_assets(name, human_check))


@tool(
    "get_asset",
    args_schema=AssetNameInput,
    description=(
        "Fetch one asset by name. Returns its type, url, every competence link with ai_check "
        "and human_check, and a short transcription excerpt. "
        "Use it after list_assets_by_competence when the user wants to know what the asset says, "
        "or when they ask for the url or the review state of one asset. "
        "If several assets match, returns those names and no excerpt: retry with an exact name. "
        "If the transcription is empty or ai_check is false, the payload is marked incomplete: "
        "report the gap instead of inventing the missing text or classification."
    ),
)
def get_asset(name: str) -> str:
    """Return the url, review checks, and a transcription excerpt for one asset."""
    logger.info("tool get_asset name=%s", name)
    return run(lambda: _query_get_asset(name))


@tool(
    "list_users",
    args_schema=ListUsersInput,
    description=(
        "List people in the catalog. Omit both filters to return every user. "
        "Pass name for an accent-insensitive match, or email for a case-insensitive match. "
        "Use it when the user asks who someone is, or which people exist. "
        "Does not return the courses that person is taking: call list_trainings_by_user for that. "
        "Several matches are all returned. If nothing matches, returns the known names or emails "
        "so you can retry or ask the user to clarify."
    ),
)
def list_users(name: str | None = None, email: str | None = None) -> str:
    """Return users, optionally filtered by name or email."""
    logger.info("tool list_users name=%s email=%s", name, email)
    return run(lambda: _query_list_users(name, email))


@tool(
    "list_courses",
    args_schema=ListCoursesInput,
    description=(
        "List courses in the catalog, with icon and competence. "
        "Omit name to return every course. Pass name for an accent-insensitive filter. "
        "Use it when the user asks which courses exist. "
        "Does not say who is enrolled. For a person's courses, call list_trainings_by_user. "
        "Several matches are all returned. If the name does not exist, returns the valid course names."
    ),
)
def list_courses(name: str | None = None) -> str:
    """Return courses and their competence, without enrollments."""
    logger.info("tool list_courses name=%s", name)
    return run(lambda: _query_list_courses(name))


@tool(
    "list_trainings_by_user",
    args_schema=UserEmailInput,
    description=(
        "List every training course assigned to one user by email. "
        "Returns each course name, icon, competence, and progress as an integer percentage from 0 to 100. "
        "Report that number as the percentage. Do not scale it. "
        "Use it whenever the user asks which trainings a person has, even if they also "
        "ask for the progress of one competence in that list. This payload already answers both. "
        "If the email has no courses, returns the known emails so you can retry "
        "or ask which person they mean."
    ),
)
def list_trainings_by_user(email: str) -> str:
    """Return the trainings assigned to a user email."""
    logger.info("tool list_trainings_by_user email=%s", email)
    return run(lambda: _query_trainings_by_user(email))


@tool(
    "list_trainings_competence",
    args_schema=CompetenceNameInput,
    description=(
        "List the enrollments activated for one competence. "
        "Returns each course name, icon, user name, user email, and progress as an integer percentage from 0 to 100. "
        "Report that number as the percentage. Do not scale it. "
        "Use it when the user asks who is training a competence. "
        "To list the course catalog itself, call list_courses. "
        "If the competence does not exist, returns the valid competence names so you can retry or ask."
    ),
)
def list_trainings_competence(name: str) -> str:
    """Return the trainings activated for one competence."""
    logger.info("tool list_trainings_competence name=%s", name)
    return run(lambda: _query_trainings_competence(name))


@tool(
    "list_trainings_by_user_and_competence",
    args_schema=UserAndCompetenceInput,
    description=(
        "List only the training courses assigned to one user email for one competence. "
        "Returns each course name, icon, and progress as an integer percentage from 0 to 100. "
        "Report that number as the percentage. Do not scale it. "
        "Use it when the question is limited to that person on that competence and does not ask "
        "for the person's other courses. "
        "If the user asks for every training assigned to the person, call list_trainings_by_user "
        "instead. This tool does not include courses on other competences. "
        "If the email or the competence does not match, returns a recoverable error with the "
        "valid values so you can retry or ask the user to clarify."
    ),
)
def list_trainings_by_user_and_competence(email: str, name: str) -> str:
    """Return one user's trainings for one competence."""
    logger.info(
        "tool list_trainings_by_user_and_competence email=%s name=%s",
        email,
        name,
    )
    return run(lambda: _query_trainings_by_user_and_competence(email, name))


tools = [
    get_competence,
    list_assets_by_competence,
    get_asset,
    list_users,
    list_courses,
    list_trainings_by_user,
    list_trainings_competence,
    list_trainings_by_user_and_competence,
]
