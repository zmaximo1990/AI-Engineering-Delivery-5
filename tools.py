"""LangChain tools over the catalog queries.

The model chooses a tool from its description. These functions do not route
the conversation. Query functions are imported under aliases so the wrappers
do not shadow them.
"""

import logging

from langchain_core.tools import tool

from repository import get_asset as query_get_asset
from repository import get_competence as query_get_competence
from repository import list_assets_by_competence as query_list_assets_by_competence
from repository import list_courses as query_list_courses
from repository import list_trainings_by_user as query_list_trainings_by_user
from repository import (
    list_trainings_by_user_and_competence as query_list_trainings_by_user_and_competence,
)
from repository import list_trainings_competence as query_list_trainings_competence
from repository import list_users as query_list_users
from repository import run
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

logger = logging.getLogger(__name__)


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
    return run(lambda: query_get_competence(name))


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
    return run(lambda: query_list_assets_by_competence(name, human_check))


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
    return run(lambda: query_get_asset(name))


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
    return run(lambda: query_list_users(name, email))


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
    return run(lambda: query_list_courses(name))


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
    return run(lambda: query_list_trainings_by_user(email))


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
    return run(lambda: query_list_trainings_competence(name))


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
    return run(lambda: query_list_trainings_by_user_and_competence(email, name))


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
