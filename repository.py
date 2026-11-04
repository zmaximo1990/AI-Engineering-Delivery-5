"""Queries over the core catalog models.

Each function takes the caller's ``Session`` and returns model instances.
Those instances are valid only while that session is open. The LangChain
tools in ``tools.py`` open the session, call these queries, and project the
models to the JSON the agent reads.
"""

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from models import (
    Asset,
    AssetCompetenceMap,
    Competence,
    Course,
    HumanCheck,
    TrainingCourse,
    User,
)


class LookupStatus(str, Enum):
    """How a name lookup resolved."""

    FOUND = "found"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


@dataclass(frozen=True)
class NameLookup[T]:
    """Rows selected by a name lookup.

    ``found`` puts one row in ``match`` when the lookup is unique, and every
    selected row in ``options``. ``ambiguous`` and ``missing`` leave ``match``
    empty and put the candidate rows in ``options``.
    """

    status: LookupStatus
    match: T | None
    options: list[T]


def _fold(value: str) -> str:
    """Case-fold and strip accents so Empatía and empatia match."""
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return without_marks.casefold().strip()


def _name_hits[T](rows: list[T], query: str, name_of: Callable[[T], str]) -> list[T]:
    """Exact folded matches, or partial matches when the query is long enough."""
    folded = _fold(query)
    exact = [row for row in rows if _fold(name_of(row)) == folded]
    if exact:
        return exact
    if len(folded) < 4:
        return []
    return [row for row in rows if folded in _fold(name_of(row))]


def match_names[T](
    rows: list[T],
    query: str,
    name_of: Callable[[T], str],
) -> NameLookup[T]:
    """Return every name match. Several matches are ``found``. None is ``missing``."""
    matched = _name_hits(rows, query, name_of)
    if not matched:
        return NameLookup(LookupStatus.MISSING, None, list(rows))
    match = matched[0] if len(matched) == 1 else None
    return NameLookup(LookupStatus.FOUND, match, matched)


def resolve_name[T](
    rows: list[T],
    query: str,
    name_of: Callable[[T], str],
) -> NameLookup[T]:
    """Return one name match. Several matches are ``ambiguous``. None is ``missing``."""
    matched = _name_hits(rows, query, name_of)
    if len(matched) == 1:
        return NameLookup(LookupStatus.FOUND, matched[0], matched)
    if len(matched) > 1:
        return NameLookup(LookupStatus.AMBIGUOUS, None, matched)
    return NameLookup(LookupStatus.MISSING, None, list(rows))


def _email_matches(email: str, query: str) -> bool:
    folded_email = email.casefold()
    folded_query = query.strip().casefold()
    if folded_email == folded_query:
        return True
    return len(folded_query) >= 4 and folded_query in folded_email


def competences(db: Session) -> list[Competence]:
    """Return every competence ordered by name."""
    return list(db.scalars(select(Competence).order_by(Competence.name)).all())


def assets(db: Session) -> list[Asset]:
    """Return every asset ordered by name."""
    return list(db.scalars(select(Asset).order_by(Asset.name)).all())


def asset_links(db: Session, asset: Asset) -> list[AssetCompetenceMap]:
    """Return the competence links of one asset, with the competence loaded."""
    return list(
        db.scalars(
            select(AssetCompetenceMap)
            .where(AssetCompetenceMap.asset_id == asset.id)
            .options(joinedload(AssetCompetenceMap.competence))
            .order_by(AssetCompetenceMap.id)
        ).all()
    )


def assets_for_competence(
    db: Session,
    competence: Competence,
    *,
    human_check: HumanCheck | None = None,
    pending_only: bool = False,
) -> list[tuple[Asset, AssetCompetenceMap]]:
    """Return assets linked to a competence, ordered by asset name.

    ``pending_only`` keeps links that nobody has reviewed. ``human_check`` keeps
    one reviewed state. With neither, every review state is returned.
    """
    statement = (
        select(Asset, AssetCompetenceMap)
        .join(AssetCompetenceMap, AssetCompetenceMap.asset_id == Asset.id)
        .where(AssetCompetenceMap.competence_id == competence.id)
        .order_by(Asset.name)
    )
    if pending_only:
        statement = statement.where(AssetCompetenceMap.human_check.is_(None))
    elif human_check is not None:
        statement = statement.where(AssetCompetenceMap.human_check == human_check)
    return [(asset, link) for asset, link in db.execute(statement)]


def users(db: Session) -> list[User]:
    """Return every user ordered by name."""
    return list(db.scalars(select(User).order_by(User.name)).all())


def filter_email(rows: list[User], query: str) -> list[User]:
    """Keep users whose email equals the query, or contains it when it is long enough."""
    return [user for user in rows if _email_matches(user.email, query)]


def known_emails(db: Session) -> list[str]:
    """Return every user email ordered by email."""
    return list(db.scalars(select(User.email).order_by(User.email)).all())


def courses(db: Session) -> list[Course]:
    """Return every course ordered by name, with the competence loaded."""
    return list(
        db.scalars(
            select(Course).options(joinedload(Course.competence)).order_by(Course.name)
        ).all()
    )


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


def trainings_for_user(db: Session, email: str) -> list[TrainingCourse]:
    """Return the trainings assigned to an email, ordered by course name."""
    folded = email.strip().casefold()
    return list(
        db.scalars(
            _training_statement()
            .where(func.lower(User.email) == folded)
            .order_by(Course.name)
        )
        .unique()
        .all()
    )


def trainings_for_competence(db: Session, competence: Competence) -> list[TrainingCourse]:
    """Return the trainings activated for one competence."""
    return list(
        db.scalars(
            _training_statement()
            .where(Course.competence_id == competence.id)
            .order_by(User.email, Course.name)
        )
        .unique()
        .all()
    )


def trainings_for_user_and_competence(
    db: Session,
    email: str,
    competence: Competence,
) -> list[TrainingCourse]:
    """Return one user's trainings for one competence, ordered by course name."""
    folded = email.strip().casefold()
    return list(
        db.scalars(
            _training_statement()
            .where(Course.competence_id == competence.id)
            .where(func.lower(User.email) == folded)
            .order_by(Course.name)
        )
        .unique()
        .all()
    )
