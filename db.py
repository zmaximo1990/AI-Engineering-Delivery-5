"""SQLAlchemy queries over the core catalog.

Each query returns JSON text, including when the lookup fails, so the agent
can retry with a valid name or ask the user to clarify. The LangChain tools
that call these queries live in ``tools.py``.
"""

import json
import logging
import unicodedata
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, joinedload, sessionmaker
from sqlalchemy.pool import NullPool

from models import Asset, AssetCompetenceMap, Competence, HumanCheck, TrainingCourse
from schemas import HumanCheckFilter
from settings import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the process-wide engine for the core catalog database."""
    global _engine
    if _engine is None:
        settings.core_db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite:///{settings.core_db_path}",
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        _engine = engine
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return a session factory bound to the core catalog engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Open a short-lived session and close it when the query finishes."""
    db = get_session_factory()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_core_db() -> None:
    """Drop and recreate the core catalog tables. Does not touch the checkpointer."""
    from models import Base

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def _dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fold(value: str) -> str:
    """Case-fold and strip accents so Empatía and empatia match."""
    normalized = unicodedata.normalize("NFKD", value)
    without_marks = "".join(character for character in normalized if not unicodedata.combining(character))
    return without_marks.casefold().strip()


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
    rows = db.scalars(select(TrainingCourse.user_email).distinct().order_by(TrainingCourse.user_email)).all()
    return list(rows)


def _training_payload(row: TrainingCourse) -> dict[str, object]:
    return {
        "name": row.name,
        "icon": row.icon,
        "user_name": row.user_name,
        "user_email": row.user_email,
        "competence": row.competence.name,
        "progress": int(row.progress),
    }

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


def list_trainings_by_user(email: str) -> str:
    with session_scope() as db:
        folded = email.strip().casefold()
        matched = list(
            db.scalars(
                select(TrainingCourse)
                .options(joinedload(TrainingCourse.competence))
                .where(func.lower(TrainingCourse.user_email) == folded)
                .order_by(TrainingCourse.name)
            ).all()
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
                "email": matched[0].user_email,
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
                select(TrainingCourse)
                .where(TrainingCourse.competence_id == competence.id)
                .options(joinedload(TrainingCourse.competence))
                .order_by(TrainingCourse.user_email, TrainingCourse.name)
            ).all()
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
                select(TrainingCourse)
                .where(TrainingCourse.competence_id == competence.id)
                .where(func.lower(TrainingCourse.user_email) == folded)
                .options(joinedload(TrainingCourse.competence))
                .order_by(TrainingCourse.name)
            ).all()
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
                "email": matched[0].user_email,
                "competence": competence.name,
                "trainings": [_training_payload(row) for row in matched],
            }
        )
