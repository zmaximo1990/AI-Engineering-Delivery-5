from enum import Enum

from pydantic import BaseModel, Field

from settings import settings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMConfig(BaseModel):
    """Provider, model, and sampling settings for one agent run."""

    provider: LLMProvider
    model: str | None = None
    max_tokens: int = Field(default=settings.default_max_tokens, gt=0)
    temperature: float = Field(default=settings.default_temperature, ge=0.0, le=2.0)


class HumanCheckFilter(str, Enum):
    """Tool-facing review filter. ``pending`` is a NULL human_check in the table."""

    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


class CompetenceNameInput(BaseModel):
    """Name of one competence in the catalog."""

    name: str = Field(
        ...,
        min_length=1,
        description="Competence name, for example Empatía, Liderazgo, or Inteligencia Social.",
    )


class ListAssetsInput(BaseModel):
    """Competence plus an optional human-review filter."""

    name: str = Field(
        ...,
        min_length=1,
        description="Competence name, for example Empatía or Liderazgo.",
    )
    human_check: HumanCheckFilter | None = Field(
        default=None,
        description=(
            "Filter the asset-competence link by human review. "
            "approved: a human accepted the link. "
            "rejected: a human rejected it. "
            "pending: nobody has reviewed it yet. "
            "Omit this field to return every review state."
        ),
    )


class AssetNameInput(BaseModel):
    """Name of one asset."""

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact asset name returned by list_assets_by_competence, "
            "for example La pausa antes de responder."
        ),
    )


class ListUsersInput(BaseModel):
    """Optional filters for the user directory. Omit both to list every user."""

    name: str | None = Field(
        default=None,
        description=(
            "Person name, for example Ana Pérez. "
            "Accent-insensitive. Omit to ignore the name."
        ),
    )
    email: str | None = Field(
        default=None,
        description=(
            "User email, for example ana@nimbus.test. "
            "Case-insensitive. Omit to ignore the email."
        ),
    )


class ListCoursesInput(BaseModel):
    """Optional name filter for the course catalog. Omit it to list every course."""

    name: str | None = Field(
        default=None,
        description=(
            "Course name, for example Empatía en el 1:1. "
            "Accent-insensitive. Omit to return every course. "
            "Does not filter by who is enrolled."
        ),
    )


class UserEmailInput(BaseModel):
    """Email of the person assigned to a training course."""

    email: str = Field(
        ...,
        min_length=3,
        description="User email, for example ana@nimbus.test.",
    )


class UserAndCompetenceInput(BaseModel):
    """One person and one competence."""

    email: str = Field(
        ...,
        min_length=3,
        description="User email, for example ana@nimbus.test.",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Competence name, for example Empatía or Liderazgo.",
    )
