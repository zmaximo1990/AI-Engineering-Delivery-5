"""Core catalog tables. Review state lives on the asset-competence link."""

from enum import Enum

from sqlalchemy import CheckConstraint
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for the core catalog."""


class HumanCheck(str, Enum):
    """Human review of one asset-competence link. NULL in the column means not reviewed."""

    APPROVED = "approved"
    REJECTED = "rejected"


def _enum_values(enum_cls: type[Enum]) -> list[str]:
    return [item.value for item in enum_cls]


class Asset(Base):
    """A content asset. The transcription is stored in full and excerpted by the tools."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    transcription: Mapped[str] = mapped_column(Text, nullable=False, default="")
    url: Mapped[str] = mapped_column(String(500), nullable=False)

    competence_links: Mapped[list["AssetCompetenceMap"]] = relationship(
        back_populates="asset",
    )


class Competence(Base):
    """A predefined competence, such as Liderazgo or Empatía."""

    __tablename__ = "competences"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    asset_links: Mapped[list["AssetCompetenceMap"]] = relationship(
        back_populates="competence",
    )
    trainings: Mapped[list["TrainingCourse"]] = relationship(back_populates="competence")


class AssetCompetenceMap(Base):
    """Link between an asset and a competence, with AI and human review flags."""

    __tablename__ = "asset_competence_map"
    __table_args__ = (
        UniqueConstraint("asset_id", "competence_id", name="uq_asset_competence"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), nullable=False, index=True)
    competence_id: Mapped[int] = mapped_column(
        ForeignKey("competences.id"),
        nullable=False,
        index=True,
    )
    ai_check: Mapped[bool] = mapped_column(nullable=False, default=False)
    human_check: Mapped[HumanCheck | None] = mapped_column(
        SqlEnum(
            HumanCheck,
            native_enum=False,
            length=16,
            values_callable=_enum_values,
        ),
        nullable=True,
    )

    asset: Mapped[Asset] = relationship(back_populates="competence_links")
    competence: Mapped[Competence] = relationship(back_populates="asset_links")


class TrainingCourse(Base):
    """A training course assigned to a person for one competence."""

    __tablename__ = "training_courses"
    __table_args__ = (
        CheckConstraint("progress >= 0 AND progress <= 100", name="ck_training_progress"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    icon: Mapped[str] = mapped_column(String(32), nullable=False)
    user_name: Mapped[str] = mapped_column(String(200), nullable=False)
    user_email: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False)
    competence_id: Mapped[int] = mapped_column(
        ForeignKey("competences.id"),
        nullable=False,
        index=True,
    )

    competence: Mapped[Competence] = relationship(back_populates="trainings")
