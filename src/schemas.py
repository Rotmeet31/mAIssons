from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class StepTrace(BaseModel):
    tool_name: str
    args: dict[str, Any]
    result_summary: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunTrace(BaseModel):
    steps: list[StepTrace] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def add_step(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.steps.append(
            StepTrace(
                tool_name=tool_name,
                args=args,
                result_summary=str(result)[:200],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        self.total_prompt_tokens += prompt_tokens or 0
        self.total_completion_tokens += completion_tokens or 0


class LeanSummaries(BaseModel):
    left: str = ""
    center: str = ""
    right: str = ""


class AgentResponse(BaseModel):
    topic_overview: str = Field(
        description="2-3 sentence neutral summary of what is happening across the matched stories"
    )
    shared_ground: list[str] = Field(
        min_length=1,
        description="Facts all leans report consistently across stories, with inline outlet citations",
    )
    left_emphasis: list[str] = Field(
        default_factory=list,
        description="Angles left outlets add across stories that right outlets omit or downplay",
    )
    right_emphasis: list[str] = Field(
        default_factory=list,
        description="Angles right outlets add across stories that left outlets omit or downplay",
    )
    center_angle: str = Field(
        default="",
        description="What center sources uniquely contribute. Empty string if nothing distinctive.",
    )

    @field_validator("topic_overview")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic_overview must not be empty")
        return v


class FactCheckVerdict(BaseModel):
    verdict: Literal["confirmed", "disputed", "misleading", "unverifiable"]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Numeric confidence 0–1. "
            "1.0 = multiple concordant primary sources across leans. "
            "0.7 = strong evidence from one direction. "
            "0.4 = ambiguous or contradictory evidence. "
            "0.1 = almost no relevant coverage found."
        ),
    )
    one_line_explanation: str = Field(
        description="Single sentence explaining the verdict"
    )
    evidence_for: list[str] = Field(
        default_factory=list,
        description="Citations supporting the claim (outlet + headline or web title + url)",
    )
    evidence_against: list[str] = Field(
        default_factory=list,
        description="Citations contradicting or contextualising the claim",
    )
    lean_emphasis: LeanSummaries = Field(
        description="What left/center/right coverage emphasises about this claim"
    )
    database_coverage_note: str = Field(
        description="How many sources address this? Is database coverage sufficient?"
    )

    @model_validator(mode="after")
    def require_evidence_unless_unverifiable(self) -> "FactCheckVerdict":
        if self.verdict != "unverifiable" and not self.evidence_for and not self.evidence_against:
            raise ValueError(
                "At least one of evidence_for or evidence_against must be non-empty "
                "for verdicts other than 'unverifiable'"
            )
        return self


class CoverageItem(BaseModel):
    claim: str = Field(
        description="What this side emphasizes that the other side omits or downplays"
    )
    coverage: Literal["omitted", "downplayed"] = Field(
        description=(
            "'omitted' = the other side has zero mention of this topic/angle. "
            "'downplayed' = the other side mentions it but not prominently "
            "(buried late in article, one passing sentence, not in headline/lede)."
        )
    )


class ClusterAnalysisResult(BaseModel):
    summary: str = Field(
        description=(
            "2-3 sentence neutral summary of what happened. "
            "Anchored in facts all sides report. No editorial framing."
        )
    )
    shared_ground: list[str] = Field(
        min_length=1,
        max_length=4,
        description="Facts that most leans report consistently, with inline outlet citations",
    )
    left_not_right: list[CoverageItem] = Field(
        default_factory=list,
        max_length=3,
        description="What left media emphasizes that right media omits or downplays",
    )
    right_not_left: list[CoverageItem] = Field(
        default_factory=list,
        max_length=3,
        description="What right media emphasizes that left media omits or downplays",
    )
    center_angle: str = Field(
        default="",
        description=(
            "What center sources (Reuters, AP, BBC, Al Jazeera) uniquely add or frame "
            "differently from both left and right. "
            "Empty string if center simply splits the difference."
        ),
    )

    @field_validator("shared_ground", mode="before")
    @classmethod
    def coerce_str_to_list(cls, v: Any) -> Any:
        return [v] if isinstance(v, str) else v

    @field_validator("left_not_right", "right_not_left", mode="before")
    @classmethod
    def coerce_coverage_items(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, str):
                result.append({"claim": item, "coverage": "downplayed"})
            else:
                result.append(item)
        return result
