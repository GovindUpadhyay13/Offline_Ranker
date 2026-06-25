"""Streaming reader for candidates.jsonl into typed records.

Parses one JSON object per line and tolerates absent optional fields.
The -1 placeholders in github_activity_score and offer_acceptance_rate are
data sentinels for "no history", so they become None rather than a low value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Iterator, Optional

NO_HISTORY = -1  # sentinel meaning unlinked/absent, must not read as a real score


@dataclass(slots=True)
class Profile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: Optional[float]
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


@dataclass(slots=True)
class Role:
    company: str
    title: str
    start: Optional[date]
    end: Optional[date]
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass(slots=True)
class Education:
    institution: str
    degree: str
    field_of_study: str
    start_year: Optional[int]
    end_year: Optional[int]
    grade: Optional[str]
    tier: str


@dataclass(slots=True)
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: Optional[int]


@dataclass(slots=True)
class Certification:
    name: str
    issuer: str
    year: Optional[int]


@dataclass(slots=True)
class Language:
    language: str
    proficiency: str


@dataclass(slots=True)
class Signals:
    profile_completeness_score: Optional[float]
    signup_date: Optional[date]
    last_active_date: Optional[date]
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: Optional[float]
    avg_response_time_hours: Optional[float]
    skill_assessment_scores: dict
    connection_count: int
    endorsements_received: int
    notice_period_days: Optional[int]
    salary_min: Optional[float]
    salary_max: Optional[float]
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: Optional[float]  # None when source value is the -1 sentinel
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: Optional[float]
    offer_acceptance_rate: Optional[float]  # None when source value is the -1 sentinel
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    profile: Profile
    career_history: list[Role]
    education: list[Education]
    skills: list[Skill]
    certifications: list[Certification]
    languages: list[Language]
    signals: Signals


def read_candidates(path) -> Iterator[Candidate]:
    """Yield one Candidate per non-blank line, holding only that line in memory."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield _candidate(json.loads(line))


def _date(s) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _drop_sentinel(x) -> Optional[float]:
    if x is None or x <= NO_HISTORY:
        return None
    return x


def _candidate(d) -> Candidate:
    return Candidate(
        candidate_id=d["candidate_id"],
        profile=_profile(d.get("profile", {})),
        career_history=[_role(r) for r in d.get("career_history", [])],
        education=[_education(e) for e in d.get("education", [])],
        skills=[_skill(s) for s in d.get("skills", [])],
        certifications=[_certification(c) for c in d.get("certifications", [])],
        languages=[_language(l) for l in d.get("languages", [])],
        signals=_signals(d.get("redrob_signals", {})),
    )


def _profile(p) -> Profile:
    return Profile(
        anonymized_name=p.get("anonymized_name", ""),
        headline=p.get("headline", ""),
        summary=p.get("summary", ""),
        location=p.get("location", ""),
        country=p.get("country", ""),
        years_of_experience=p.get("years_of_experience"),
        current_title=p.get("current_title", ""),
        current_company=p.get("current_company", ""),
        current_company_size=p.get("current_company_size", ""),
        current_industry=p.get("current_industry", ""),
    )


def _role(r) -> Role:
    return Role(
        company=r.get("company", ""),
        title=r.get("title", ""),
        start=_date(r.get("start_date")),
        end=_date(r.get("end_date")),
        duration_months=r.get("duration_months", 0),
        is_current=r.get("is_current", False),
        industry=r.get("industry", ""),
        company_size=r.get("company_size", ""),
        description=r.get("description", ""),
    )


def _education(e) -> Education:
    return Education(
        institution=e.get("institution", ""),
        degree=e.get("degree", ""),
        field_of_study=e.get("field_of_study", ""),
        start_year=e.get("start_year"),
        end_year=e.get("end_year"),
        grade=e.get("grade"),
        tier=e.get("tier", "unknown"),
    )


def _skill(s) -> Skill:
    return Skill(
        name=s.get("name", ""),
        proficiency=s.get("proficiency", ""),
        endorsements=s.get("endorsements", 0),
        duration_months=s.get("duration_months"),
    )


def _certification(c) -> Certification:
    return Certification(name=c.get("name", ""), issuer=c.get("issuer", ""), year=c.get("year"))


def _language(l) -> Language:
    return Language(language=l.get("language", ""), proficiency=l.get("proficiency", ""))


def _signals(s) -> Signals:
    salary = s.get("expected_salary_range_inr_lpa") or {}
    return Signals(
        profile_completeness_score=s.get("profile_completeness_score"),
        signup_date=_date(s.get("signup_date")),
        last_active_date=_date(s.get("last_active_date")),
        open_to_work_flag=s.get("open_to_work_flag", False),
        profile_views_received_30d=s.get("profile_views_received_30d", 0),
        applications_submitted_30d=s.get("applications_submitted_30d", 0),
        recruiter_response_rate=s.get("recruiter_response_rate"),
        avg_response_time_hours=s.get("avg_response_time_hours"),
        skill_assessment_scores=s.get("skill_assessment_scores") or {},
        connection_count=s.get("connection_count", 0),
        endorsements_received=s.get("endorsements_received", 0),
        notice_period_days=s.get("notice_period_days"),
        salary_min=salary.get("min"),
        salary_max=salary.get("max"),
        preferred_work_mode=s.get("preferred_work_mode", ""),
        willing_to_relocate=s.get("willing_to_relocate", False),
        github_activity_score=_drop_sentinel(s.get("github_activity_score")),
        search_appearance_30d=s.get("search_appearance_30d", 0),
        saved_by_recruiters_30d=s.get("saved_by_recruiters_30d", 0),
        interview_completion_rate=s.get("interview_completion_rate"),
        offer_acceptance_rate=_drop_sentinel(s.get("offer_acceptance_rate")),
        verified_email=s.get("verified_email", False),
        verified_phone=s.get("verified_phone", False),
        linkedin_connected=s.get("linkedin_connected", False),
    )
