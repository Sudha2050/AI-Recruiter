"""
Comprehensive candidate feature extraction.
Extracts 50+ features from profile, career, skills, and signals.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, date
from collections import Counter

from config.settings import (
    PRODUCT_COMPANIES, CONSULTING_COMPANIES, AI_CORE_SKILLS,
    PREFERRED_LOCATIONS, BEHAVIORAL_MULTIPLIERS
)


@dataclass
class CandidateFeatures:
    """All engineered features for a single candidate."""
    candidate_id: str
    
    # ── Profile Features ──
    years_experience: float
    current_title: str
    title_category: str  # "ai_ml", "engineering", "data", "other"
    title_match_score: float  # 0-1 similarity to target title
    
    # ── Career Features ──
    num_companies: int
    avg_tenure_months: float
    product_company_ratio: float
    consulting_company_ratio: float
    has_product_company_current: bool
    has_consulting_only_career: bool
    career_progression_score: float
    max_company_size: str
    
    # ── Skill Features ──
    ai_core_skill_count: int
    ai_core_skill_depth: float  # weighted by proficiency + duration
    top_skill_proficiency: str
    skill_diversity: float
    has_embedding_skills: bool
    has_vector_db_skills: bool
    has_python: bool
    has_eval_framework_exp: bool
    has_llm_finetuning: bool
    has_ltr_exp: bool
    
    # ── Education Features ──
    highest_degree: str
    highest_tier: str  # tier_1, tier_2, etc.
    field_relevance: float  # CS/ML/AI = 1.0, other = 0.5
    
    # ── Signal Features ──
    profile_completeness: float
    open_to_work: bool
    response_rate: float
    response_time_hours: float
    recent_activity_days: int
    recruiter_saves_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    github_score: float
    search_appearances_30d: int
    notice_period_days: int
    expected_salary_min: float
    expected_salary_max: float
    preferred_work_mode: str
    willing_to_relocate: bool
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool
    
    # ── Derived Behavioral Score ──
    behavioral_multiplier: float
    availability_score: float
    
    # ── Honeypot Risk Flags ──
    honeypot_risk_score: float  # 0 = safe, 1 = definitely honeypot
    impossible_profile_flag: bool
    keyword_stuffer_flag: bool
    ghost_profile_flag: bool


class CandidateFeatureExtractor:
    """
    Extracts all features from a raw candidate JSON object.
    """
    
    # Title category mapping
    AI_ML_TITLES = {
        "AI Engineer", "ML Engineer", "Machine Learning Engineer", 
        "Data Scientist", "Applied ML Engineer", "NLP Engineer",
        "Recommendation Systems Engineer", "Search Engineer",
        "Backend Engineer", "Data Engineer", "Software Engineer",
        "Full Stack Developer", "DevOps Engineer", "Cloud Engineer",
        "Frontend Engineer", "Java Developer", ".NET Developer",
        "Mobile Developer", "QA Engineer", "Python Developer",
    }
    
    ENGINEERING_TITLES = {
        "Software Engineer", "Backend Engineer", "Frontend Engineer",
        "Full Stack Developer", "DevOps Engineer", "Cloud Engineer",
        "Java Developer", ".NET Developer", "Mobile Developer",
        "QA Engineer", "Data Engineer",
    }
    
    DATA_TITLES = {
        "Data Scientist", "Data Analyst", "Analytics Engineer",
        "Business Analyst", "Data Engineer",
    }
    
    # Vector DB skills from JD
    VECTOR_DB_SKILLS = {
        "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
        "Elasticsearch", "FAISS", "Chroma", "Redis",
    }
    
    EMBEDDING_SKILLS = {
        "Embeddings", "Sentence Transformers", "Hugging Face Transformers",
        "OpenAI Embeddings", "BGE", "E5", "Vector Search", "Information Retrieval",
    }
    
    EVAL_SKILLS = {
        "NDCG", "MRR", "MAP", "A/B Testing", "Offline Evaluation",
        "Online Evaluation", "Ranking Evaluation",
    }
    
    LLM_FINETUNE_SKILLS = {
        "LoRA", "PEFT", "QLoRA", "Fine-tuning LLMs", "Parameter-Efficient Fine-Tuning",
    }
    
    LTR_SKILLS = {
        "XGBoost", "LightGBM", "Learning-to-Rank", "RankNet", "LambdaMART",
        "Recommendation Systems",
    }
    
    def __init__(self, jd_parsed):
        self.jd = jd_parsed
        
    def extract(self, candidate: Dict) -> CandidateFeatures:
        """Extract all features from a candidate dict."""
        
        profile = candidate["profile"]
        career = candidate["career_history"]
        skills = candidate["skills"]
        education = candidate.get("education", [])
        signals = candidate["redrob_signals"]
        
        # Basic profile features
        years_exp = profile.get("years_of_experience", 0)
        current_title = profile.get("current_title", "")
        
        # Title categorization
        title_category = self._categorize_title(current_title)
        title_match_score = self._compute_title_match(current_title, title_category)
        
        # Career features
        career_features = self._extract_career_features(career)
        
        # Skill features
        skill_features = self._extract_skill_features(skills)
        
        # Education features
        edu_features = self._extract_education_features(education)
        
        # Signal features
        signal_features = self._extract_signal_features(signals)
        
        # Behavioral multiplier
        behavioral_mult = self._compute_behavioral_multiplier(signals)
        availability = self._compute_availability_score(signals)
        
        # Honeypot detection
        honeypot_score, impossible, stuffer, ghost = self._detect_honeypot(
            candidate, years_exp, career, skills, signals
        )
        
        return CandidateFeatures(
            candidate_id=candidate["candidate_id"],
            years_experience=years_exp,
            current_title=current_title,
            title_category=title_category,
            title_match_score=title_match_score,
            num_companies=career_features["num_companies"],
            avg_tenure_months=career_features["avg_tenure"],
            product_company_ratio=career_features["product_ratio"],
            consulting_company_ratio=career_features["consulting_ratio"],
            has_product_company_current=career_features["has_product_current"],
            has_consulting_only_career=career_features["consulting_only"],
            career_progression_score=career_features["progression"],
            max_company_size=career_features["max_size"],
            ai_core_skill_count=skill_features["ai_core_count"],
            ai_core_skill_depth=skill_features["ai_core_depth"],
            top_skill_proficiency=skill_features["top_proficiency"],
            skill_diversity=skill_features["diversity"],
            has_embedding_skills=skill_features["has_embedding"],
            has_vector_db_skills=skill_features["has_vector_db"],
            has_python=skill_features["has_python"],
            has_eval_framework_exp=skill_features["has_eval"],
            has_llm_finetuning=skill_features["has_llm_ft"],
            has_ltr_exp=skill_features["has_ltr"],
            highest_degree=edu_features["highest_degree"],
            highest_tier=edu_features["highest_tier"],
            field_relevance=edu_features["field_relevance"],
            profile_completeness=signals.get("profile_completeness_score", 0),
            open_to_work=signals.get("open_to_work_flag", False),
            response_rate=signals.get("recruiter_response_rate", 0),
            response_time_hours=signals.get("avg_response_time_hours", 999),
            recent_activity_days=self._days_since_active(signals.get("last_active_date")),
            recruiter_saves_30d=signals.get("saved_by_recruiters_30d", 0),
            interview_completion_rate=signals.get("interview_completion_rate", 0),
            offer_acceptance_rate=signals.get("offer_acceptance_rate", -1),
            github_score=signals.get("github_activity_score", -1),
            search_appearances_30d=signals.get("search_appearance_30d", 0),
            notice_period_days=signals.get("notice_period_days", 90),
            expected_salary_min=signals.get("expected_salary_range_inr_lpa", {}).get("min", 0),
            expected_salary_max=signals.get("expected_salary_range_inr_lpa", {}).get("max", 0),
            preferred_work_mode=signals.get("preferred_work_mode", "flexible"),
            willing_to_relocate=signals.get("willing_to_relocate", False),
            verified_email=signals.get("verified_email", False),
            verified_phone=signals.get("verified_phone", False),
            linkedin_connected=signals.get("linkedin_connected", False),
            behavioral_multiplier=behavioral_mult,
            availability_score=availability,
            honeypot_risk_score=honeypot_score,
            impossible_profile_flag=impossible,
            keyword_stuffer_flag=stuffer,
            ghost_profile_flag=ghost,
        )
    
    def _categorize_title(self, title: str) -> str:
        """Categorize job title into ai_ml, engineering, data, or other."""
        title_lower = title.lower()
        if any(t.lower() in title_lower for t in self.AI_ML_TITLES):
            if any(ml in title_lower for ml in ["ml", "ai", "machine learning", "data scientist"]):
                return "ai_ml"
            return "engineering"
        if any(t.lower() in title_lower for t in self.DATA_TITLES):
            return "data"
        return "other"
    
    def _compute_title_match(self, title: str, category: str) -> float:
        """Score how well the title matches the JD target."""
        target = self.jd.title_target.lower()
        title_lower = title.lower()
        
        # Direct match
        if target in title_lower or title_lower in target:
            return 1.0
        
        # Category match
        if category == "ai_ml":
            return 0.8
        elif category == "engineering":
            return 0.6
        elif category == "data":
            return 0.4
        else:
            return 0.2
    
    def _extract_career_features(self, career: List[Dict]) -> Dict:
        """Extract career trajectory features."""
        if not career:
            return {
                "num_companies": 0, "avg_tenure": 0, "product_ratio": 0,
                "consulting_ratio": 0, "has_product_current": False,
                "consulting_only": False, "progression": 0, "max_size": "1-10"
            }
        
        companies = [c["company"] for c in career]
        durations = [c.get("duration_months", 0) for c in career]
        
        # Company type analysis
        product_count = sum(1 for c in companies if c in PRODUCT_COMPANIES)
        consulting_count = sum(1 for c in companies if c in CONSULTING_COMPANIES)
        total = len(companies)
        
        product_ratio = product_count / total if total > 0 else 0
        consulting_ratio = consulting_count / total if total > 0 else 0
        
        # Current company type
        current_company = career[0]["company"] if career else ""
        has_product_current = current_company in PRODUCT_COMPANIES
        
        # Consulting-only check (JD disqualifier)
        consulting_only = consulting_count > 0 and product_count == 0 and consulting_ratio >= 0.8
        
        # Career progression (simple heuristic: title seniority over time)
        progression = self._compute_progression(career)
        
        # Company size
        sizes = [c.get("company_size", "1-10") for c in career]
        max_size = max(sizes, key=lambda s: self._size_to_int(s))
        
        return {
            "num_companies": total,
            "avg_tenure": sum(durations) / len(durations) if durations else 0,
            "product_ratio": product_ratio,
            "consulting_ratio": consulting_ratio,
            "has_product_current": has_product_current,
            "consulting_only": consulting_only,
            "progression": progression,
            "max_size": max_size,
        }
    
    def _compute_progression(self, career: List[Dict]) -> float:
        """Compute career progression score."""
        if len(career) < 2:
            return 0.5
        
        # Simple heuristic: check if titles show progression
        seniority_keywords = ["senior", "lead", "staff", "principal", "architect", "manager"]
        progression_score = 0.5
        
        for i, role in enumerate(career):
            title = role.get("title", "").lower()
            seniority = sum(1 for kw in seniority_keywords if kw in title)
            # More recent roles should have higher seniority
            recency_weight = 1.0 - (i / len(career)) * 0.5
            progression_score += seniority * recency_weight * 0.1
        
        return min(1.0, progression_score)
    
    def _size_to_int(self, size: str) -> int:
        """Convert company size string to numeric for comparison."""
        size_map = {
            "1-10": 1, "11-50": 2, "51-200": 3, "201-500": 4,
            "501-1000": 5, "1001-5000": 6, "5001-10000": 7, "10001+": 8
        }
        return size_map.get(size, 0)
    
    def _extract_skill_features(self, skills: List[Dict]) -> Dict:
        """Extract comprehensive skill features."""
        if not skills:
            return {
                "ai_core_count": 0, "ai_core_depth": 0, "top_proficiency": "beginner",
                "diversity": 0, "has_embedding": False, "has_vector_db": False,
                "has_python": False, "has_eval": False, "has_llm_ft": False,
                "has_ltr": False,
            }
        
        skill_names = [s["name"] for s in skills]
        proficiencies = [s.get("proficiency", "beginner") for s in skills]
        endorsements = [s.get("endorsements", 0) for s in skills]
        durations = [s.get("duration_months", 0) for s in skills]
        
        # Proficiency scoring
        prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
        prof_scores = [prof_map.get(p, 1) for p in proficiencies]
        
        # AI core skills
        ai_core_count = sum(1 for s in skill_names if s in AI_CORE_SKILLS)
        ai_core_depth = sum(
            prof_scores[i] * (1 + endorsements[i] / 10) * (1 + durations[i] / 12)
            for i, s in enumerate(skill_names) if s in AI_CORE_SKILLS
        )
        
        # Specific skill checks
        has_embedding = any(s in self.EMBEDDING_SKILLS for s in skill_names)
        has_vector_db = any(s in self.VECTOR_DB_SKILLS for s in skill_names)
        has_python = "Python" in skill_names
        has_eval = any(s in self.EVAL_SKILLS for s in skill_names)
        has_llm_ft = any(s in self.LLM_FINETUNE_SKILLS for s in skill_names)
        has_ltr = any(s in self.LTR_SKILLS for s in skill_names)
        
        # Diversity (unique skill categories)
        categories = set()
        for s in skill_names:
            if s in AI_CORE_SKILLS:
                categories.add("ai_ml")
            elif s in ENGINEERING_SKILLS:
                categories.add("engineering")
            else:
                categories.add("other")
        
        return {
            "ai_core_count": ai_core_count,
            "ai_core_depth": ai_core_depth,
            "top_proficiency": max(proficiencies, key=lambda p: prof_map.get(p, 1)),
            "diversity": len(categories) / 3.0,
            "has_embedding": has_embedding,
            "has_vector_db": has_vector_db,
            "has_python": has_python,
            "has_eval": has_eval,
            "has_llm_ft": has_llm_ft,
            "has_ltr": has_ltr,
        }
    
    def _extract_education_features(self, education: List[Dict]) -> Dict:
        """Extract education features."""
        if not education:
            return {
                "highest_degree": "Unknown", "highest_tier": "tier_4",
                "field_relevance": 0.3,
            }
        
        # Find highest degree and tier
        degree_order = {"Ph.D": 4, "M.Tech": 3, "M.E.": 3, "M.Sc": 3, 
                       "M.S.": 3, "B.Tech": 2, "B.E.": 2, "B.Sc": 1}
        
        highest_degree = "Unknown"
        highest_tier = "tier_4"
        best_degree_score = 0
        
        for edu in education:
            degree = edu.get("degree", "")
            tier = edu.get("tier", "tier_4")
            field = edu.get("field_of_study", "").lower()
            
            degree_score = degree_order.get(degree, 0)
            tier_score = int(tier.replace("tier_", "")) if tier.startswith("tier_") else 4
            
            if degree_score > best_degree_score or \
               (degree_score == best_degree_score and tier_score < int(highest_tier.replace("tier_", ""))):
                highest_degree = degree
                highest_tier = tier
                best_degree_score = degree_score
        
        # Field relevance
        relevant_fields = ["computer science", "machine learning", "artificial intelligence",
                          "data science", "information technology", "statistics", "mathematics"]
        field_relevance = 1.0 if any(f in education[0].get("field_of_study", "").lower() 
                                    for f in relevant_fields) else 0.5
        
        return {
            "highest_degree": highest_degree,
            "highest_tier": highest_tier,
            "field_relevance": field_relevance,
        }
    
    def _extract_signal_features(self, signals: Dict) -> Dict:
        """Extract and normalize signal features."""
        # Already handled in main extract method
        return {}
    
    def _compute_behavioral_multiplier(self, signals: Dict) -> float:
        """
        Compute behavioral multiplier based on redrob signals.
        This is a multiplicative factor applied to base score.
        """
        multiplier = 1.0
        
        # Availability boost
        if signals.get("open_to_work_flag", False):
            multiplier *= BEHAVIORAL_MULTIPLIERS["open_to_work"]
        
        # Response quality
        response_rate = signals.get("recruiter_response_rate", 0)
        if response_rate > 0.7:
            multiplier *= BEHAVIORAL_MULTIPLIERS["high_response_rate"]
        elif response_rate < 0.2:
            multiplier *= BEHAVIORAL_MULTIPLIERS["low_response_rate"]
        
        # Recency
        days_active = self._days_since_active(signals.get("last_active_date"))
        if days_active < 30:
            multiplier *= BEHAVIORAL_MULTIPLIERS["recent_activity"]
        elif days_active > 180:
            multiplier *= BEHAVIORAL_MULTIPLIERS["inactive_6mo"]
        
        # Recruiter interest
        if signals.get("saved_by_recruiters_30d", 0) > 5:
            multiplier *= BEHAVIORAL_MULTIPLIERS["saved_by_recruiters"]
        
        # Verification
        if signals.get("verified_email", False) and signals.get("verified_phone", False):
            multiplier *= BEHAVIORAL_MULTIPLIERS["verified_profile"]
        
        # Notice period
        if signals.get("notice_period_days", 90) > 90:
            multiplier *= BEHAVIORAL_MULTIPLIERS["long_notice_period"]
        
        # Ghost profile check
        if days_active > 180 and response_rate < 0.1:
            multiplier *= BEHAVIORAL_MULTIPLIERS["ghost_profile"]
        
        return max(0.1, min(2.0, multiplier))  # Clamp between 0.1 and 2.0
    
    def _compute_availability_score(self, signals: Dict) -> float:
        """Score how available/hirable the candidate is."""
        score = 0.5
        
        if signals.get("open_to_work_flag", False):
            score += 0.3
        
        response_rate = signals.get("recruiter_response_rate", 0)
        score += response_rate * 0.2
        
        if signals.get("notice_period_days", 90) <= 30:
            score += 0.15
        
        return min(1.0, score)
    
    def _days_since_active(self, last_active: str) -> int:
        """Compute days since last activity."""
        if not last_active:
            return 999
        try:
            last_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            today = date(2026, 6, 17)  # Current date from context
            return (today - last_date).days
        except:
            return 999
    
    def _detect_honeypot(self, candidate: Dict, years_exp: float, 
                        career: List[Dict], skills: List[Dict], 
                        signals: Dict) -> Tuple[float, bool, bool, bool]:
        """
        Detect honeypot candidates with impossible or suspicious profiles.
        Returns: (risk_score, impossible_flag, stuffer_flag, ghost_flag)
        """
        risk_score = 0.0
        impossible = False
        stuffer = False
        ghost = False
        
        # Check 1: Experience vs company age impossibility
        for role in career:
            company = role.get("company", "")
            duration = role.get("duration_months", 0)
            start_year = self._extract_year(role.get("start_date", ""))
            
            # Check if experience at company exceeds plausible company tenure
            # or if total experience is impossible given career timeline
            if years_exp > 0 and len(career) > 0:
                total_career_months = sum(r.get("duration_months", 0) for r in career)
                # Large gap between claimed exp and actual career months
                if years_exp * 12 > total_career_months * 1.5:
                    risk_score += 0.3
        
        # Check 2: Expert skills with zero endorsements/duration (skip for freshers)
        title = candidate["profile"].get("current_title", "").lower()
        is_fresher = (years_exp < 1.5) or any(t in title for t in ['student', 'intern', 'graduate', 'fresher'])

        expert_no_validation = 0
        for skill in skills:
            if skill.get("proficiency") == "expert" and \
               skill.get("endorsements", 0) == 0 and \
               skill.get("duration_months", 0) < 6:
                expert_no_validation += 1
        
        if expert_no_validation >= 3 and not is_fresher:
            risk_score += 0.4
            stuffer = True
        
        # Check 3: Skill-title mismatch (keyword stuffing) (skip for freshers)
        has_ml_title = any(t in title for t in ["ml", "ai", "machine learning", "data scientist"])
        
        ai_skill_count = sum(1 for s in skills if s["name"] in AI_CORE_SKILLS)
        if ai_skill_count > 10 and not has_ml_title and not is_fresher:
            risk_score += 0.2
            stuffer = True
        
        # Check 4: Ghost profile
        if signals.get("profile_completeness_score", 0) < 30:
            risk_score += 0.1
        if signals.get("recruiter_response_rate", 0) < 0.05:
            risk_score += 0.2
            ghost = True
        
        # Check 5: Impossible salary range
        salary = signals.get("expected_salary_range_inr_lpa", {})
        if salary.get("min", 0) > salary.get("max", 0):
            risk_score += 0.3
            impossible = True
        
        # Check 6: Suspicious assessment scores
        assessments = signals.get("skill_assessment_scores", {})
        if assessments:
            avg_score = sum(assessments.values()) / len(assessments)
            if avg_score > 95 or avg_score < 10:  # Too perfect or too low
                risk_score += 0.1
        
        # Final classification
        if risk_score > 0.6:
            impossible = True
        
        return min(1.0, risk_score), impossible, stuffer, ghost