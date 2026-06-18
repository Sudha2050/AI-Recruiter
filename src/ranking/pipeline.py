"""
Multi-Stage Ranking Pipeline.

Stage 1: Honeypot & Trap Filter (fast elimination)
Stage 2: Coarse Ranking (heuristic scoring, top 5K)
Stage 3: Semantic Ranking (embedding similarity, top 500)
Stage 4: Fine Ranking (ML model + behavioral integration, top 100)
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import heapq

# Constants used in the pipeline
TOP_K = 100
SCORE_DECIMALS = 4
PREFERRED_LOCATIONS = ['bangalore', 'hyderabad', 'delhi', 'mumbai', 'pune', 'chennai']
REASONING_MAX_LEN = 200

from config.settings import (
    STAGE1_CONFIG, STAGE2_CONFIG, STAGE3_CONFIG, STAGE4_CONFIG,
    JD_SCORING_WEIGHTS, BEHAVIORAL_MULTIPLIERS
)
from src.features.candidate_features import CandidateFeatureExtractor, CandidateFeatures
from src.embeddings.similarity import EmbeddingSimilarity


@dataclass
class RankedCandidate:
    """A candidate with their final rank, score, and reasoning."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    features: CandidateFeatures


class RankingPipeline:
    """
    Four-stage ranking pipeline for intelligent candidate discovery.
    """
    
    def __init__(self, jd_parsed, embedding_index=None, model=None):
        self.jd = jd_parsed
        self.feature_extractor = CandidateFeatureExtractor(jd_parsed)
        self.embedding_sim = embedding_index
        self.model = model  # LightGBM model for Stage 4
        
        # Pre-compute JD embedding if available
        self.jd_embedding = None
        if embedding_index:
            self.jd_embedding = embedding_index.get_jd_embedding(jd_parsed.raw_text)
    
    def rank(self, candidates: List[Dict]) -> List[RankedCandidate]:
        """
        Run full ranking pipeline on candidate list.
        Returns top 100 ranked candidates.
        """
        print(f"Starting ranking pipeline with {len(candidates)} candidates...")
        
        # Stage 1: Filter honeypots and traps
        stage1_candidates = self._stage1_filter(candidates)
        print(f"Stage 1: {len(stage1_candidates)} candidates after filtering")
        
        # Stage 2: Coarse heuristic ranking
        stage2_candidates = self._stage2_coarse_rank(stage1_candidates)
        print(f"Stage 2: {len(stage2_candidates)} candidates after coarse ranking")
        
        # Stage 3: Semantic ranking with embeddings
        stage3_candidates = self._stage3_semantic_rank(stage2_candidates)
        print(f"Stage 3: {len(stage3_candidates)} candidates after semantic ranking")
        
        # Stage 4: Fine ranking with ML model
        final_candidates = self._stage4_fine_rank(stage3_candidates)
        print(f"Stage 4: Final top {len(final_candidates)} candidates")
        
        return final_candidates
    
    def _stage1_filter(self, candidates: List[Dict]) -> List[Dict]:
        """
        Stage 1: Eliminate honeypots, keyword stuffers, and ghost profiles.
        Fast filter based on hard rules.
        """
        filtered = []
        
        for candidate in candidates:
            features = self.feature_extractor.extract(candidate)
            
            # Hard elimination rules
            if features.honeypot_risk_score > 0.7:
                continue  # Definite honeypot
            
            if features.impossible_profile_flag:
                continue  # Impossible profile
            
            if features.ghost_profile_flag and features.response_rate < 0.05:
                continue  # Ghost profile with no engagement
            
            # Soft filters (keep but mark for penalty)
            if features.keyword_stuffer_flag:
                # Keep but will be heavily penalized in later stages
                pass
            
            filtered.append(candidate)
        
        return filtered
    
    def _stage2_coarse_rank(self, candidates: List[Dict]) -> List[Tuple[Dict, float, CandidateFeatures]]:
        """
        Stage 2: Fast heuristic scoring to get top 5K.
        Uses rule-based scoring with JD weights.
        """
        scored = []
        
        for candidate in candidates:
            features = self.feature_extractor.extract(candidate)
            score = self._compute_heuristic_score(features)
            scored.append((candidate, score, features))
        
        # Sort by score descending, take top 5K
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:STAGE2_CONFIG.max_candidates]
    
    def _compute_heuristic_score(self, features: CandidateFeatures) -> float:
        """
        Compute heuristic score based on JD requirements.
        Fast rule-based scoring.
        """
        score = 0.0
        
        # Must-have bonuses (binary)
        if features.has_embedding_skills:
            score += JD_SCORING_WEIGHTS["has_embedding_retrieval_exp"]
        if features.has_vector_db_skills:
            score += JD_SCORING_WEIGHTS["has_vector_db_exp"]
        if features.has_python:
            score += JD_SCORING_WEIGHTS["has_python_strong"]
        if features.has_eval_framework_exp:
            score += JD_SCORING_WEIGHTS["has_ranking_eval_exp"]
        
        # Experience band
        exp = features.years_experience
        if 5 <= exp <= 9:
            score += JD_SCORING_WEIGHTS["years_in_range_5_9"]
        elif 4 <= exp <= 10:
            score += JD_SCORING_WEIGHTS["years_in_range_5_9"] * 0.5
        
        # Product company preference
        if features.has_product_company_current:
            score += JD_SCORING_WEIGHTS["product_company_exp"]
        if features.has_consulting_only_career:
            score += JD_SCORING_WEIGHTS["consulting_only_career"]
        
        # Nice-to-haves
        if features.has_llm_finetuning:
            score += JD_SCORING_WEIGHTS["has_llm_finetuning_exp"]
        if features.has_ltr_exp:
            score += JD_SCORING_WEIGHTS["has_ltr_exp"]
        
        # Title match
        score += features.title_match_score * 20
        
        # Skill depth
        score += min(features.ai_core_skill_depth, 50)  # Cap at 50
        
        # Penalties
        if features.honeypot_risk_score > 0.3:
            score -= 20 * features.honeypot_risk_score
        
        # Apply behavioral multiplier
        score *= features.behavioral_multiplier
        
        return score
    
    def _stage3_semantic_rank(self, candidates: List[Tuple[Dict, float, CandidateFeatures]]) -> List[Tuple[Dict, float, CandidateFeatures]]:
        """
        Stage 3: Semantic ranking using precomputed embeddings.
        Reranks top 5K using cosine similarity between JD and candidate embeddings.
        """
        if self.embedding_sim is None or self.jd_embedding is None:
            # Fallback: skip semantic ranking, return top 500 from stage 2
            return candidates[:STAGE3_CONFIG.max_candidates]
        
        scored = []
        
        for candidate, heuristic_score, features in candidates:
            # Get candidate embedding
            candidate_text = self._candidate_to_text(candidate)
            candidate_embedding = self.embedding_sim.get_embedding(candidate_text)
            
            # Compute similarity
            similarity = self.embedding_sim.cosine_similarity(
                self.jd_embedding, candidate_embedding
            )
            
            # Combine heuristic and semantic scores
            # Semantic similarity weighted heavily for quality fit
            combined_score = heuristic_score * 0.6 + similarity * 100 * 0.4
            
            scored.append((candidate, combined_score, features))
        
        # Resort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:STAGE3_CONFIG.max_candidates]
    
    def _stage4_fine_rank(self, candidates: List[Tuple[Dict, float, CandidateFeatures]]) -> List[RankedCandidate]:
        """
        Stage 4: Fine ranking with ML model and behavioral integration.
        Produces final top 100 with detailed reasoning.
        """
        if self.model is not None:
            # Use trained ML model for scoring
            final_scored = self._model_based_rank(candidates)
        else:
            # Fallback: refined heuristic with full feature integration
            final_scored = self._refined_heuristic_rank(candidates)
        
        # Generate reasoning for top 100
        ranked = []
        for i, (candidate, score, features) in enumerate(final_scored[:TOP_K]):
            reasoning = self._generate_reasoning(candidate, features, score, i+1)
            ranked.append(RankedCandidate(
                candidate_id=candidate["candidate_id"],
                rank=i+1,
                score=round(score, SCORE_DECIMALS),
                reasoning=reasoning,
                features=features
            ))
        
        return ranked
    
    def _model_based_rank(self, candidates: List[Tuple[Dict, float, CandidateFeatures]]) -> List[Tuple[Dict, float, CandidateFeatures]]:
        """Use trained LightGBM model for final scoring."""
        # Extract feature vectors
        X = []
        for candidate, _, features in candidates:
            feature_vector = self._features_to_vector(features)
            X.append(feature_vector)
        
        X = np.array(X)
        
        # Predict scores
        scores = self.model.predict(X)
        
        # Combine with candidates
        scored = []
        for i, (candidate, _, features) in enumerate(candidates):
            scored.append((candidate, scores[i], features))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _refined_heuristic_rank(self, candidates: List[Tuple[Dict, float, CandidateFeatures]]) -> List[Tuple[Dict, float, CandidateFeatures]]:
        """
        Refined heuristic scoring for final ranking.
        More nuanced than Stage 2 with full behavioral integration.
        """
        scored = []
        
        for candidate, prev_score, features in candidates:
            # Base score from previous stage
            score = prev_score
            
            # Fine-tune with detailed behavioral signals
            # Recency and engagement
            if features.recent_activity_days < 7:
                score *= 1.05
            elif features.recent_activity_days > 90:
                score *= 0.9
            
            # Response quality
            if features.response_rate > 0.8:
                score *= 1.08
            elif features.response_rate < 0.2:
                score *= 0.7
            
            # Interview reliability
            if features.interview_completion_rate > 0.8:
                score *= 1.03
            
            # GitHub activity (for engineering roles)
            if features.github_score > 20:
                score *= 1.05
            
            # Salary alignment (JD mentions no specific range, but check for reasonableness)
            # No penalty for reasonable expectations
            
            # Location fit
            location = candidate["profile"].get("location", "").lower()
            if any(loc.lower() in location for loc in PREFERRED_LOCATIONS):
                score *= 1.03
            if features.willing_to_relocate:
                score *= 1.02
            
            # Notice period (JD prefers sub-30, can buy out 30)
            if features.notice_period_days <= 30:
                score *= 1.05
            elif features.notice_period_days <= 60:
                score *= 1.0
            else:
                score *= 0.9
            
            # Profile completeness
            if features.profile_completeness > 80:
                score *= 1.02
            
            scored.append((candidate, score, features))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def _features_to_vector(self, features: CandidateFeatures) -> List[float]:
        """Convert features to numeric vector for ML model."""
        return [
            features.years_experience,
            features.title_match_score,
            features.product_company_ratio,
            features.consulting_company_ratio,
            features.ai_core_skill_count,
            features.ai_core_skill_depth,
            features.has_embedding_skills,
            features.has_vector_db_skills,
            features.has_python,
            features.has_eval_framework_exp,
            features.has_llm_finetuning,
            features.has_ltr_exp,
            features.behavioral_multiplier,
            features.availability_score,
            features.response_rate,
            features.recent_activity_days,
            features.profile_completeness,
            features.honeypot_risk_score,
        ]
    
    def _candidate_to_text(self, candidate: Dict) -> str:
        """Convert candidate to text for embedding."""
        parts = [
            candidate["profile"].get("headline", ""),
            candidate["profile"].get("summary", ""),
            candidate["profile"].get("current_title", ""),
        ]
        # Add career descriptions
        for role in candidate.get("career_history", [])[:3]:
            parts.append(role.get("description", ""))
        # Add skills
        skills = [s["name"] for s in candidate.get("skills", [])]
        parts.append(" ".join(skills))
        
        return " ".join(parts)
    
    def _generate_reasoning(self, candidate: Dict, features: CandidateFeatures, 
                           score: float, rank: int) -> str:
        """
        Generate 1-2 sentence reasoning for ranking.
        Must be specific, honest, and connected to JD requirements.
        """
        profile = candidate["profile"]
        title = profile.get("current_title", "")
        years = features.years_experience
        
        # Build reasoning components
        strengths = []
        concerns = []
        
        # Title and experience
        if features.title_match_score > 0.7:
            strengths.append(f"{title} with {years:.1f} years")
        elif features.title_match_score > 0.4:
            strengths.append(f"{years:.1f} years experience in {title}")
        else:
            concerns.append(f"title mismatch ({title})")
        
        # Key skills
        if features.has_embedding_skills:
            strengths.append("embedding/retrieval expertise")
        if features.has_vector_db_skills:
            strengths.append("vector DB experience")
        if features.has_python:
            strengths.append("strong Python")
        if features.has_eval_framework_exp:
            strengths.append("ranking evaluation background")
        
        # Company type
        if features.has_product_company_current:
            strengths.append("product company experience")
        elif features.has_consulting_only_career:
            concerns.append("consulting-only background")
        
        # Behavioral
        if features.open_to_work:
            strengths.append("actively looking")
        if features.response_rate > 0.7:
            strengths.append(f"high response rate ({features.response_rate:.0%})")
        elif features.response_rate < 0.2:
            concerns.append("low response rate")
        
        if features.notice_period_days <= 30:
            strengths.append("short notice period")
        elif features.notice_period_days > 90:
            concerns.append(f"long notice ({features.notice_period_days}d)")
        
        # Honeypot concerns
        if features.honeypot_risk_score > 0.3:
            concerns.append("some profile inconsistencies")
        
        # Compose reasoning
        if rank <= 10:
            # Top candidates: emphasize strengths
            reason = f"{title} with {years:.1f}y; {', '.join(strengths[:3])}"
            if concerns and len(reason) < 150:
                reason += f"; note: {concerns[0]}"
        elif rank <= 50:
            # Mid candidates: balanced
            reason = f"{title}, {years:.1f}y; {', '.join(strengths[:2])}"
            if concerns:
                reason += f"; concern: {concerns[0]}"
        else:
            # Lower candidates: honest about limitations
            reason = f"{title}, {years:.1f}y; {strengths[0] if strengths else 'adjacent skills'}"
            if concerns:
                reason += f"; {concerns[0]}"
            reason += ". Included for diversity/experience signals."

        # Ensure length constraint
        if len(reason) > REASONING_MAX_LEN:
            reason = reason[:REASONING_MAX_LEN - 3] + "..."

        return reason