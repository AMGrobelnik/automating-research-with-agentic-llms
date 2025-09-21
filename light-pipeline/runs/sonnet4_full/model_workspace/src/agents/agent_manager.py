"""AgentManager for coordinating multi-agent interactions in the Cite-and-Challenge protocol."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import time
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answering_agent import AnsweringAgent, AgentResponse
from .challenger_agent import ChallengerAgent, ChallengeReport
from ..config.config_manager import ConfigManager


@dataclass
class SessionMetrics:
    """Metrics for a complete agent interaction session."""
    
    session_id: str
    claim: str
    domain: Optional[str]
    
    # Agent responses
    agent_responses: List[AgentResponse]
    challenge_reports: List[ChallengeReport]
    
    # Timing metrics
    total_processing_time: float
    answering_time: float
    challenging_time: float
    
    # Token usage
    total_tokens: int
    answering_tokens: int
    challenging_tokens: int
    
    # Quality metrics
    requires_revision: bool
    avg_confidence: float
    total_challenges: int
    critical_challenges: int
    
    # Final results
    final_assessment: str
    consensus_reached: bool


@dataclass
class AgentConfiguration:
    """Configuration for agent instances."""
    
    max_answering_agents: int = 2
    challenger_enabled: bool = True
    max_search_results_per_agent: int = 10
    max_tokens_per_response: int = 2000
    confidence_threshold: float = 0.7
    challenge_severity_threshold: float = 0.3
    timeout_seconds: int = 300


class AgentManager:
    """
    Coordination system for managing multiple agents in the Cite-and-Challenge protocol.
    
    Manages the interaction between answering agents and challenger agents,
    ensuring proper communication protocols and resource management.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the agent manager."""
        
        self.config = config or ConfigManager()
        self.agent_config = AgentConfiguration()
        
        # Initialize agents
        self.answering_agents: Dict[str, AnsweringAgent] = {}
        self.challenger_agent: Optional[ChallengerAgent] = None
        
        # Session state
        self.active_sessions: Dict[str, SessionMetrics] = {}
        self.session_counter = 0
        
        # Performance tracking
        self.total_sessions = 0
        self.successful_sessions = 0
        self.failed_sessions = 0
        
        logger.info("AgentManager initialized")
        
        # Setup agents
        asyncio.create_task(self._initialize_agents())
    
    async def _initialize_agents(self):
        """Initialize all agent instances."""
        try:
            # Initialize answering agents
            for i in range(self.agent_config.max_answering_agents):
                agent_id = f"answering_agent_{i+1}"
                agent = AnsweringAgent(
                    agent_id=agent_id,
                    max_search_results=self.agent_config.max_search_results_per_agent,
                    max_tokens_per_response=self.agent_config.max_tokens_per_response,
                    confidence_threshold=self.agent_config.confidence_threshold
                )
                self.answering_agents[agent_id] = agent
                logger.info(f"Initialized {agent_id}")
            
            # Initialize challenger agent
            if self.agent_config.challenger_enabled:
                self.challenger_agent = ChallengerAgent(
                    challenger_id="challenger_agent_1",
                    min_challenge_severity=self.agent_config.challenge_severity_threshold
                )
                logger.info("Initialized challenger_agent_1")
            
            logger.success(f"All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    
    async def process_claim(
        self,
        claim: str,
        domain: Optional[str] = None,
        additional_context: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> SessionMetrics:
        """
        Process a factual claim through the complete Cite-and-Challenge protocol.
        
        Args:
            claim: The factual claim to research and validate
            domain: Domain category (science, health, history, finance)
            additional_context: Additional context for agents
            session_id: Optional session identifier
            
        Returns:
            SessionMetrics with complete session results
        """
        
        # Generate session ID if not provided
        if not session_id:
            self.session_counter += 1
            session_id = f"session_{self.session_counter:04d}_{int(time.time())}"
        
        logger.info(f"Processing claim in {session_id}: {claim[:100]}...")
        start_time = time.time()
        
        try:
            # Phase 1: Parallel answering agent research
            logger.info("Phase 1: Deploying answering agents")
            answering_start = time.time()
            
            agent_responses = await self._deploy_answering_agents(
                claim, domain, additional_context
            )
            
            answering_time = time.time() - answering_start
            logger.info(f"Answering phase completed in {answering_time:.2f}s")
            
            # Phase 2: Challenger agent analysis
            challenge_reports = []
            challenging_time = 0.0
            
            if self.challenger_agent and agent_responses:
                logger.info("Phase 2: Deploying challenger agent")
                challenging_start = time.time()
                
                challenge_reports = await self._deploy_challenger_agent(
                    agent_responses, claim, additional_context
                )
                
                challenging_time = time.time() - challenging_start
                logger.info(f"Challenging phase completed in {challenging_time:.2f}s")
            
            # Phase 3: Analysis and metrics calculation
            session_metrics = await self._calculate_session_metrics(
                session_id=session_id,
                claim=claim,
                domain=domain,
                agent_responses=agent_responses,
                challenge_reports=challenge_reports,
                total_time=time.time() - start_time,
                answering_time=answering_time,
                challenging_time=challenging_time
            )
            
            # Store session for future reference
            self.active_sessions[session_id] = session_metrics
            self.total_sessions += 1
            self.successful_sessions += 1
            
            logger.success(
                f"Session {session_id} completed successfully "
                f"(total time: {session_metrics.total_processing_time:.2f}s)"
            )
            
            return session_metrics
            
        except Exception as e:
            self.failed_sessions += 1
            logger.error(f"Session {session_id} failed: {str(e)}")
            raise
    
    async def _deploy_answering_agents(
        self,
        claim: str,
        domain: Optional[str],
        additional_context: Optional[str]
    ) -> List[AgentResponse]:
        """Deploy answering agents in parallel to research the claim."""
        
        async def research_with_agent(agent: AnsweringAgent) -> Optional[AgentResponse]:
            try:
                return await asyncio.wait_for(
                    agent.research_claim(claim, domain, additional_context),
                    timeout=self.agent_config.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Agent {agent.agent_id} timed out")
                return None
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} failed: {str(e)}")
                return None
        
        # Execute all agents in parallel
        tasks = [
            research_with_agent(agent) 
            for agent in self.answering_agents.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful responses
        successful_responses = []
        for result in results:
            if isinstance(result, AgentResponse):
                successful_responses.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Agent task failed with exception: {result}")
        
        logger.info(
            f"Answering agents completed: {len(successful_responses)}/{len(tasks)} successful"
        )
        
        return successful_responses
    
    async def _deploy_challenger_agent(
        self,
        agent_responses: List[AgentResponse],
        original_claim: str,
        additional_context: Optional[str]
    ) -> List[ChallengeReport]:
        """Deploy challenger agent to analyze all agent responses."""
        
        async def challenge_response(response: AgentResponse) -> Optional[ChallengeReport]:
            try:
                return await asyncio.wait_for(
                    self.challenger_agent.challenge_response(
                        response, original_claim, additional_context
                    ),
                    timeout=self.agent_config.timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Challenger timed out analyzing {response.agent_id}")
                return None
            except Exception as e:
                logger.error(f"Challenger failed analyzing {response.agent_id}: {str(e)}")
                return None
        
        # Challenge all responses in parallel
        tasks = [challenge_response(response) for response in agent_responses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful challenge reports
        successful_reports = []
        for result in results:
            if isinstance(result, ChallengeReport):
                successful_reports.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Challenger task failed with exception: {result}")
        
        logger.info(f"Challenger analysis completed: {len(successful_reports)} reports")
        
        return successful_reports
    
    async def _calculate_session_metrics(
        self,
        session_id: str,
        claim: str,
        domain: Optional[str],
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport],
        total_time: float,
        answering_time: float,
        challenging_time: float
    ) -> SessionMetrics:
        """Calculate comprehensive metrics for the session."""
        
        # Token usage calculations
        answering_tokens = sum(r.token_usage for r in agent_responses)
        challenging_tokens = sum(r.token_usage for r in challenge_reports)
        total_tokens = answering_tokens + challenging_tokens
        
        # Quality metrics
        avg_confidence = (
            sum(r.confidence_score for r in agent_responses) / len(agent_responses)
            if agent_responses else 0.0
        )
        
        all_challenges = []
        for report in challenge_reports:
            all_challenges.extend(report.challenges)
        
        total_challenges = len(all_challenges)
        critical_challenges = len([c for c in all_challenges if c.severity >= 0.8])
        
        # Determine if revision is required
        requires_revision = any(report.requires_revision for report in challenge_reports)
        
        # Generate final assessment
        final_assessment = await self._generate_final_assessment(
            agent_responses, challenge_reports, claim
        )
        
        # Determine consensus
        consensus_reached = self._assess_consensus(agent_responses, challenge_reports)
        
        return SessionMetrics(
            session_id=session_id,
            claim=claim,
            domain=domain,
            agent_responses=agent_responses,
            challenge_reports=challenge_reports,
            total_processing_time=total_time,
            answering_time=answering_time,
            challenging_time=challenging_time,
            total_tokens=total_tokens,
            answering_tokens=answering_tokens,
            challenging_tokens=challenging_tokens,
            requires_revision=requires_revision,
            avg_confidence=avg_confidence,
            total_challenges=total_challenges,
            critical_challenges=critical_challenges,
            final_assessment=final_assessment,
            consensus_reached=consensus_reached
        )
    
    async def _generate_final_assessment(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport],
        claim: str
    ) -> str:
        """Generate final assessment of the claim processing."""
        
        if not agent_responses:
            return f"No successful agent responses obtained for claim: '{claim}'"
        
        # Analyze agent agreement
        supporting_responses = []
        contradicting_responses = []
        
        for response in agent_responses:
            if "SUPPORTED" in response.answer.upper():
                supporting_responses.append(response)
            elif "CONTRADICTED" in response.answer.upper():
                contradicting_responses.append(response)
        
        assessment_parts = []
        
        # Agent consensus analysis
        if len(supporting_responses) > len(contradicting_responses):
            assessment_parts.append(f"MAJORITY SUPPORT: {len(supporting_responses)}/{len(agent_responses)} agents support the claim.")
        elif len(contradicting_responses) > len(supporting_responses):
            assessment_parts.append(f"MAJORITY CONTRADICTION: {len(contradicting_responses)}/{len(agent_responses)} agents contradict the claim.")
        else:
            assessment_parts.append(f"SPLIT OPINION: Equal agent support and contradiction ({len(agent_responses)} total responses).")
        
        # Challenge analysis
        if challenge_reports:
            total_challenges = sum(len(report.challenges) for report in challenge_reports)
            critical_challenges = sum(
                len([c for c in report.challenges if c.severity >= 0.8]) 
                for report in challenge_reports
            )
            
            assessment_parts.append(f"CHALLENGES IDENTIFIED: {total_challenges} total ({critical_challenges} critical).")
            
            if critical_challenges > 0:
                assessment_parts.append("RECOMMENDATION: Significant revision required before acceptance.")
            elif total_challenges >= 5:
                assessment_parts.append("RECOMMENDATION: Moderate revision suggested.")
            else:
                assessment_parts.append("RECOMMENDATION: Minor improvements could enhance quality.")
        
        # Confidence analysis
        avg_confidence = sum(r.confidence_score for r in agent_responses) / len(agent_responses)
        assessment_parts.append(f"AVERAGE CONFIDENCE: {avg_confidence:.2f}/1.0")
        
        return " ".join(assessment_parts)
    
    def _assess_consensus(
        self,
        agent_responses: List[AgentResponse],
        challenge_reports: List[ChallengeReport]
    ) -> bool:
        """Determine if consensus was reached among agents."""
        
        if len(agent_responses) < 2:
            return False
        
        # Check confidence alignment
        confidences = [r.confidence_score for r in agent_responses]
        confidence_std = self._calculate_std(confidences)
        
        if confidence_std > 0.3:  # High variance in confidence
            return False
        
        # Check conclusion alignment
        conclusions = []
        for response in agent_responses:
            if "SUPPORTED" in response.answer.upper():
                conclusions.append("support")
            elif "CONTRADICTED" in response.answer.upper():
                conclusions.append("contradict")
            else:
                conclusions.append("mixed")
        
        # Consensus if majority (>60%) agree
        conclusion_counts = {c: conclusions.count(c) for c in set(conclusions)}
        max_count = max(conclusion_counts.values())
        
        consensus_threshold = 0.6
        return (max_count / len(conclusions)) >= consensus_threshold
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed summary of a specific session."""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "claim": session.claim,
            "domain": session.domain,
            "processing_metrics": {
                "total_time": session.total_processing_time,
                "answering_time": session.answering_time,
                "challenging_time": session.challenging_time,
                "total_tokens": session.total_tokens
            },
            "quality_metrics": {
                "avg_confidence": session.avg_confidence,
                "total_challenges": session.total_challenges,
                "critical_challenges": session.critical_challenges,
                "requires_revision": session.requires_revision,
                "consensus_reached": session.consensus_reached
            },
            "agent_performance": {
                "successful_responses": len(session.agent_responses),
                "challenge_reports": len(session.challenge_reports)
            },
            "final_assessment": session.final_assessment
        }
    
    async def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager performance statistics."""
        
        # Agent statistics
        agent_stats = {}
        for agent_id, agent in self.answering_agents.items():
            agent_stats[agent_id] = agent.get_agent_stats()
        
        challenger_stats = None
        if self.challenger_agent:
            challenger_stats = self.challenger_agent.get_challenger_stats()
        
        # Session statistics
        if self.active_sessions:
            avg_processing_time = sum(
                session.total_processing_time for session in self.active_sessions.values()
            ) / len(self.active_sessions)
            
            avg_tokens_per_session = sum(
                session.total_tokens for session in self.active_sessions.values()
            ) / len(self.active_sessions)
            
            consensus_rate = sum(
                1 for session in self.active_sessions.values() 
                if session.consensus_reached
            ) / len(self.active_sessions)
        else:
            avg_processing_time = 0.0
            avg_tokens_per_session = 0.0
            consensus_rate = 0.0
        
        return {
            "manager_metrics": {
                "total_sessions": self.total_sessions,
                "successful_sessions": self.successful_sessions,
                "failed_sessions": self.failed_sessions,
                "success_rate": self.successful_sessions / self.total_sessions if self.total_sessions > 0 else 0.0,
                "active_sessions": len(self.active_sessions)
            },
            "performance_metrics": {
                "avg_processing_time": avg_processing_time,
                "avg_tokens_per_session": avg_tokens_per_session,
                "consensus_rate": consensus_rate
            },
            "agent_statistics": {
                "answering_agents": agent_stats,
                "challenger_agent": challenger_stats
            },
            "configuration": {
                "max_answering_agents": self.agent_config.max_answering_agents,
                "challenger_enabled": self.agent_config.challenger_enabled,
                "timeout_seconds": self.agent_config.timeout_seconds
            }
        }
    
    async def reset_all_agents(self):
        """Reset all agents and clear session data."""
        
        # Reset answering agents
        for agent in self.answering_agents.values():
            await agent.reset_agent()
        
        # Reset challenger agent
        if self.challenger_agent:
            await self.challenger_agent.reset_challenger()
        
        # Clear session data
        self.active_sessions.clear()
        self.session_counter = 0
        
        logger.info("All agents reset and session data cleared")
    
    async def update_configuration(self, new_config: AgentConfiguration):
        """Update agent manager configuration."""
        
        old_config = self.agent_config
        self.agent_config = new_config
        
        logger.info(
            f"Configuration updated - "
            f"Max agents: {old_config.max_answering_agents} → {new_config.max_answering_agents}, "
            f"Challenger enabled: {old_config.challenger_enabled} → {new_config.challenger_enabled}"
        )
        
        # Reinitialize agents if configuration changed significantly
        if (new_config.max_answering_agents != old_config.max_answering_agents or
            new_config.challenger_enabled != old_config.challenger_enabled):
            
            logger.info("Reinitializing agents due to configuration changes")
            await self._initialize_agents()
    
    async def cleanup(self):
        """Cleanup resources and shutdown agents."""
        
        logger.info("Cleaning up AgentManager resources")
        
        # Reset all agents
        await self.reset_all_agents()
        
        # Clear all references
        self.answering_agents.clear()
        self.challenger_agent = None
        
        logger.info("AgentManager cleanup completed")