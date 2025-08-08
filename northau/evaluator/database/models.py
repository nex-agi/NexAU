"""SQLModel database models for the AutoTuning system."""

from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
from enum import Enum


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionMode(str, Enum):
    EVALUATION = "evaluation"


# Base models with common fields
class TimestampedBase(SQLModel):
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


# Session model
class SessionBase(SQLModel):
    name: Optional[str] = Field(default=None, max_length=200)
    mode: SessionMode = Field(nullable=False)
    dataset_name: str = Field(max_length=100, nullable=False)
    dataset_version: str = Field(max_length=20, nullable=False)
    initial_config_id: Optional[str] = Field(default=None, max_length=100)
    parameters: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Results summary
    total_experiments: Optional[int] = Field(default=0)
    best_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    best_config_id: Optional[str] = Field(default=None, max_length=100)
    convergence_reached: Optional[bool] = Field(default=False)
    
    # Resource tracking
    total_time_seconds: Optional[int] = Field(default=0)
    total_tokens: Optional[int] = Field(default=0)
    total_cost_usd: Optional[float] = Field(default=0.0, ge=0.0)


class Session(SessionBase, TimestampedBase, table=True):
    __tablename__ = "sessions"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    
    # Relationships
    experiments: List["Experiment"] = Relationship(back_populates="session")


# Experiment model
class ExperimentBase(SQLModel):
    session_id: str = Field(foreign_key="sessions.id", nullable=False)
    config_id: str = Field(max_length=100, nullable=False)
    config_data: Dict[str, Any] = Field(sa_column=Column(JSON))
    
    dataset_items: int = Field(default=0, ge=0)
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    execution_time_seconds: Optional[int] = Field(default=None, ge=0)
    
    # Results
    overall_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metric_scores: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Resource usage
    token_usage: Dict[str, int] = Field(default_factory=dict, sa_column=Column(JSON))
    api_calls: Optional[int] = Field(default=0, ge=0)
    cost_usd: Optional[float] = Field(default=0.0, ge=0.0)
    
    # Error tracking
    error_message: Optional[str] = Field(default=None)
    error_count: int = Field(default=0, ge=0)


class Experiment(ExperimentBase, TimestampedBase, table=True):
    __tablename__ = "experiments"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    
    # Relationships
    session: Session = Relationship(back_populates="experiments")
    item_results: List["ItemResult"] = Relationship(back_populates="experiment")


# Item result model
class ItemResultBase(SQLModel):
    experiment_id: str = Field(foreign_key="experiments.id", nullable=False)
    item_id: str = Field(max_length=100, nullable=False)
    
    # Results
    score: float = Field(ge=0.0, le=1.0, nullable=False)
    metric_scores: Dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Agent output
    agent_output: Optional[str] = Field(default=None)
    execution_time: Optional[float] = Field(default=None, ge=0.0)
    
    # Token usage
    token_usage: Dict[str, int] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Evaluation
    evaluation_method: Optional[str] = Field(default=None, max_length=50)
    evaluation_feedback: Optional[str] = Field(default=None)
    
    # Error tracking
    error_message: Optional[str] = Field(default=None)


class ItemResult(ItemResultBase, TimestampedBase, table=True):
    __tablename__ = "item_results"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    
    # Relationships
    experiment: Experiment = Relationship(back_populates="item_results")


# Configuration model (for storing generated configs)
class ConfigBase(SQLModel):
    config_id: str = Field(max_length=100, nullable=False, unique=True)
    session_id: Optional[str] = Field(default=None, max_length=100)
    parent_config_id: Optional[str] = Field(default=None, max_length=100)
    
    # Config content
    system_prompts: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    tool_descriptions: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    llm_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    agent_parameters: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Metadata
    generation_method: Optional[str] = Field(default=None, max_length=50)
    mutations_applied: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    performance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Usage tracking
    experiment_count: int = Field(default=0, ge=0)
    best_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Config(ConfigBase, TimestampedBase, table=True):
    __tablename__ = "configs"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)


# Dataset metadata model
class DatasetMetadataBase(SQLModel):
    name: str = Field(max_length=100, nullable=False, unique=True)
    version: str = Field(max_length=20, nullable=False)
    description: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None, max_length=100)
    
    # Content info
    item_count: int = Field(default=0, ge=0)
    schema_version: str = Field(default="1.0", max_length=20)
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    # Evaluation info
    evaluation_metrics: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    evaluation_methods: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Data splits
    train_split: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    validation_split: Optional[float] = Field(default=0.2, ge=0.0, le=1.0)
    test_split: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)
    
    # Storage info
    file_path: str = Field(nullable=False)
    file_size: Optional[int] = Field(default=None, ge=0)
    checksum: Optional[str] = Field(default=None, max_length=64)


class DatasetMetadata(DatasetMetadataBase, TimestampedBase, table=True):
    __tablename__ = "dataset_metadata"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)