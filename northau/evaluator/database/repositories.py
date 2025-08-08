"""Repository pattern for database access."""

from sqlmodel import Session as SQLSession, select
from typing import Optional, Dict, Any, List
from datetime import datetime

from .models import (
    Session, SessionBase,
    Experiment, ExperimentBase, ExperimentStatus,
    ItemResult, ItemResultBase,
    Config, ConfigBase,
    DatasetMetadata, DatasetMetadataBase
)


class SessionRepository:
    """Repository for managing experiment sessions."""
    
    def __init__(self, session: SQLSession):
        self.session = session
    
    def create_session(self, session_data: SessionBase) -> Session:
        """Create a new experiment session."""
        session_obj = Session(**session_data.dict())
        self.session.add(session_obj)
        self.session.commit()
        self.session.refresh(session_obj)
        return session_obj
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.session.get(Session, session_id)
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[Session]:
        """Update session with given fields."""
        session_obj = self.session.get(Session, session_id)
        if session_obj:
            for key, value in updates.items():
                if hasattr(session_obj, key):
                    setattr(session_obj, key, value)
            session_obj.updated_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(session_obj)
        return session_obj
    
    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[Session]:
        """List sessions with pagination."""
        statement = select(Session).order_by(Session.created_at.desc()).offset(offset).limit(limit)
        return list(self.session.exec(statement))
    
    def get_active_sessions(self) -> List[Session]:
        """Get all currently active sessions."""
        statement = select(Session).where(
            Session.status.in_([ExperimentStatus.PENDING, ExperimentStatus.RUNNING])
        )
        return list(self.session.exec(statement))


class ExperimentRepository:
    """Repository for managing experiments."""
    
    def __init__(self, session: SQLSession):
        self.session = session
    
    def create_experiment(self, experiment_data: ExperimentBase) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(**experiment_data.dict())
        self.session.add(experiment)
        self.session.commit()
        self.session.refresh(experiment)
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.session.get(Experiment, experiment_id)
    
    def get_session_experiments(self, session_id: str) -> List[Experiment]:
        """Get all experiments for a session."""
        statement = select(Experiment).where(Experiment.session_id == session_id)
        return list(self.session.exec(statement))
    
    def update_experiment_status(
        self, 
        experiment_id: str, 
        status: ExperimentStatus, 
        **kwargs
    ) -> Optional[Experiment]:
        """Update experiment status and other fields."""
        experiment = self.session.get(Experiment, experiment_id)
        if experiment:
            experiment.status = status
            experiment.updated_at = datetime.utcnow()
            for key, value in kwargs.items():
                if hasattr(experiment, key):
                    setattr(experiment, key, value)
            self.session.commit()
            self.session.refresh(experiment)
        return experiment
    
    def get_best_experiments(self, session_id: str, limit: int = 10) -> List[Experiment]:
        """Get best performing experiments for a session."""
        statement = (
            select(Experiment)
            .where(Experiment.session_id == session_id)
            .where(Experiment.overall_score.is_not(None))
            .order_by(Experiment.overall_score.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement))


class ItemResultRepository:
    """Repository for managing item results."""
    
    def __init__(self, session: SQLSession):
        self.session = session
    
    def create_item_result(self, item_result_data: ItemResultBase) -> ItemResult:
        """Create a new item result."""
        item_result = ItemResult(**item_result_data.dict())
        self.session.add(item_result)
        self.session.commit()
        self.session.refresh(item_result)
        return item_result
    
    def batch_create_item_results(self, item_results_data: List[ItemResultBase]) -> List[ItemResult]:
        """Create multiple item results in batch."""
        item_results = [ItemResult(**data.dict()) for data in item_results_data]
        self.session.add_all(item_results)
        self.session.commit()
        for item_result in item_results:
            self.session.refresh(item_result)
        return item_results
    
    def get_experiment_results(self, experiment_id: str) -> List[ItemResult]:
        """Get all item results for an experiment."""
        statement = select(ItemResult).where(ItemResult.experiment_id == experiment_id)
        return list(self.session.exec(statement))
    
    def get_item_result(self, result_id: str) -> Optional[ItemResult]:
        """Get item result by ID."""
        return self.session.get(ItemResult, result_id)


class ConfigRepository:
    """Repository for managing configurations."""
    
    def __init__(self, session: SQLSession):
        self.session = session
    
    def create_config(self, config_data: ConfigBase) -> Config:
        """Create a new configuration."""
        config = Config(**config_data.dict())
        self.session.add(config)
        self.session.commit()
        self.session.refresh(config)
        return config
    
    def get_config(self, config_id: str) -> Optional[Config]:
        """Get configuration by config_id."""
        statement = select(Config).where(Config.config_id == config_id)
        return self.session.exec(statement).first()
    
    def get_config_by_db_id(self, db_id: str) -> Optional[Config]:
        """Get configuration by database ID."""
        return self.session.get(Config, db_id)
    
    def get_session_configs(self, session_id: str) -> List[Config]:
        """Get all configurations generated in a session."""
        statement = select(Config).where(Config.session_id == session_id)
        return list(self.session.exec(statement))
    
    def update_config_performance(self, config_id: str, performance_score: float) -> Optional[Config]:
        """Update configuration performance score."""
        config = self.get_config(config_id)
        if config:
            if config.best_score is None or performance_score > config.best_score:
                config.best_score = performance_score
            config.experiment_count += 1
            config.updated_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(config)
        return config
    
    def get_best_configs(self, limit: int = 10) -> List[Config]:
        """Get best performing configurations."""
        statement = (
            select(Config)
            .where(Config.best_score.is_not(None))
            .order_by(Config.best_score.desc())
            .limit(limit)
        )
        return list(self.session.exec(statement))


class DatasetRepository:
    """Repository for managing dataset metadata."""
    
    def __init__(self, session: SQLSession):
        self.session = session
    
    def create_dataset_metadata(self, dataset_data: DatasetMetadataBase) -> DatasetMetadata:
        """Create new dataset metadata."""
        dataset = DatasetMetadata(**dataset_data.dict())
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset
    
    def get_dataset_metadata(self, name: str, version: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name and version."""
        statement = select(DatasetMetadata).where(
            DatasetMetadata.name == name,
            DatasetMetadata.version == version
        )
        return self.session.exec(statement).first()
    
    def list_datasets(self) -> List[DatasetMetadata]:
        """List all available datasets."""
        statement = select(DatasetMetadata).order_by(DatasetMetadata.name, DatasetMetadata.version)
        return list(self.session.exec(statement))
    
    def get_latest_dataset_version(self, name: str) -> Optional[DatasetMetadata]:
        """Get the latest version of a dataset."""
        statement = (
            select(DatasetMetadata)
            .where(DatasetMetadata.name == name)
            .order_by(DatasetMetadata.version.desc())
            .limit(1)
        )
        return self.session.exec(statement).first()