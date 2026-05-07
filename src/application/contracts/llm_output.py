from pydantic import BaseModel, Field
from src.domain.models.classification import PrimaryClass, Subcategory

class LLMOutput(BaseModel):
    """Structured output from a single LLM call."""
    primary_class: PrimaryClass
    subcategory: Subcategory
    hospital_name: str | None = Field(
        default=None,
        description="Hospital or clinic name if this is a medical document, null otherwise",
    )
