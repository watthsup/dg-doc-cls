from pydantic import BaseModel, Field

class LogprobAnalysis(BaseModel):
    top1_token: str = Field(description="Highest probability token")
    top1_logprob: float = Field(description="ln(p) of top token")
    top1_prob_pct: float = Field(ge=0.0, le=100.0, description="Probability % of top token")
    top2_token: str = Field(description="Runner-up token")
    top2_logprob: float = Field(description="ln(p) of runner-up token")
    top2_prob_pct: float = Field(ge=0.0, le=100.0, description="Probability % of runner-up token")
    margin_score: float = Field(
        description="Confidence delta: top1_logprob - top2_logprob",
    )
    confidence_pct: float = Field(
        ge=0.0, le=100.0,
        description="exp(top1_logprob) * 100 (Alias for top1_prob_pct)",
    )
