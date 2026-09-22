"""Self-run fallback on Modal GPUs: TRL + PEFT (SFT, DPO, GenRM) and verl (GRPO/DAPO/GSPO,
Rank-GRPO, OPD). Used when a model or loss is not available on Tinker. M3."""

from c2a import NotYetImplemented


def submit(cfg) -> str:
    raise NotYetImplemented("Modal fallback trainer", "M3")
