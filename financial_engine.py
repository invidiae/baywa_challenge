"""Business-interruption claim calculation for renewable energy outages."""

from __future__ import annotations

import math


class BusinessInterruptionEngine:
    """Apply revenue, deductible, gross-profit, and ICOW policy terms."""

    def __init__(self, ppa_rate_per_kwh: float = 0.08, ptc_rate_per_kwh: float = 0.025,
                 gross_profit_rate: float = 0.85, deductible_hours: int = 24) -> None:
        if min(ppa_rate_per_kwh, ptc_rate_per_kwh, deductible_hours) < 0 or not 0 <= gross_profit_rate <= 1:
            raise ValueError("Invalid non-negative rates, deductible, or gross-profit rate.")
        self.ppa_rate_per_kwh = float(ppa_rate_per_kwh)
        self.ptc_rate_per_kwh = float(ptc_rate_per_kwh)
        self.gross_profit_rate = float(gross_profit_rate)
        self.deductible_hours = int(deductible_hours)

    def calculate_net_claim(self, gross_loss_euro: float, outage_duration_hours: float,
                            saved_variable_costs: float = 0.0, icow_costs: float = 0.0) -> dict[str, float]:
        """Return a policy-adjusted claim; time excess is pro-rated over the event."""
        values = (gross_loss_euro, outage_duration_hours, saved_variable_costs, icow_costs)
        if any(not math.isfinite(float(value)) or value < 0 for value in values):
            raise ValueError("Claim inputs must be finite and non-negative.")
        deductible = gross_loss_euro * (min(self.deductible_hours, outage_duration_hours) / outage_duration_hours
                                         if outage_duration_hours else 0.0)
        post_deductible = gross_loss_euro - deductible
        saved_loss = post_deductible * self.gross_profit_rate
        approved_icow = icow_costs if icow_costs <= saved_loss else 0.0
        return {
            "gross_revenue_loss_euro": round(gross_loss_euro, 2),
            "deductible_deduction_euro": round(deductible, 2),
            "net_claim_payable_euro": round(max(0.0, saved_loss - saved_variable_costs) + approved_icow, 2),
            "saved_costs_deducted_euro": round(saved_variable_costs, 2),
            "icow_approved_euro": round(approved_icow, 2),
        }
