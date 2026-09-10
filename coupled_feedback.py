"""
Coupled Atmospheric Physics & Chemical Transport Feedback Engine for Delhi NCR.
Simulates two-way bidirectional feedback loops between aerosols (PM2.5) and meteorology:
1. Radiative Solar Dimming (Beer-Lambert optical attenuation).
2. Surface Temperature Depression and stronger nocturnal inversion.
3. Planetary Boundary Layer (PBL) Height Collapse (thermal convection suppression).
4. Near-Surface Volume Trapping & Ground-Level Concentration Compression.
5. Aerosol-Induced Surface Stagnation & Wind Deceleration.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("airloop.coupled_feedback")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


@dataclass
class CoupledFeedbackStep:
    """Diagnostic state of the coupled atmospheric-chemical column at hour t."""
    hour: int
    datetime_str: str
    unperturbed_solar_flux: float  # W/m²
    coupled_solar_flux: float      # W/m²
    solar_dimming_pct: float       # % reduction in solar radiation reaching surface
    unperturbed_temp_c: float      # °C
    coupled_temp_c: float          # °C
    temp_depression_c: float       # Surface cooling ΔT
    unperturbed_pbl_m: float       # Baseline PBL mixing height (m)
    coupled_pbl_m: float           # Aerosol-suppressed PBL mixing height (m)
    pbl_collapse_pct: float        # % collapse in PBL depth
    entrapment_factor: float       # Concentration amplification factor (H_0 / H_pbl)^α
    unperturbed_wind_speed: float  # Synoptic wind speed (m/s)
    coupled_wind_speed: float      # Aerosol-decelerated wind speed (m/s)
    wind_deceleration_pct: float   # % induced stillness
    decoupled_smoke_index: float   # Traditional uncoupled ground concentration
    coupled_smoke_index: float     # Coupled feedback ground concentration


@dataclass
class CoupledSimulationResult:
    """Full 72-hour coupled simulation results across Delhi NCR."""
    forecast_hours: int
    times: List[str]
    steps: List[CoupledFeedbackStep]
    decoupled_delhi_mean: List[float]
    coupled_delhi_mean: List[float]
    decoupled_delhi_peak: List[float]
    coupled_delhi_peak: List[float]
    pbl_height_series: List[float]
    solar_dimming_series: List[float]
    entrapment_multiplier_series: List[float]
    peak_smog_hour: int
    peak_smog_time: str
    underprediction_max_gap: float  # Maximum AQI/smoke gap where decoupled model missed


class CoupledAtmosphericChemistryEngine:
    """
    Two-way coupled numerical solver modeling aerosol-radiative-boundary layer feedbacks.
    """

    def __init__(
        self,
        optical_extinction_kappa: float = 0.0018,  # Extinction coefficient per unit smoke
        temp_depression_coeff: float = 2.8,       # Max surface cooling factor (°C)
        pbl_suppression_gamma: float = 0.0022,    # PBL thermal collapse sensitivity
        min_pbl_ratio: float = 0.35,              # Minimum boundary layer collapse floor (35%)
        compression_exponent_alpha: float = 0.85, # Boundary layer volumetric compression scaling
        wind_stagnation_eta: float = 0.28,        # Max induced wind deceleration (28%)
        stagnation_half_sat: float = 120.0,       # Half-saturation smoke concentration
    ):
        self.kappa = optical_extinction_kappa
        self.beta = temp_depression_coeff
        self.gamma = pbl_suppression_gamma
        self.min_pbl_ratio = min_pbl_ratio
        self.alpha = compression_exponent_alpha
        self.eta = wind_stagnation_eta
        self.c0 = stagnation_half_sat

    def compute_baseline_pbl_height(self, hour_of_day: int, month: int = 11) -> float:
        """
        Estimate typical unperturbed diurnal Planetary Boundary Layer (PBL) height (m)
        over Delhi during winter (November).
        - Nighttime nocturnal stable boundary layer: ~250m - 350m
        - Midday convective mixing boundary layer: ~1,100m - 1,400m
        """
        # Diurnal solar cycle peak around 14:00 (2 PM local time)
        # Solar elevation proxy
        solar_phase = (hour_of_day - 14.0) / 12.0 * math.pi
        diurnal_factor = max(0.0, math.cos(solar_phase))
        
        # In November, baseline night PBL ~300m, daytime clean peak ~1250m
        night_pbl = 300.0
        peak_convective_lift = 950.0
        return night_pbl + (diurnal_factor ** 1.5) * peak_convective_lift

    def compute_clear_sky_solar_flux(self, hour_of_day: int) -> float:
        """Estimate clear-sky surface solar irradiance (W/m²) in Delhi winter."""
        # Daylight between ~06:30 and 17:30
        if 6 <= hour_of_day <= 18:
            angle = (hour_of_day - 6.0) / 12.0 * math.pi
            return max(0.0, 780.0 * math.sin(angle))
        return 0.0

    def solve_coupled_step(
        self,
        hour_idx: int,
        datetime_str: str,
        unperturbed_smoke: float,
        ambient_temp_c: float,
        ambient_wind_speed: float,
    ) -> CoupledFeedbackStep:
        """
        Solve the coupled feedback equations at a single forecast time-step.
        """
        # Parse hour of day (local or UTC approx)
        hour_of_day = (hour_idx + 12) % 24  # Start around midday
        
        # 1. Baseline unperturbed meteorology
        i0_solar = self.compute_clear_sky_solar_flux(hour_of_day)
        h0_pbl = self.compute_baseline_pbl_height(hour_of_day)
        u0_wind = max(0.4, ambient_wind_speed)
        t0_temp = ambient_temp_c
        
        # 2. Aerosol Radiative Forcing (Beer-Lambert optical depth)
        # Smoke optical depth proxy tau
        tau = self.kappa * max(0.0, unperturbed_smoke)
        solar_transmittance = math.exp(-tau)
        i_coupled_solar = i0_solar * solar_transmittance
        dimming_pct = (1.0 - solar_transmittance) * 100.0 if i0_solar > 10.0 else 0.0
        
        # 3. Surface Temperature Depression (Radiative cooling)
        # Strongest during daylight hours when aerosols scatter incoming shortwave radiation
        daylight_weight = min(1.0, i0_solar / 300.0) if i0_solar > 0 else 0.15
        delta_temp = self.beta * (1.0 - solar_transmittance) * daylight_weight
        t_coupled = t0_temp - delta_temp
        
        # 4. Planetary Boundary Layer (PBL) Height Collapse
        # Decreased surface sensible heat flux collapses convective turbulent kinetic energy (TKE)
        pbl_suppression_factor = max(
            self.min_pbl_ratio,
            math.exp(-self.gamma * max(0.0, unperturbed_smoke))
        )
        # Additional collapse at night due to stronger surface radiation cooling
        if i0_solar < 10.0 and unperturbed_smoke > 20.0:
            pbl_suppression_factor *= 0.88
            pbl_suppression_factor = max(self.min_pbl_ratio, pbl_suppression_factor)
            
        h_coupled_pbl = h0_pbl * pbl_suppression_factor
        pbl_collapse_pct = (1.0 - pbl_suppression_factor) * 100.0
        
        # 5. Entrapment / Volumetric Compression Multiplier
        # As PBL collapses from H0 to H_coupled, pollutants are compressed into a shallower volume:
        # C_ground = C_advected * (H_0 / H_pbl)^alpha
        entrapment_factor = (h0_pbl / max(150.0, h_coupled_pbl)) ** self.alpha
        
        # 6. Induced Stagnation / Surface Wind Deceleration Feedback
        # Suppressed vertical momentum exchange dampens surface winds (induced stillness)
        wind_deceleration = self.eta * (unperturbed_smoke / (unperturbed_smoke + self.c0))
        u_coupled_wind = u0_wind * (1.0 - wind_deceleration)
        wind_decel_pct = wind_deceleration * 100.0
        
        # 7. Coupled Ground Concentration (Final positive feedback concentration)
        # Slower wind further reduces horizontal ventilation: ventil_factor = (u0 / u_coupled)^0.3
        ventil_ratio = (u0_wind / max(0.3, u_coupled_wind)) ** 0.35
        coupled_smoke = unperturbed_smoke * entrapment_factor * ventil_ratio
        
        return CoupledFeedbackStep(
            hour=hour_idx,
            datetime_str=datetime_str,
            unperturbed_solar_flux=round(i0_solar, 1),
            coupled_solar_flux=round(i_coupled_solar, 1),
            solar_dimming_pct=round(dimming_pct, 1),
            unperturbed_temp_c=round(t0_temp, 1),
            coupled_temp_c=round(t_coupled, 1),
            temp_depression_c=round(delta_temp, 2),
            unperturbed_pbl_m=round(h0_pbl, 0),
            coupled_pbl_m=round(h_coupled_pbl, 0),
            pbl_collapse_pct=round(pbl_collapse_pct, 1),
            entrapment_factor=round(entrapment_factor, 2),
            unperturbed_wind_speed=round(u0_wind, 2),
            coupled_wind_speed=round(u_coupled_wind, 2),
            wind_deceleration_pct=round(wind_decel_pct, 1),
            decoupled_smoke_index=round(unperturbed_smoke, 1),
            coupled_smoke_index=round(coupled_smoke, 1),
        )

    def simulate_72h_coupled_trajectory(
        self,
        decoupled_delhi_mean: List[float],
        decoupled_delhi_max: List[float],
        ambient_temps: List[float],
        ambient_winds: List[float],
        timestamps: List[str],
    ) -> CoupledSimulationResult:
        """
        Execute full 72-hour coupled atmospheric-chemical simulation across Delhi NCR.
        """
        n_hours = min(72, len(decoupled_delhi_mean))
        steps: List[CoupledFeedbackStep] = []
        coupled_means: List[float] = []
        coupled_peaks: List[float] = []
        pbl_heights: List[float] = []
        dimmings: List[float] = []
        entrapments: List[float] = []
        
        logger.info(f"Simulating 72-hour coupled aerosol-meteorology feedback loops ({n_hours} hours)...")
        
        for h in range(n_hours):
            t_str = timestamps[h] if h < len(timestamps) else f"+{h}h"
            u_smoke = decoupled_delhi_mean[h]
            t_ambient = ambient_temps[h] if h < len(ambient_temps) else 22.0
            w_ambient = ambient_winds[h] if h < len(ambient_winds) else 2.2
            
            step = self.solve_coupled_step(
                hour_idx=h,
                datetime_str=t_str,
                unperturbed_smoke=u_smoke,
                ambient_temp_c=t_ambient,
                ambient_wind_speed=w_ambient,
            )
            steps.append(step)
            coupled_means.append(step.coupled_smoke_index)
            
            # Scale peak with step entrapment factor
            peak_val = decoupled_delhi_max[h] * step.entrapment_factor * (1.0 + step.wind_deceleration_pct / 100.0 * 0.4)
            coupled_peaks.append(round(peak_val, 1))
            
            pbl_heights.append(step.coupled_pbl_m)
            dimmings.append(step.solar_dimming_pct)
            entrapments.append(step.entrapment_factor)
            
        peak_hour = int(np.argmax(coupled_means))
        peak_time = timestamps[peak_hour] if peak_hour < len(timestamps) else f"+{peak_hour}h"
        max_gap = float(np.max(np.array(coupled_means) - np.array(decoupled_delhi_mean[:n_hours])))
        
        logger.info(
            f"Coupled simulation completed! Peak Delhi Smog at Hour +{peak_hour} ({peak_time}). "
            f"Max feedback entrapment gap: +{max_gap:.1f} index points."
        )
        
        return CoupledSimulationResult(
            forecast_hours=n_hours,
            times=timestamps[:n_hours],
            steps=steps,
            decoupled_delhi_mean=[round(v, 1) for v in decoupled_delhi_mean[:n_hours]],
            coupled_delhi_mean=coupled_means,
            decoupled_delhi_peak=[round(v, 1) for v in decoupled_delhi_max[:n_hours]],
            coupled_delhi_peak=coupled_peaks,
            pbl_height_series=pbl_heights,
            solar_dimming_series=dimmings,
            entrapment_multiplier_series=entrapments,
            peak_smog_hour=peak_hour,
            peak_smog_time=peak_time,
            underprediction_max_gap=round(max_gap, 1),
        )
