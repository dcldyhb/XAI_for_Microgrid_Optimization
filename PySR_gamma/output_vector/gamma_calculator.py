# This file is auto-generated. Do not edit it manually.
import numpy as np
import torch


class GeneratedGammaCalculator:
    def __init__(self):
        self.feature_names = ['temperature_C', 'wind_speed_ms', 'illuminance_lux', 'precipitation_mm', 'elec_vmin', 'elec_vmax', 'elec_vavg', 'elec_line_loading_max', 'elec_line_loading_top1', 'elec_line_loading_top2', 'elec_line_loading_top3', 'elec_vmax_nonslack', 'elec_line_i_ka_max', 'elec_p_loss_mw', 'elec_soc_1', 'elec_soc_2', 'elec_soc_3', 'elec_soc_4', 'elec_soc_5', 'elec_any_violate', 'elec_violate_v', 'elec_violate_line']
        self.gamma_signals = ['yhat_ess1_up', 'yhat_ess1_down', 'yhat_ess2_up', 'yhat_ess2_down', 'yhat_ess3_up', 'yhat_ess3_down', 'yhat_ess4_up', 'yhat_ess4_down', 'yhat_ess5_up', 'yhat_ess5_down', 'yhat_sop1_fwd', 'yhat_sop1_rev', 'yhat_sop2_fwd', 'yhat_sop2_rev', 'yhat_pv1', 'yhat_pv2', 'yhat_pv3', 'yhat_pv4', 'yhat_pv5', 'yhat_pv6', 'yhat_wt1', 'yhat_wt2', 'yhat_wt3', 'yhat_wt4', 'yhat_wt5', 'yhat_wt6']

    def compute(self, state):
        # Unpack state variables to match PySR's x_i notation
        x0 = state['temperature_C']
        x1 = state['wind_speed_ms']
        x2 = state['illuminance_lux']
        x3 = state['precipitation_mm']
        x4 = state['elec_vmin']
        x5 = state['elec_vmax']
        x6 = state['elec_vavg']
        x7 = state['elec_line_loading_max']
        x8 = state['elec_line_loading_top1']
        x9 = state['elec_line_loading_top2']
        x10 = state['elec_line_loading_top3']
        x11 = state['elec_vmax_nonslack']
        x12 = state['elec_line_i_ka_max']
        x13 = state['elec_p_loss_mw']
        x14 = state['elec_soc_1']
        x15 = state['elec_soc_2']
        x16 = state['elec_soc_3']
        x17 = state['elec_soc_4']
        x18 = state['elec_soc_5']
        x19 = state['elec_any_violate']
        x20 = state['elec_violate_v']
        x21 = state['elec_violate_line']

        # Calculate each gamma component
        gamma_0 = x20  # Formula for yhat_ess1_up
        gamma_1 = np.cos((x5 + -0.7314017) / (x10 + (np.sin((x5 + -0.73140293) / ((x10 - (-0.8376213 / np.cos(x0))) + -0.5171355)) - (-0.8377111 / np.cos(x0))))) * x20  # Formula for yhat_ess1_down
        gamma_2 = np.sin((x20 - (np.exp((np.cos(x9 * -0.9293466) + np.cos(x10 / -1.8063388)) + np.cos(x0 + 1.8242577)) * (-0.4715142 / x0))) / 0.68531096)  # Formula for yhat_ess2_up
        gamma_3 = x19 * np.sin(3.6668582 * (-0.35382104 + np.sin(x6 / np.sin((x19 / x12) + 0.42984277))))  # Formula for yhat_ess2_down
        gamma_4 = np.cos(np.exp(-0.28565803 - (x19 - np.cos(np.cos(np.exp(np.exp(np.sin(np.exp(np.exp(np.cos(np.cos(np.exp(x4)) * x15))))))) / np.exp(np.cos(x15 * np.cos(np.exp(x4))))))) - 0.44017023)  # Formula for yhat_ess3_up
        gamma_5 = x19 * np.cos(np.cos(-1.5064305 / np.sin(-1.1989906 / np.sin((-0.26150653 - x16) / (x4 / 0.66241705)))) * np.log(x12))  # Formula for yhat_ess3_down
        gamma_6 = (0.04455137 / (np.sin((x17 / x4) + 1.1965765) + (x20 + 0.43819284))) + (x20 - 0.071441166)  # Formula for yhat_ess4_up
        gamma_7 = np.sin(np.sin(x20 * (np.exp(np.sin(-0.79819804 / x12)) - np.cos(2.744782 / x13))) / 0.61423206)  # Formula for yhat_ess4_down
        gamma_8 = np.cos((x4 - x20) * 1.6209244)  # Formula for yhat_ess5_up
        gamma_9 = (3.567692 / ((x8 + (35.434715 * (-1.1516261 - x20))) * (2.2788258 - x14))) + x20  # Formula for yhat_ess5_down
        gamma_10 = (x13 + ((x20 * 0.7780336) - 0.13307951)) * np.sin(0.59257233 / x13)  # Formula for yhat_sop1_fwd
        gamma_11 = np.cos(np.cos(np.exp(np.exp(x4 + 0.19219609))) + 0.6686028)  # Formula for yhat_sop1_rev
        gamma_12 = x19 * np.exp(np.exp(x16 - x10) / -6.3184824)  # Formula for yhat_sop2_fwd
        gamma_13 = np.sin(x19 - (-6.786821 / x4))  # Formula for yhat_sop2_rev
        gamma_14 = x20 * x20  # Formula for yhat_pv1
        gamma_15 = x19  # Formula for yhat_pv2
        gamma_16 = x19  # Formula for yhat_pv3
        gamma_17 = x20  # Formula for yhat_pv4
        gamma_18 = x20  # Formula for yhat_pv5
        gamma_19 = np.cos(-0.009958512 / np.log(x6)) * x20  # Formula for yhat_pv6
        gamma_20 = x19  # Formula for yhat_wt1
        gamma_21 = x20  # Formula for yhat_wt2
        gamma_22 = x20  # Formula for yhat_wt3
        gamma_23 = x20  # Formula for yhat_wt4
        gamma_24 = x19  # Formula for yhat_wt5
        gamma_25 = x20  # Formula for yhat_wt6

        # Combine components into a vector
        gamma_values = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6, gamma_7, gamma_8, gamma_9, gamma_10, gamma_11, gamma_12, gamma_13, gamma_14, gamma_15, gamma_16, gamma_17, gamma_18, gamma_19, gamma_20, gamma_21, gamma_22, gamma_23, gamma_24, gamma_25]

        # Convert to a PyTorch tensor with a batch dimension
        return torch.tensor(gamma_values, dtype=torch.float32).unsqueeze(0)