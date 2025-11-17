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
        gamma_1 = np.cos(np.cos(np.log(x14) / np.sin(4.6223493 / np.sin(x9))) * (np.sin(x1) / (x2 + (((-7.1510797 - x16) * (1.3045293 * np.cos(x0 - 4.6223493))) + x7)))) * x19  # Formula for yhat_ess1_down
        gamma_2 = x19 + (np.exp(np.exp(((np.sin((1.0003977 - x9) / -2.4805896) - np.sin(x17 / (x13 - 0.67480236))) / x6) - x19)) * 0.00039764764)  # Formula for yhat_ess2_up
        gamma_3 = np.sin((np.exp(np.sin(np.sin(np.sin(0.59462315 / (np.sin(0.60017574 / (x13 * 1.0549589)) * 2.145705)) / 0.4505158))) * x20) * 0.7304368)  # Formula for yhat_ess2_down
        gamma_4 = (0.38482314 / (np.exp(np.sin(0.79689807 * x10)) * np.exp(np.exp((x20 + np.sin(x10 * 0.5233885)) - -1.1199477) + np.sin(x9)))) + x20  # Formula for yhat_ess3_up
        gamma_5 = (x20 * 1.0300412) * np.cos(np.cos((x13 / (-0.12644365 / np.log(x9))) * 6.102543) / ((x15 * x12) * np.cos((x13 / 0.91409487) / -0.12287817)))  # Formula for yhat_ess3_down
        gamma_6 = (x4 * x19) - (-0.032532744 * np.exp(np.sin(-0.29794207 - (x10 / 0.98438436)) / (-1.17327 - np.sin(np.sin(x10 * -0.7279275)))))  # Formula for yhat_ess4_up
        gamma_7 = np.sin(x19 / np.cos(np.cos(np.cos((-0.4136501 - x11) / x13)) / (x4 * ((x12 / -0.342813) + -0.2866511))))  # Formula for yhat_ess4_down
        gamma_8 = (np.sin(x9) / (((np.cos(x9 / -2.162963) - 1.0624925) * x16) / (1.0624925 - x19))) + x19  # Formula for yhat_ess5_up
        gamma_9 = x19 * np.sin(np.exp(np.exp(np.cos(x9 * (-12.192526 / (((x16 - x9) - np.cos((x3 * x12) - x9)) * 10.445083))))))  # Formula for yhat_ess5_down
        gamma_10 = np.sin((((x11 + 1.7653188) + ((x19 / x13) / 1.8240205)) + x13) / -0.9306248)  # Formula for yhat_sop1_fwd
        gamma_11 = np.cos(0.83093876 - np.cos(np.exp(x6 * 3.354825)))  # Formula for yhat_sop1_rev
        gamma_12 = np.sin(x19 + np.sin((np.cos(2.316559 / x13) * x19) / -0.7918499))  # Formula for yhat_sop2_fwd
        gamma_13 = np.cos(np.cos(x14 / x6))  # Formula for yhat_sop2_rev
        gamma_14 = x20  # Formula for yhat_pv1
        gamma_15 = x19 * x20  # Formula for yhat_pv2
        gamma_16 = x20 * x19  # Formula for yhat_pv3
        gamma_17 = x20  # Formula for yhat_pv4
        gamma_18 = x19 * 1.0  # Formula for yhat_pv5
        gamma_19 = np.cos(x21 - x13) * x20  # Formula for yhat_pv6
        gamma_20 = x20 * x20  # Formula for yhat_wt1
        gamma_21 = x19  # Formula for yhat_wt2
        gamma_22 = x19  # Formula for yhat_wt3
        gamma_23 = x20  # Formula for yhat_wt4
        gamma_24 = x19 * x19  # Formula for yhat_wt5
        gamma_25 = x19  # Formula for yhat_wt6

        # Combine components into a vector
        gamma_values = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6, gamma_7, gamma_8, gamma_9, gamma_10, gamma_11, gamma_12, gamma_13, gamma_14, gamma_15, gamma_16, gamma_17, gamma_18, gamma_19, gamma_20, gamma_21, gamma_22, gamma_23, gamma_24, gamma_25]

        # Convert to a PyTorch tensor with a batch dimension
        return torch.tensor(gamma_values, dtype=torch.float32).unsqueeze(0)