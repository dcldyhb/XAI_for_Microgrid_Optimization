# PySR 信号公式汇总

## 1. 输入特征映射

| 符号 ($x_i$) | 物理含义 |
| :--- | :--- |
| $x_{0}$ | `temperature` |
| $x_{1}$ | `wind_speed` |
| $x_{2}$ | `solar_irradiance` |
| $x_{3}$ | `precipitation_mm` |
| $x_{4}$ | `elec_vmin` |
| $x_{5}$ | `elec_vmax` |
| $x_{6}$ | `elec_vavg` |
| $x_{7}$ | `elec_line_loading_max` |
| $x_{8}$ | `elec_line_loading_top1` |
| $x_{9}$ | `elec_line_loading_top2` |
| $x_{10}$ | `elec_line_loading_top3` |
| $x_{11}$ | `elec_vmax_nonslack` |
| $x_{12}$ | `elec_line_i_ka_max` |
| $x_{13}$ | `elec_p_loss_mw` |
| $x_{14}$ | `elec_soc_1` |
| $x_{15}$ | `elec_soc_2` |
| $x_{16}$ | `elec_soc_3` |
| $x_{17}$ | `elec_soc_4` |
| $x_{18}$ | `elec_soc_5` |
| $x_{19}$ | `grid_load_demand` |
| $x_{20}$ | `hour_of_day` |
| $x_{21}$ | `day_of_week` |

## 2. 挖掘结果

### 目标: `sens_action_batt_0`
- **R2 Score**: 0.2368
$$
\frac{5.589 \cdot 10^{-8}}{\left(- x_{13} + x_{20} + x_{3} + x_{3} - 0.4162 - 0.4000\right)^{2}}
$$
---

### 目标: `sens_action_batt_1`
- **R2 Score**: 0.0439
$$
\frac{x_{0}^{3}}{x_{19}^{3}} \cdot 83.02
$$
---

### 目标: `sens_action_batt_2`
- **R2 Score**: 0.2584
$$
\frac{x_{20}}{x_{1}^{3} \left(x_{13} + x_{3}\right)^{3} + x_{13} + x_{20}^{2}}
$$
---

### 目标: `sens_action_batt_3`
- **R2 Score**: 0.1139
$$
\frac{1.786}{x_{1} + x_{20} - -1.921}
$$
---

### 目标: `sens_action_batt_4`
- **R2 Score**: -0.0010
$$
\frac{2.290}{x_{7}^{3} x_{13}^{2} \left(- x_{15} + \frac{x_{7}}{1.567}\right)^{2}}
$$
---
