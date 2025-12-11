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
| $x_{19}$ | `elec_any_violate` |
| $x_{20}$ | `elec_violate_v` |
| $x_{21}$ | `elec_violate_line` |
| $x_{22}$ | `grid_load_demand` |
| $x_{23}$ | `hour_of_day` |
| $x_{24}$ | `day_of_week` |

## 2. 挖掘结果

### 目标: `yhat_ess1_up`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_ess1_down`
- **R2 Score**: 0.9999
$$
\sqrt{x_{20}}
$$
---

### 目标: `yhat_ess2_up`
- **R2 Score**: 0.9753
$$
x_{20}^{2}
$$
---

### 目标: `yhat_ess2_down`
- **R2 Score**: 0.9853
$$
x_{19} + \frac{x_{19}}{0.9260 \left(x_{14} - \left(- 1.393 x_{14} + x_{8} + 0.4807\right)^{2}\right)}
$$
---

### 目标: `yhat_ess3_up`
- **R2 Score**: 0.9495
$$
x_{20} + \left|{\frac{\left(4.725 x_{12} - 0.3419 x_{19} - 5.771 + \frac{5.000}{x_{6}}\right)^{2}}{\left|{\sqrt{x_{12}}}\right|}}\right|
$$
---

### 目标: `yhat_ess3_down`
- **R2 Score**: 0.9778
$$
\left|{x_{19} \left(1.060 - \frac{0.4638}{0.4638 x_{10} - \frac{x_{20}}{\frac{1}{3.743} x_{12}}}\right)}\right|
$$
---

### 目标: `yhat_ess4_up`
- **R2 Score**: 0.9238
$$
x_{20} + \frac{x_{8}}{\left(\frac{x_{17} x_{6}}{0.2196} - \left(x_{17} - \frac{x_{20}}{0.04759 \frac{1}{\left|{x_{16}}\right|}} + x_{7} + x_{8}\right) - 1.202\right)^{2}}
$$
---

### 目标: `yhat_ess4_down`
- **R2 Score**: 0.9571
$$
x_{12} x_{20} \sqrt{x_{23} + 3.135}
$$
---

### 目标: `yhat_ess5_up`
- **R2 Score**: 0.9854
$$
x_{19} + \left|{\frac{\sqrt{x_{10}}}{0.5227 x_{15} \left(- x_{10} + x_{15} + 2.638\right) - x_{22} \left(x_{19} + 0.2276\right)}}\right|
$$
---

### 目标: `yhat_ess5_down`
- **R2 Score**: 0.9847
$$
x_{20} - 0.4004 \frac{0.1603 x_{20}^{2}}{\left(- x_{17} + x_{7} \cdot 0.6583\right)^{2}}
$$
---

### 目标: `yhat_sop1_fwd`
- **R2 Score**: 0.8729
$$
\sqrt{\left|{x_{13} - 0.1884}\right|} \left|{x_{13} - 2.114 x_{20}}\right|
$$
---

### 目标: `yhat_sop1_rev`
- **R2 Score**: 0.7160
$$
- x_{13} + 3.177 - \frac{163.1}{x_{7}}
$$
---

### 目标: `yhat_sop2_fwd`
- **R2 Score**: 0.9014
$$
\frac{x_{19} x_{23}}{\left|{\frac{x_{13}}{1.318 - \frac{1.284}{x_{6}}}}\right|}
$$
---

### 目标: `yhat_sop2_rev`
- **R2 Score**: 0.3659
$$
x_{8} \cdot 0.01085
$$
---

### 目标: `yhat_pv1`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_pv2`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_pv3`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_pv4`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_pv5`
- **R2 Score**: 1.0000
$$
x_{20}
$$
---

### 目标: `yhat_pv6`
- **R2 Score**: 0.9645
$$
\sqrt{x_{20}} \sqrt{\left|{20.07 - \frac{21.03}{x_{11}}}\right|}
$$
---

### 目标: `yhat_wt1`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_wt2`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---

### 目标: `yhat_wt3`
- **R2 Score**: 1.0000
$$
x_{20}
$$
---

### 目标: `yhat_wt4`
- **R2 Score**: 1.0000
$$
x_{20}
$$
---

### 目标: `yhat_wt5`
- **R2 Score**: 1.0000
$$
\sqrt{x_{20}}
$$
---

### 目标: `yhat_wt6`
- **R2 Score**: 1.0000
$$
x_{19}
$$
---
