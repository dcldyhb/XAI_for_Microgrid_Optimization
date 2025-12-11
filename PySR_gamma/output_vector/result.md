# PySR Gamma 信号公式汇总

## 特征索引映射表

公式中的 $x_i$ 符号与以下实际特征相对应：

| 符号 ($x_i$) | 特征变量名 |
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


---

### Gamma 信号: `yhat_ess1_up`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_ess1_down`

**最佳公式:**

$$
x_{19} - \frac{x_{19} e^{\frac{0.8443}{x_{23} + \sin{\left(e^{x_{24}} \right)}}}}{x_{0} \left(x_{16} - \cos{\left(\frac{e^{x_{5}}}{x_{12}} \right)}\right) \left(x_{17} + x_{20} + x_{22} \cdot 1.426\right)}
$$

---

### Gamma 信号: `yhat_ess2_up`

**最佳公式:**

$$
x_{20} + \frac{\sin{\left(0.3800 x_{9} + 0.7997 \right)} + \cos{\left(x_{15} \right)} + 0.2875}{x_{14} x_{5} e^{x_{20} + \sin{\left(x_{0} \right)} + \sin{\left(\sin{\left(x_{9} \right)} \right)}}}
$$

---

### Gamma 信号: `yhat_ess2_down`

**最佳公式:**

$$
\sin{\left(x_{19} \left(1.954 + \frac{0.8289}{x_{23} - x_{6}}\right) \right)} 1.095
$$

---

### Gamma 信号: `yhat_ess3_up`

**最佳公式:**

$$
x_{19} + e^{e^{\frac{\left(0.8462 - \left(x_{19} - 0.8331\right)\right) e^{x_{12}}}{x_{6}}} \sin{\left(\frac{x_{10}}{\left(-1.723\right) \left(-1\right) 1.288} \right)}} 0.0001679
$$

---

### Gamma 信号: `yhat_ess3_down`

**最佳公式:**

$$
\sin{\left(\frac{\sin{\left(\sin{\left(x_{20} \left(-0.7548\right) \frac{1}{1.095 - \frac{x_{9}}{x_{14}}} \right)} \right)}}{0.4542} \right)}
$$

---

### Gamma 信号: `yhat_ess4_up`

**最佳公式:**

$$
x_{20} + \frac{\cos{\left(-0.6806 \right)}}{- x_{9} \left(- x_{19} + \cos{\left(x_{10} \left(-0.1381\right) \right)} - 1.033\right) - 0.5728}
$$

---

### Gamma 信号: `yhat_ess4_down`

**最佳公式:**

$$
\sin{\left(x_{19} \cdot 6.583 \frac{1}{e^{\sin{\left(\sin{\left(x_{4} \left(-0.2586\right) \frac{1}{e^{\sin{\left(\frac{1.884}{e^{\sin{\left(\frac{1.912}{\sin{\left(x_{13} \right)}} \right)}}} \right)} 0.4450}} \right)} \right)}}} \right)}
$$

---

### Gamma 信号: `yhat_ess5_up`

**最佳公式:**

$$
x_{19} + 0.0003642 e^{- 1.785 e^{- x_{19}} \log{\left(x_{16} \right)} \sin{\left(\frac{0.6111}{\sin{\left(x_{13} + x_{17} \right)}} \right)}}
$$

---

### Gamma 信号: `yhat_ess5_down`

**最佳公式:**

$$
\log{\left(e^{x_{19} \cos{\left(\frac{0.07597 \sin{\left(x_{0} \right)}}{\cos{\left(- x_{7} + x_{9} + 7.747 \right)}} \right)}} \right)}
$$

---

### Gamma 信号: `yhat_sop1_fwd`

**最佳公式:**

$$
x_{13} + \sin{\left(\frac{x_{12} x_{14}}{1.553} \right)} 0.2495
$$

---

### Gamma 信号: `yhat_sop1_rev`

**最佳公式:**

$$
\cos{\left(\frac{\cos{\left(2.162 \log{\left(x_{13} \right)} - \log{\left(x_{8} \right)} \right)}}{\sin{\left(x_{13} + 0.3524 \right)}} \right)}
$$

---

### Gamma 信号: `yhat_sop2_fwd`

**最佳公式:**

$$
\sin{\left(x_{20} \left(x_{5} + \sin{\left(\frac{2.009}{x_{13}} \right)}\right) \right)}
$$

---

### Gamma 信号: `yhat_sop2_rev`

**最佳公式:**

$$
\cos{\left(\cos{\left(\frac{x_{18}}{x_{6}} \right)} \right)}
$$

---

### Gamma 信号: `yhat_pv1`

**最佳公式:**

$$
x_{19} x_{19}
$$

---

### Gamma 信号: `yhat_pv2`

**最佳公式:**

$$
x_{19} \cdot 1.000
$$

---

### Gamma 信号: `yhat_pv3`

**最佳公式:**

$$
x_{19} x_{19}
$$

---

### Gamma 信号: `yhat_pv4`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_pv5`

**最佳公式:**

$$
x_{20} x_{20}
$$

---

### Gamma 信号: `yhat_pv6`

**最佳公式:**

$$
x_{19} \sin{\left(\cos{\left(x_{16} \left(x_{5} - -0.1516\right) \right)} - -0.9469 \right)}
$$

---

### Gamma 信号: `yhat_wt1`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt2`

**最佳公式:**

$$
x_{19} x_{20}
$$

---

### Gamma 信号: `yhat_wt3`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt4`

**最佳公式:**

$$
x_{19} \cdot 1.000
$$

---

### Gamma 信号: `yhat_wt5`

**最佳公式:**

$$
x_{20} \cdot 1.000
$$

---

### Gamma 信号: `yhat_wt6`

**最佳公式:**

$$
x_{19}
$$
