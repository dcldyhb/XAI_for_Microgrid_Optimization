# PySR Gamma 信号公式汇总

## 特征索引映射表

公式中的 $x_i$ 符号与以下实际特征相对应：

| 符号 ($x_i$) | 特征变量名 |
| :--- | :--- |
| $x_{0}$ | `temperature_C` |
| $x_{1}$ | `wind_speed_ms` |
| $x_{2}$ | `illuminance_lux` |
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


---

### Gamma 信号: `yhat_ess1_up`

**最佳公式:**

$$
x_{19} x_{19}
$$

---

### Gamma 信号: `yhat_ess1_down`

**最佳公式:**

$$
\sin{\left(x_{20} \left(1.560 + \frac{0.06526}{- \left(x_{10} + 0.08374\right) \left(x_{3} + \cos{\left(x_{0} \right)}\right) \left(\log{\left(x_{0} \right)} + 0.8425\right) - 3.839}\right) \right)}
$$

---

### Gamma 信号: `yhat_ess2_up`

**最佳公式:**

$$
x_{19} - \frac{0.002407}{x_{19} - \sin{\left(x_{15} \left(2.731 - x_{6}\right) \right)} \left(-0.3877\right)}
$$

---

### Gamma 信号: `yhat_ess2_down`

**最佳公式:**

$$
x_{20} \cos{\left(\sin{\left(\sin{\left(\frac{0.4251}{\sin{\left(- \frac{1.769}{\sin{\left(e^{\sin{\left(- \frac{1.759}{x_{13}} \right)} + 0.9009} \right)}} \right)}} \right)} \right)} + 0.5576 \right)}
$$

---

### Gamma 信号: `yhat_ess3_up`

**最佳公式:**

$$
\sin{\left(1.524 \left(x_{20} + \left(- (\sin{\left(x_{10} \right)} + \sin{\left(x_{8} \right)}) - 0.9575\right) \left(x_{4} - 0.9799\right) 2.408\right) \right)}
$$

---

### Gamma 信号: `yhat_ess3_down`

**最佳公式:**

$$
x_{19} \left(\cos{\left(\frac{0.2181}{\sin{\left(\frac{x_{0}}{\frac{1}{\frac{1}{x_{0}} x_{16}} \left(x_{13} x_{14} - 1.190\right)} \right)}} \right)} 0.5879 + 0.4409\right)
$$

---

### Gamma 信号: `yhat_ess4_up`

**最佳公式:**

$$
\cos{\left(\cos{\left(x_{17} \cos{\left(x_{4} \right)} \right)} - - \frac{0.2305}{x_{20} - 0.3628} \right)}
$$

---

### Gamma 信号: `yhat_ess4_down`

**最佳公式:**

$$
\sin{\left(x_{19} e^{e^{\sin{\left(- \sin{\left(- \frac{2.548}{x_{13}} \right)} - 0.8261 \right)}}} \right)}
$$

---

### Gamma 信号: `yhat_ess5_up`

**最佳公式:**

$$
x_{19} + 0.007158
$$

---

### Gamma 信号: `yhat_ess5_down`

**最佳公式:**

$$
x_{20} \cos{\left(\frac{0.4666}{x_{12} + x_{15} - x_{9} - 0.2209 \left(- x_{7} - 0.1686\right) - 0.4569} \right)}
$$

---

### Gamma 信号: `yhat_sop1_fwd`

**最佳公式:**

$$
\sin{\left(\left(x_{13} \cdot 1.437 + 6.052\right) e^{x_{19} \left(x_{12} - 0.1703\right)} \right)}
$$

---

### Gamma 信号: `yhat_sop1_rev`

**最佳公式:**

$$
\cos{\left(\sin{\left(x_{19} + \cos{\left(x_{16} \cdot 1.055 \frac{1}{x_{4}} \right)} \right)} - 0.8088 \right)}
$$

---

### Gamma 信号: `yhat_sop2_fwd`

**最佳公式:**

$$
x_{20} \sin{\left(x_{18} - \sin{\left(- x_{20} - \frac{2.448}{x_{13}} \right)} - 3.880 \right)}
$$

---

### Gamma 信号: `yhat_sop2_rev`

**最佳公式:**

$$
\cos{\left(\cos{\left(\frac{x_{15}}{x_{6}} \right)} \right)}
$$

---

### Gamma 信号: `yhat_pv1`

**最佳公式:**

$$
x_{20} x_{20}
$$

---

### Gamma 信号: `yhat_pv2`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_pv3`

**最佳公式:**

$$
x_{19} \cdot 1.000
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
x_{19}
$$

---

### Gamma 信号: `yhat_pv6`

**最佳公式:**

$$
\sin{\left(\frac{x_{20} \left(x_{21} + 0.2997\right)}{\frac{1}{1.106} x_{13}} \right)}
$$

---

### Gamma 信号: `yhat_wt1`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_wt2`

**最佳公式:**

$$
x_{20}
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
x_{19}
$$

---

### Gamma 信号: `yhat_wt5`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt6`

**最佳公式:**

$$
x_{19} x_{19}
$$
