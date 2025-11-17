# PySR Gamma 信号公式汇总

## 特征索引映射表

公式中的 $x_i$ 符号与以下实际特征相对应：

| 符号 ($x_i$) | 特征变量名 |
| :--- | :--- |
| $x_0$ | `temperature_C` |
| $x_1$ | `wind_speed_ms` |
| $x_2$ | `illuminance_lux` |
| $x_3$ | `precipitation_mm` |
| $x_4$ | `elec_vmin` |
| $x_5$ | `elec_vmax` |
| $x_6$ | `elec_vavg` |
| $x_7$ | `elec_line_loading_max` |
| $x_8$ | `elec_line_loading_top1` |
| $x_9$ | `elec_line_loading_top2` |
| $x_10$ | `elec_line_loading_top3` |
| $x_11$ | `elec_vmax_nonslack` |
| $x_12$ | `elec_line_i_ka_max` |
| $x_13$ | `elec_p_loss_mw` |
| $x_14$ | `elec_soc_1` |
| $x_15$ | `elec_soc_2` |
| $x_16$ | `elec_soc_3` |
| $x_17$ | `elec_soc_4` |
| $x_18$ | `elec_soc_5` |
| $x_19$ | `elec_any_violate` |
| $x_20$ | `elec_violate_v` |
| $x_21$ | `elec_violate_line` |


---

### Gamma 信号: `yhat_ess1_up`

**最佳公式 (LaTeX):**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_ess1_down`

**最佳公式 (LaTeX):**
$$
x_{19} \cos{\left(\frac{\sin{\left(x_{1} \right)} \cos{\left(\frac{\log{\left(x_{14} \right)}}{\sin{\left(\frac{4.622}{\sin{\left(x_{9} \right)}} \right)}} \right)}}{x_{2} + x_{7} + 1.305 \left(- x_{16} - 7.151\right) \cos{\left(x_{0} - 4.622 \right)}} \right)}
$$

---

### Gamma 信号: `yhat_ess2_up`

**最佳公式 (LaTeX):**

$$
x_{19} + e^{e^{- x_{19} + \frac{\sin{\left(\frac{1.000 - x_{9}}{-2.481} \right)} - \sin{\left(\frac{x_{17}}{x_{13} - 0.6748} \right)}}{x_{6}}}} \cdot 0.0003976
$$

---

### Gamma 信号: `yhat_ess2_down`

**最佳公式 (LaTeX):**

$$
\sin{\left(e^{\sin{\left(\sin{\left(\frac{\sin{\left(\frac{0.5946}{2.146 \sin{\left(\frac{0.6002}{1.055 x_{13}} \right)}} \right)}}{0.4505} \right)} \right)}} x_{20} \cdot 0.7304 \right)}
$$

---

### Gamma 信号: `yhat_ess3_up`

**最佳公式 (LaTeX):**

$$
x_{20} + \frac{0.3848}{e^{e^{x_{20} + \sin{\left(x_{10} \cdot 0.5234 \right)} - -1.120} + \sin{\left(x_{9} \right)}} e^{\sin{\left(0.7969 x_{10} \right)}}}
$$

---

### Gamma 信号: `yhat_ess3_down`

**最佳公式 (LaTeX):**

$$
1.030 x_{20} \cos{\left(\frac{\cos{\left(48.26 x_{13} \log{\left(x_{9} \right)} \right)}}{x_{12} x_{15} \cos{\left(8.903 x_{13} \right)}} \right)}
$$

---

### Gamma 信号: `yhat_ess4_up`

**最佳公式 (LaTeX):**

$$
x_{19} x_{4} - - 0.03253 e^{\frac{\sin{\left(- \frac{x_{10}}{0.9844} - 0.2979 \right)}}{- \sin{\left(\sin{\left(x_{10} \left(-0.7279\right) \right)} \right)} - 1.173}}
$$

---

### Gamma 信号: `yhat_ess4_down`

**最佳公式 (LaTeX):**

$$
\sin{\left(\frac{x_{19}}{\cos{\left(\frac{\cos{\left(\cos{\left(\frac{- x_{11} - 0.4137}{x_{13}} \right)} \right)}}{x_{4} \left(\frac{x_{12}}{-0.3428} - 0.2867\right)} \right)}} \right)}
$$

---

### Gamma 信号: `yhat_ess5_up`

**最佳公式 (LaTeX):**

$$
x_{19} + \frac{\sin{\left(x_{9} \right)}}{x_{16} \frac{1}{1.063 - x_{19}} \left(\cos{\left(\frac{x_{9}}{-2.163} \right)} - 1.063\right)}
$$

---

### Gamma 信号: `yhat_ess5_down`

**最佳公式 (LaTeX):**

$$
x_{19} \sin{\left(e^{e^{\cos{\left(x_{9} \left(-12.19\right) \frac{1}{10.45 \left(x_{16} - x_{9} - \cos{\left(x_{12} x_{3} - x_{9} \right)}\right)} \right)}}} \right)}
$$

---

### Gamma 信号: `yhat_sop1_fwd`

**最佳公式 (LaTeX):**

$$
\sin{\left(\frac{x_{11} + x_{13} + 1.765 + \frac{x_{19}}{1.824 x_{13}}}{-0.9306} \right)}
$$

---

### Gamma 信号: `yhat_sop1_rev`

**最佳公式 (LaTeX):**

$$
\cos{\left(0.8309 - \cos{\left(e^{x_{6} \cdot 3.355} \right)} \right)}
$$

---

### Gamma 信号: `yhat_sop2_fwd`

**最佳公式 (LaTeX):**

$$
\sin{\left(x_{19} + \sin{\left(\frac{x_{19} \cos{\left(\frac{2.317}{x_{13}} \right)}}{-0.7918} \right)} \right)}
$$

---

### Gamma 信号: `yhat_sop2_rev`

**最佳公式 (LaTeX):**

$$
\cos{\left(\cos{\left(\frac{x_{14}}{x_{6}} \right)} \right)}
$$

---

### Gamma 信号: `yhat_pv1`

**最佳公式 (LaTeX):**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_pv2`

**最佳公式 (LaTeX):**

$$
x_{19} x_{20}
$$

---

### Gamma 信号: `yhat_pv3`

**最佳公式 (LaTeX):**

$$
x_{19} x_{20}
$$

---

### Gamma 信号: `yhat_pv4`

**最佳公式 (LaTeX):**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_pv5`

**最佳公式 (LaTeX):**

$$
x_{19} \cdot 1.000
$$

---

### Gamma 信号: `yhat_pv6`

**最佳公式 (LaTeX):**

$$
x_{20} \cos{\left(- x_{13} + x_{21} \right)}
$$

---

### Gamma 信号: `yhat_wt1`

**最佳公式 (LaTeX):**

$$
x_{20} x_{20}
$$

---

### Gamma 信号: `yhat_wt2`

**最佳公式 (LaTeX):**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt3`

**最佳公式 (LaTeX):**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt4`

**最佳公式 (LaTeX):**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_wt5`

**最佳公式 (LaTeX):**

$$
x_{19} x_{19}
$$

---

### Gamma 信号: `yhat_wt6`

**最佳公式 (LaTeX):**

$$
x_{19}
$$
