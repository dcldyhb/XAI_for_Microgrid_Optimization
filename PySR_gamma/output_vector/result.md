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
x_{20} \cos{\left(\frac{x_{5} - 0.7314}{x_{10} + \sin{\left(\frac{x_{5} - 0.7314}{x_{10} - 0.5171 - - \frac{0.8376}{\cos{\left(x_{0} \right)}}} \right)} - - \frac{0.8377}{\cos{\left(x_{0} \right)}}} \right)}
$$

---

### Gamma 信号: `yhat_ess2_up`

**最佳公式 (LaTeX):**

$$
\sin{\left(\frac{x_{20} - e^{\cos{\left(x_{9} \left(-0.9293\right) \right)} + \cos{\left(\frac{x_{10}}{-1.806} \right)} + \cos{\left(x_{0} + 1.824 \right)}} \left(-0.4715\right) \frac{1}{x_{0}}}{0.6853} \right)}
$$

---

### Gamma 信号: `yhat_ess2_down`

**最佳公式 (LaTeX):**

$$
x_{19} \sin{\left(3.667 \left(\sin{\left(\frac{x_{6}}{\sin{\left(0.4298 + \frac{x_{19}}{x_{12}} \right)}} \right)} - 0.3538\right) \right)}
$$

---

### Gamma 信号: `yhat_ess3_up`

**最佳公式 (LaTeX):**

$$
\cos{\left(e^{- (x_{19} - \cos{\left(\frac{\cos{\left(e^{e^{\sin{\left(e^{e^{\cos{\left(x_{15} \cos{\left(e^{x_{4}} \right)} \right)}}} \right)}}} \right)}}{e^{\cos{\left(x_{15} \cos{\left(e^{x_{4}} \right)} \right)}}} \right)}) - 0.2857} - 0.4402 \right)}
$$

---

### Gamma 信号: `yhat_ess3_down`

**最佳公式 (LaTeX):**

$$
x_{19} \cos{\left(\log{\left(x_{12} \right)} \cos{\left(\frac{1.506}{\sin{\left(\frac{1.199}{\sin{\left(\frac{0.6624 \left(- x_{16} - 0.2615\right)}{x_{4}} \right)}} \right)}} \right)} \right)}
$$

---

### Gamma 信号: `yhat_ess4_up`

**最佳公式 (LaTeX):**

$$
x_{20} - 0.07144 + \frac{0.04455}{x_{20} + \sin{\left(\frac{x_{17}}{x_{4}} + 1.197 \right)} + 0.4382}
$$

---

### Gamma 信号: `yhat_ess4_down`

**最佳公式 (LaTeX):**

$$
\sin{\left(\frac{\sin{\left(x_{20} \left(e^{\sin{\left(- \frac{0.7982}{x_{12}} \right)}} - \cos{\left(\frac{2.745}{x_{13}} \right)}\right) \right)}}{0.6142} \right)}
$$

---

### Gamma 信号: `yhat_ess5_up`

**最佳公式 (LaTeX):**

$$
\cos{\left(\left(- x_{20} + x_{4}\right) 1.621 \right)}
$$

---

### Gamma 信号: `yhat_ess5_down`

**最佳公式 (LaTeX):**

$$
x_{20} + \frac{3.568}{\left(2.279 - x_{14}\right) \left(x_{8} + 35.43 \left(- x_{20} - 1.152\right)\right)}
$$

---

### Gamma 信号: `yhat_sop1_fwd`

**最佳公式 (LaTeX):**

$$
\left(x_{13} + x_{20} \cdot 0.7780 - 0.1331\right) \sin{\left(\frac{0.5926}{x_{13}} \right)}
$$

---

### Gamma 信号: `yhat_sop1_rev`

**最佳公式 (LaTeX):**

$$
\cos{\left(\cos{\left(e^{e^{x_{4} + 0.1922}} \right)} + 0.6686 \right)}
$$

---

### Gamma 信号: `yhat_sop2_fwd`

**最佳公式 (LaTeX):**

$$
x_{19} e^{\frac{e^{- x_{10} + x_{16}}}{-6.318}}
$$

---

### Gamma 信号: `yhat_sop2_rev`

**最佳公式 (LaTeX):**

$$
\sin{\left(x_{19} - - \frac{6.787}{x_{4}} \right)}
$$

---

### Gamma 信号: `yhat_pv1`

**最佳公式 (LaTeX):**

$$
x_{20} x_{20}
$$

---

### Gamma 信号: `yhat_pv2`

**最佳公式 (LaTeX):**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_pv3`

**最佳公式 (LaTeX):**

$$
x_{19}
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
x_{20}
$$

---

### Gamma 信号: `yhat_pv6`

**最佳公式 (LaTeX):**

$$
x_{20} \cos{\left(\frac{0.009959}{\log{\left(x_{6} \right)}} \right)}
$$

---

### Gamma 信号: `yhat_wt1`

**最佳公式 (LaTeX):**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt2`

**最佳公式 (LaTeX):**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_wt3`

**最佳公式 (LaTeX):**

$$
x_{20}
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
x_{19}
$$

---

### Gamma 信号: `yhat_wt6`

**最佳公式 (LaTeX):**

$$
x_{20}
$$
