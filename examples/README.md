# 可直接打开的示例项目

在主界面点击“打开项目”，选择对应目录下的 `project.optproj`。文件内已包含 60 行数据与完整配置，无需另外导入 CSV。

| 示例 | 项目 | 数据 |
| :--- | :--- | :--- |
| 电池液冷 | [battery/project.optproj](battery/project.optproj) | [battery/data.csv](battery/data.csv) |
| 换热器 | [exchanger/project.optproj](exchanger/project.optproj) | [exchanger/data.csv](exchanger/data.csv) |
| 结构设计 | [structure/project.optproj](structure/project.optproj) | [structure/data.csv](structure/data.csv) |

## 数据来源

全部为 `mtpv_optimizer.presets.example()` 生成的确定性函数演示，种子 73。样本包括上下界组合，整数或有限数值集合变量按其类型处理；`case` 以 `SYNTHETIC_` 开头，`batch` 为六组标记，`time` 是顺序索引。

这些演示变量有工程含义，但响应函数并非经过标定的物理模型，不能用于真实设计结论或物理验证。

### 电池液冷

令 `f = flow_L_min`、`c = channel_mm`、`n = channels`：

```text
Tmax_K   = 320 - 5f - 0.45c - 0.5n + 0.2f²
deltaT_K = 8 - 0.7f - 0.3c - 0.25n + 0.04c²
pump_W   = 0.5 + 1.2f² + 0.15n + 0.05c
```

三个目标均最小化，约束 `Tmax_K <= 315`；通道数量为整数变量。

### 换热器

令 `v = velocity_m_s`、`p = pitch_mm`：

```text
heat_W = 200 + 80v - 15p + 2vp
dp_Pa  = 20 + 15v² + 4p
```

热量最大化、压降最小化，约束 `dp_Pa <= 350`；间距为 `{2, 4, 6}` mm。

### 结构设计

令 `t = thickness_mm`、`w = width_mm`：

```text
mass_kg       = 0.002tw
deflection_mm = 30 / (t²w)
safety_margin = (tw - 100) / 100
cost          = 10 + 0.08tw + 0.1w²
```

质量、挠度、成本最小化，安全裕量最大化；约束 `safety_margin >= 0`。

## 示例运行设置

项目使用种群 80、40 代、两次独立运行、搜索种子 42、随机五折验证和 IDEAL 等权决策。电池/换热器使用二次响应面，结构使用高斯过程。相同设置仍可能因依赖版本和计算环境产生数值差异。

运行 `python scripts/capture_screenshots.py` 会重新生成三个示例并完成实际界面优化、截图。MTPV/MTEG 需要 [自行提供数据](../data/README.md)，真实研究记录不随仓库发布。
