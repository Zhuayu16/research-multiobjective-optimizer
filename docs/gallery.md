# 软件界面与验证图集

[返回首页](../README.md) · [验证方法、指标与数据](verification.md)

**18 张真实软件截图 + 8 组数值验证图。** 界面截图直接由运行中的 Qt 窗口抓取；公开案例全部来自合成或解析函数。截图复现：[基础视图](../scripts/capture_screenshots.py) · [计算与验证流程](../scripts/capture_workflow.py)。

## 定义问题与启动计算

变量类型、边界及数值集合；通用研究界面中的目标权衡；工程约束；模型和搜索设置。

![变量定义](assets/workflow/variables.png)

![目标权衡与偏好排序](assets/overview.png)

![响应约束](assets/constraints.png)

![模型和运行设置](assets/workflow/run_settings.png)

## 模型比较与独立留出

Auto 实际比较多个模型族；计算汇总记录种子与数据；电池示例按批次保留 10 个独立测试样本、50 个训练样本。二次合成函数的近零误差仅说明模型与函数相容。

![Auto 模型比较](assets/workflow/model_comparison.png)

![运行汇总与追溯](assets/workflow/summary.png)

![独立留出样本预测](assets/workflow/holdout.png)

![交叉验证视图](assets/validation.png)

## 检查候选与辅助约束

预测候选与输入样本中的 Pareto 点分别展示。覆盖距离用于检查候选与已有数据的接近程度；它不等于置信区间。辅助响应案例训练三个响应，只优化两个目标，并将温差用于约束。

![预测候选](assets/workflow/predicted.png)

![输入表可行 Pareto](assets/workflow/input_pareto.png)

![候选覆盖与响应验证](assets/workflow/coverage.png)

![三响应两目标的辅助约束](assets/workflow/auxiliary_response.png)

![偏好方法比较](assets/workflow/preferences.png)

## 更换对象、恢复项目与解析答案

换热器使用两个混合方向目标和数值集合变量；结构设计使用四个目标。项目恢复图来自重开保存项目后的实际重新计算；解析最优解截图来自验证脚本生成的数据。专用 MTPV/MTEG 截图为尚未加载研究数据的界面。

![换热器两目标示例](assets/exchanger.png)

![结构设计四目标示例](assets/structure.png)

![恢复项目后重新计算](assets/workflow/project_restored.png)

![连续解析问题的优化结果](assets/workflow/known_optimum.png)

![MTPV/MTEG 专用界面](assets/mtpv.png)

## 数值结果与已知答案对照

方法、种子、阈值与适用范围见 [验证报告](verification.md)，原始表格见 [结果目录](../validation/results/)。图中使用合成函数真值，不代表实测或 CFD 验证。

![二维 ZDT1 前沿与等预算随机基线](assets/verification/zdt1_benchmark.png)

![连续和混合变量问题的解析最优解误差](assets/verification/known_optima.png)

![独立留出预测](assets/verification/independent_holdout.png)

![三个合成领域的原函数复算](assets/verification/domain_checks.png)

![电池液冷合成案例](assets/verification/battery_pareto.png)

![换热器合成案例](assets/verification/exchanger_pareto.png)

![结构设计合成案例](assets/verification/structure_pareto.png)

![混合约束问题的验证视图](assets/verification/mixed_validation.png)

### 可编辑及矢量文件

| 图 | SVG | PDF |
| :--- | :--- | :--- |
| ZDT1 前沿与基线 | [SVG](assets/verification/zdt1_benchmark.svg) | [PDF](assets/verification/zdt1_benchmark.pdf) |
| 解析最优解误差 | [SVG](assets/verification/known_optima.svg) | [PDF](assets/verification/known_optima.pdf) |
| 独立留出 | [SVG](assets/verification/independent_holdout.svg) | [PDF](assets/verification/independent_holdout.pdf) |
| 跨领域原函数复算 | [SVG](assets/verification/domain_checks.svg) | [PDF](assets/verification/domain_checks.pdf) |
| 电池 Pareto | [SVG](assets/verification/battery_pareto.svg) | [PDF](assets/verification/battery_pareto.pdf) |
| 换热器 Pareto | [SVG](assets/verification/exchanger_pareto.svg) | [PDF](assets/verification/exchanger_pareto.pdf) |
| 结构 Pareto | [SVG](assets/verification/structure_pareto.svg) | [PDF](assets/verification/structure_pareto.pdf) |
| 混合约束验证 | [SVG](assets/verification/mixed_validation.svg) | [PDF](assets/verification/mixed_validation.pdf) |
