# SEM Copilot（结构方程模型副驾）

> **一个轻量、透明、可审计的结构方程模型（SEM）引擎，作为 Chelate Shell 生态的子模块。**
> 面向社科研究者，以及所有需要可复现、高透明度建模流程的科研人员。

[English](README.md) · **简体中文**

**版本：** 1.0.0 | **可独立运行**的附属插件 | **架构层级：** 独立运行 / 主库自适应兼容

---

## 安全与开源声明

> **免责声明**：本仓库包含作者为自动化科研工作流而开发的辅助工具，并非经过同行评审的学术项目。部分样板代码可能在 AI 辅助下生成，并经过作者高层面的审查与修改。作者对任何第三方代码问题不承担责任，但承诺及时处理任何合理的诉求。
> 
> 生成的报告与图表仅为“数据分析草稿”，最终学术责任由使用者本人及所在机构承担。

---

## 为什么需要 SEM Copilot？

当前社科研究中的结构方程模型（SEM）分析，往往面临以下工程与透明度痛点：

- **闭源黑盒难以复核。** 传统图形化商业软件给出拟合指数，但底层算法封闭，审稿人难以从原始数据独立复现验证。
- **手动建模难以自动化。** 在图形界面中拖拽画图、连线，耗时且易错，无法无缝嵌入现代自动化的数据处理流水线。
- **跨平台与协作限制。** 某些软件依赖特定的操作系统或高昂的商业授权，限制了团队间的协作与开源共享。

### 极简体验
1. 使用自然语言或标准的 `lavaan` 语法描述你的模型。
2. SEM Copilot 自动拟合模型并输出详尽的统计指标。
3. 直接导出可用于顶刊投稿的矢量 PDF 路径图。

全程基于代码，无需鼠标，完全免费且支持跨平台运行。

---

## 技术背书：双引擎交叉验证

单一估计器的结果往往难以复核。SEM Copilot 同时运行两套彼此独立的引擎并交叉验证：

- **`semopy` 引擎**：成熟的开源 SEM 估计器，支持中介效应、多组不变性、有序变量 WLSMV、稳健 ML 标准误、修正指数与 bootstrap 置信区间。
- **纯 Numpy 独立重推**：不依赖 `semopy` 的从零实现，直接由数据协方差矩阵重算连续型 SEM 的似然与拟合指数（CFI / TLI / RMSEA）。

两套引擎各自独立实现，审稿人可以从原始数据独立复推出你的拟合指数。这在代码层面保证了计算的透明度与可审计性。

---

## 核心架构

本仓库是一个完全自洽的独立插件，既可作为单独工具使用，也可无缝融入主库生态。核心逻辑被严格收敛在**单一技能**中，确保无冗余文件散落。

```text
sem-copilot/
│
├── 核心基建层 (Base Infrastructure)
│   ├── core/             # 独立微内核 (Router, Sandbox)
│   └── contracts/        # 契约层 (gate_result)
│
├── 共享工具层 (Shared Utils)
│   └── lib/chelate_core/ # 基础适配器与路径处理
│
└── 闭环处理层 (Workflow Processor)
    └── skills/sem-copilot/ # 唯一的三合一技能
        ├── SKILL.md
        └── scripts/      # engine (拟合) / plotter (绘图) / exporter (排版)
```

**上下文自适应探测**：本工具会在每次运行时自动探测当前目录（或 `--root` 指向的目录）是否属于 Chelate Shell 的生态结构——即是否同时包含 `material/` 与 `process/` 两个区。
- **生态兼容模式**：若属于，则把所有产物写入 `<root>/process/{gates,figures,latex}/`。
- **独立运行模式**：若不属于，则安全回退到当前目录的 `./sem-copilot-output/` 下，零配置开箱即用。

---

## 安装与下载

本仓库是一个跨平台的 Agent 技能套件，支持 Claude Code、WorkBuddy、OpenAI Codex 以及 DeepSeek Harness (DSH)。请选择最适合你的安装方式：

### 方式一：通过 Git 链接安装（最推荐，Agent 原生方式）

1. 打开你所在平台的插件管理（或市场配置）界面。
2. 选择“从 Git 仓库安装”或“添加自定义市场”。
3. 输入本仓库地址并确认：`https://github.com/JiamingShi-JsPhys/sem-copilot`

### 方式二：让 AI 帮你装

直接复制以下 Prompt 发给你的 AI Agent。**请把它作为“一个插件”安装，不要把散技能铺进技能目录：**

> “请访问 GitHub 仓库 `https://github.com/JiamingShi-JsPhys/sem-copilot`，并把它**作为单一插件**安装（不要把 `skills/` 目录下的各个技能打散进全局技能目录，也不要压平仓库布局）。请遵循当前 Agent 平台的插件安装规范；只有在平台没有“插件”这一概念时，才退回到技能注册。”

> **警告**：使用 AI 自动安装时，AI **有可能**将 `skills/` 目录下的独立技能直接打散，平铺复制到平台的全局技能目录中。这会导致 sem copilot 的“微内核自动调度架构”失效，退化为散装技能。如果你需要完整的插件调度功能，请使用方式一。

**Claude Code（斜杠命令）** — 一行命令、直接在对话里装（Claude Code v2.1.275+）：

```
/plugin install sem-copilot --marketplace JiamingShi-JsPhys/sem-copilot
```

这条命令会拉取 Git 仓库并把 sem copilot 作为单一插件注册，一步到位——`--marketplace` 指定来源，Claude Code 会自动注册，无需手动克隆。若你的 Claude Code 版本较旧、不支持 `--marketplace`，改用两步写法：

```
/plugin marketplace add JiamingShi-JsPhys/sem-copilot
/plugin install sem-copilot@sem-copilot-marketplace
```

### 方式三：克隆仓库（开发者 / 查看底层代码）

如果你是开发者，或者希望直接查看底层代码，可以在终端中克隆本仓库：

```bash
git clone https://github.com/JiamingShi-JsPhys/sem-copilot.git
```

---

## 快速上手

所有命令均在插件根目录执行，绝不硬编码绝对路径。

```bash
# 1. 模型拟合与门禁检查 (Gate)
python skills/sem-copilot/scripts/run.py \
    --root <工作区> --action gate \
    --model "f1 =~ x1 + x2 + x3
f2 =~ y1 + y2 + y3
f2 ~ f1" \
    --data dataset.csv

# 2. 路径图绘制 (Render)
python skills/sem-copilot/scripts/run.py \
    --root <工作区> --action render \
    --model "..." --data dataset.csv

# 3. 导出 LaTeX 报告 (Export)
python skills/sem-copilot/scripts/run.py \
    --root <工作区> --action export
```

---

## 技能依赖矩阵

| 技能 | 必需依赖 | 可选依赖（缺失则降级功能） |
| --- | --- | --- |
| `sem-copilot` | `semopy`, `pandas`, `numpy` | `matplotlib`, `seaborn` (绘图), `sem_fit` (交叉验证) |

所有重型依赖均通过沙箱隔离运行。缺失依赖时，系统会优雅降级为 `UNVERIFIED` 并给出精确提示，不会导致整个流程崩溃。

---

## 已知局限 (Limitations)

诚实的边缘情况声明；依赖以下输出前请务必注意：

- **模型拟合非绝对证明**：引擎输出的拟合指数（CFI/RMSEA）基于 ML 估计与特定的优化算法。虽然双引擎交叉验证极大提高了可信度，但结果仍需结合理论背景判断，不能完全替代人工对模型的审查。
- **双引擎结果非逐位一致**：`semopy` 与纯 Numpy 重推引擎在底层优化器上存在差异，极少数边缘模型（如恰好识别模型）可能无法完全收敛，此时系统会标记 `UNVERIFIED`。
- **非图形界面替代品**：本工具不提供类似传统商业软件的拖拽建模界面，所有模型定义均需通过代码语法或自然语言辅助进行描述。

---

## 报告问题与反馈

**常规 Bug / 功能建议**：
首选在仓库的 [Issues](https://github.com/JiamingShi-JsPhys/sem-copilot/issues) 页面提交。提交时请附上：复现步骤、预期行为、实际行为以及运行环境（操作系统、Python 版本）。

**安全漏洞报告（仅限此类型）**：
请**不要**在公开的 Issues 页面提交安全漏洞。请发送邮件至 `jiamingshi_git@outlook.com`。
- **邮件主题**：必须包含前缀 `report` 以及具体技能名与简述。
- **邮件正文**：必须包含详细的复现步骤、运行环境。

*提交清晰、可复现 Issue 或 Pull Request 的贡献者，将被列入下个版本 Release 说明的致谢名单。*

---

## 开源许可与作者

本仓库整体采用 [MIT License](LICENSE) 发布，详见根目录下的 `LICENSE` 文件。

**作者与维护者**：Jiaming Shi (Westlake) & Keqia Ying (ZUEL)

---

## 版权与免责声明

SEM Copilot is an independent open-source project and is not affiliated with, endorsed by, or sponsored by Microsoft Corporation. "Copilot" is a trademark of Microsoft Corporation.
