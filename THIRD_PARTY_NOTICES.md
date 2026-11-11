# 数据、模型和第三方资料

[MIT](LICENSE) 适用于 AgentServe 的自有代码与文档。以下外部资源依照各自的许可使用。

| 资源 | 来源与范围 |
| --- | --- |
| Online Retail II | Daqing Chen, *Online Retail II* (2012), [UCI](https://doi.org/10.24432/C5CG6D)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。使用方从 UCI 下载原件，导入程序将两张工作表转换为 SQLite 并添加只读视图。本仓库的 UCI 规则示例描述了字段和本项目的统计口径。 |
| Spider 1.0 | Tao Yu 等，*Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task* (EMNLP 2018)，[官方项目页](https://yale-lily.github.io/spider) · [数据许可 CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。`data/splits/spider_g5_v1` 和 `data/gold/spider_g5_v1` 内的 Spider 派生题目与参考 SQL 遵循同一许可。本项目对公开开发题做筛选、重编号、分组和 JSON 转换。原始数据库从官方渠道另取。 |
| BIRD Mini-Dev | [数据作者发布页](https://huggingface.co/datasets/birdsql/bird_mini_dev)。评测脚本读取用户自行准备的原始数据和数据库。 |
| Qwen 与其他模型 | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [Vicuna 7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)。从模型发布者渠道获取权重和 tokenizer，并查看相应许可。 |

第三方 Python 与前端依赖的版本见 [pyproject.toml](pyproject.toml)、[envs](envs) 和 [web/package-lock.json](web/package-lock.json)，许可随各依赖发布。
