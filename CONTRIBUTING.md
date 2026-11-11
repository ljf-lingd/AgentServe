# 贡献指南

讨论功能或缺陷时，请给出当前版本、使用模式、模型和数据库方言。提供可分享的小型合成样例、执行步骤和预期结果。实际业务数据可用合成数据替代。

~~~bash
python -m pip install -e ".[web,test]"
python scripts/setup/prepare_public_demo.py
python -m pytest -q
npm --prefix web ci
npm --prefix web run typecheck
npm --prefix web run lint
npm --prefix web run build
~~~

保持 API 与 CLI 复用 application 层，由 storage 管理状态事务。新增数据集应登记 schema、业务口径和金标。新增功能优先覆盖正常入口与会影响结果的故障路径。模型质量或延迟变化需记录真实模型、样本分母和运行条件。

GitHub Issues 用于功能与缺陷反馈。安全问题参照 [SECURITY.md](SECURITY.md)。
