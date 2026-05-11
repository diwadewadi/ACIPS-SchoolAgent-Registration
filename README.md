# ACIPS SchoolAgent Registration

学校入学办理 Agent 原型仓库。

## 目录结构

```text
ACIPS-SchoolAgent-Registration/
├── ACPs-SDK/
└── school-agent-prototype/
```

- `ACPs-SDK/`: ACPs/AIP 本地 SDK。
- `school-agent-prototype/`: 学校入学办理演示项目，包含 Leader、4 个 Partner 和 Web 前端。

## Windows 快速测试

```powershell
cd school-agent-prototype
.\scripts\install-windows.ps1
```

编辑 `school-agent-prototype/.env`，填入自己的 `OPENAI_API_KEY`，然后启动：

```powershell
.\scripts\start-windows.ps1
```

浏览器访问：

```text
http://localhost:59200
```

详细说明见 [school-agent-prototype/README.school-agent.md](school-agent-prototype/README.school-agent.md)。
