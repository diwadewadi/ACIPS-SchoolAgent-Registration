# ACIPS SchoolAgent Registration

学校入学办理 Agent 原型仓库。

## 目录结构

```text
ACIPS-SchoolAgent-Registration/
├── ACPs-SDK/
├── school-agent-prototype/
├── install-windows.ps1
└── start-windows.ps1
```

- `ACPs-SDK/`: ACPs/AIP 本地 SDK。
- `school-agent-prototype/`: 学校入学办理演示项目，包含 Leader、4 个 Partner 和 Web 前端。
- `install-windows.ps1`: Windows 本地依赖安装脚本。
- `start-windows.ps1`: Windows 本地演示启动脚本。

## Windows 快速测试

首次安装：

```powershell
.\install-windows.ps1
```

安装脚本会在 `school-agent-prototype/.venv` 创建虚拟环境，并安装 `school-agent-prototype/requirements.txt` 与本地 `ACPs-SDK`。

然后编辑：

```text
school-agent-prototype/.env
```

填入自己的兼容 OpenAI API Key：

```env
OPENAI_API_KEY=replace-with-your-api-key
```

启动演示：

```powershell
.\start-windows.ps1
```

脚本会打开 3 个 PowerShell 窗口，分别启动 Partner、Leader 和 Web。浏览器访问：

```text
http://localhost:59200
```

## 示例输入

```text
我是2026级本科新生张三，学号2026123456，计算机学院，沙河校区，男生，9月2日下午到校，已缴费并上传照片，帮我完成入学报到。
```

原型说明见 [school-agent-prototype/README.school-agent.md](school-agent-prototype/README.school-agent.md)。
