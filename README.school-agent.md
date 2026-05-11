# 学校入学办理 Agent 原型

这是基于 ACPs Demo Project 复制出的学校代理原型。当前版本为演示模式，不接入真实学校系统。

## 原型能力

- 报到注册：身份信息、学院/专业、缴费和绿色通道提示
- 宿舍入住：校区、性别、到校时间、特殊住宿需求和入住准备
- 校园卡与账号激活：统一身份认证、校园网、邮箱、校园卡领取/充值
- 入学材料提交：录取通知书、身份证、照片、档案、党团关系、体检表等材料检查

## 关键路径

- 场景配置：`leader/scenario/expert/school_onboarding/`
- Partner 配置：`partners/online/school_registration/`
- Partner 配置：`partners/online/dormitory_checkin/`
- Partner 配置：`partners/online/campus_account/`
- Partner 配置：`partners/online/admission_materials/`
- 前端入口：`web_app/index.html`

## 示例输入

```text
我是2026级本科新生张三，学号2026123456，计算机学院，沙河校区，男生，9月2日下午到校，已缴费并上传照片，帮我完成入学报到。
```

也支持单项服务：

```text
我只想查宿舍入住需要准备什么。
帮我激活统一身份认证账号和校园卡。
检查我的入学材料还缺什么。
```

## 运行

先配置可用的 OpenAI 兼容模型 Key。`leader/config.toml` 和各 Partner 的 `config.toml` 默认会读取环境变量 `OPENAI_API_KEY`。本项目依赖的 ACPs SDK 位于上一级目录 `../ACPs-SDK/`，`pyproject.toml` 已配置为本地路径依赖。

```bash
cd school-agent-prototype
./run.sh start
```

前端默认访问：

```text
http://localhost:59200
```

## 说明

当前原型已经将原旅游 Partner 移到 `partners/offline/tour_demo_reference/`，并将旧 Leader 场景移到 `leader/scenario/offline_reference/`。运行时 `partners/online/` 只保留学校入学办理相关的 4 个 Partner，`leader/scenario/expert/` 只保留 `school_onboarding` 场景。

## Windows 快速测试

给同学 clone 后测试时，目录建议保持为：

```text
school-agent/
├── ACPs-SDK/
└── school-agent-prototype/
```

首次安装：

```powershell
cd school-agent-prototype
.\scripts\install-windows.ps1
```

然后编辑 `.env`，填入自己的 `OPENAI_API_KEY`。`.env` 已被 `.gitignore` 忽略，不会提交到仓库。

启动演示：

```powershell
.\scripts\start-windows.ps1
```

脚本会分别打开 Partner、Leader、Web 三个 PowerShell 窗口。浏览器访问：

```text
http://localhost:59200
```
