# Label Studio 聚合式 ML Backend

本项目为 Label Studio 提供统一的机器学习服务。Label Studio 将项目的标注模板随请求发送给 ML 服务，ML 服务根据模板自动选择对应的标注流程，不需要为每种模板单独部署一个服务。

当前 `v1.0.0` 包含：

- 掩码语义分割：火山引擎 EntitySegment 生成 Mask，豆包 Seed 根据当前项目的标签对 Mask 分类。
- 光学字符识别（OCR）：豆包 Seed 一次请求完成文字识别、坐标生成和标签分类。

## 运行要求

- Docker Engine
- Docker Compose v2
- 已运行的 Label Studio
- Label Studio 上传文件在宿主机上的媒体目录

## 凭证传递方式

以下凭证不保存在 ML 服务的 `.env` 或 `docker-compose.yml` 中：

- 豆包 Seed API Key
- EntitySegment Access Key（AK）
- EntitySegment Secret Key（SK）
- Label Studio 用户 JWT
- Label Studio URL

用户在 Label Studio 中保存模型连接后，Label Studio 会在每次调用 ML 接口时按需传递这些信息：

- OCR：传递 Seed API Key。
- 掩码语义分割：传递 Seed API Key、EntitySegment AK 和 SK。
- 访问受保护文件：传递 Label Studio URL 和当前模型连接所属用户的 JWT。

请勿把上述密钥写入镜像、Compose 文件或代码仓库。

## 获取代码

```bash
git clone https://github.com/OpenCSGs/lb-llm-back.git
cd lb-llm-back
cp .env.example .env
```

## 配置环境变量

编辑 `.env`：

```env
USE_THIRD_PARTY_MODELS=true
LOG_LEVEL=INFO
ML_BACKEND_PORT=9090
ML_BACKEND_IMAGE=opencsgs/lb-llm-back:v1.0.0
WORKERS=1
THREADS=8

DOUBAO_MODEL=doubao-seed-2-1-turbo-260628
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_TIMEOUT=360
DOUBAO_MAX_IMAGE_SIZE=640

VOLCENGINE_ENTITY_SEGMENT_MODEL=entity_seg
VOLCENGINE_ENTITY_SEGMENT_MAX_ENTITY=10
VOLCENGINE_ENTITY_SEGMENT_REFINE_MASK=1
VOLCENGINE_ENTITY_SEGMENT_RETURN_FORMAT=3

LABEL_STUDIO_LOCAL_MEDIA_ROOT=/absolute/path/to/label-studio/data/media
```

参数说明：

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `USE_THIRD_PARTY_MODELS` | 是 | `false` | `true` 使用 EntitySegment 和 Seed；`false` 保留原专用模型路径。 |
| `ML_BACKEND_PORT` | 否 | `9090` | ML 服务在宿主机暴露的端口。 |
| `ML_BACKEND_IMAGE` | 否 | `opencsgs/lb-llm-back:v1.0.0` | 构建和运行使用的镜像名称。 |
| `LOG_LEVEL` | 否 | `INFO` | 日志级别。 |
| `WORKERS` | 否 | `1` | Gunicorn 进程数。外部接口调用型任务建议先保持为 `1`。 |
| `THREADS` | 否 | `8` | 每个 Gunicorn 进程的线程数。 |
| `DOUBAO_MODEL` | 否 | `doubao-seed-2-1-turbo-260628` | Seed 模型名称。 |
| `ARK_BASE_URL` | 否 | `https://ark.cn-beijing.volces.com/api/v3` | 方舟 API 地址。 |
| `DOUBAO_TIMEOUT` | 否 | `360` | 调用 Seed 的超时时间，单位为秒。 |
| `DOUBAO_MAX_IMAGE_SIZE` | 否 | `640` | Mask 分类辅助图的最大边长；OCR 仍发送原始图片。 |
| `VOLCENGINE_ENTITY_SEGMENT_MODEL` | 否 | `entity_seg` | 主体分割服务的 `req_key`。 |
| `VOLCENGINE_ENTITY_SEGMENT_MAX_ENTITY` | 否 | `10` | 单张图片最多返回的主体数量。 |
| `VOLCENGINE_ENTITY_SEGMENT_REFINE_MASK` | 否 | `1` | 是否细化 Mask。 |
| `VOLCENGINE_ENTITY_SEGMENT_RETURN_FORMAT` | 否 | `3` | EntitySegment 返回格式。 |
| `LABEL_STUDIO_LOCAL_MEDIA_ROOT` | 是 | 无 | Label Studio 媒体目录在宿主机上的绝对路径。 |

`BASIC_AUTH_USER` 和 `BASIC_AUTH_PASS` 为可选参数。只有需要给 ML 服务启用 Basic Auth 时才配置，并且必须同时填写。

## Label Studio 地址要求

ML 请求中的 `ls_url` 由 Label Studio 自动提供。Label Studio 自身的 `HOST` 必须配置为 ML 容器能够访问的地址，例如本机 Docker 开发环境可使用：

```env
HOST=http://host.docker.internal:8080
```

不要传递只能在 Label Studio 容器内部访问的 `localhost` 地址。生产环境应使用容器网络地址或可访问的正式域名。

## 构建镜像

```bash
docker compose build ml-backend
```

指定镜像名称构建：

```bash
ML_BACKEND_IMAGE=opencsgs/lb-llm-back:v1.0.0 docker compose build ml-backend
```

## 启动服务

```bash
docker compose up -d
```

代码修改后重新构建并替换容器：

```bash
docker compose up -d --build --force-recreate ml-backend
```

## 检查运行状态

```bash
docker compose ps
docker compose logs -f ml-backend
```

健康检查地址：

```text
http://localhost:9090/health
```

正常响应示例：

```json
{
  "model_class": "NewModel",
  "status": "UP"
}
```

## 在 Label Studio 中连接

在项目的模型设置中创建连接：

1. 后端 URL 填写 `http://<ML服务地址>:9090`。
2. 开启“使用三方模型”。
3. 填写 Seed API Key。
4. 如果项目使用分割模板，再填写 EntitySegment AK 和 SK。
5. 保存连接并确认连接状态正常。

如果当前用户没有有效的 Label Studio JWT，保存连接时应先创建 JWT。每次预测使用模型连接创建者的有效 JWT，不使用 ML 服务中的全局 Token。

## 常见问题

### ML 服务无法读取图片

依次检查：

1. `LABEL_STUDIO_LOCAL_MEDIA_ROOT` 是否为绝对路径。
2. 该目录是否与 Label Studio 的实际媒体目录一致。
3. ML 容器是否具有只读访问权限。
4. LS 请求传递的 `ls_url` 是否能从 ML 容器访问。
5. 模型连接所属用户的 JWT 是否有效。

### Seed 请求超时

默认超时时间为 360 秒。可在 `.env` 中增大 `DOUBAO_TIMEOUT`，然后重新创建容器：

```bash
docker compose up -d --force-recreate ml-backend
```

### Key 填写错误但页面没有结果

查看 ML 服务日志中的三方接口错误，并确认 Label Studio 将 ML 错误展示给用户：

```bash
docker compose logs -f ml-backend
```

### 查看完整启动配置

以下命令会解析 Compose 配置。请勿把包含敏感信息的输出发送给无关人员：

```bash
docker compose config
```
