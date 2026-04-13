
套牌车识别项目版本管理
 # 日志
 ## 2025-11-12

- **[新增] 图像裁切预处理类**
  - 文件：`data_chuli/cropper.py`
  - 内容：新增 `VehicleCropper`，使用 YOLOv8 检测车辆并裁切最大目标，可选对车牌做黑色打码（HyperLPR3）。输入输出均为内存中的 `PIL.Image`。

- **[集成] GUI 预测前调用裁切**
  - 文件：`Siamese-pytorch-master/my_predict_gui.py`
  - 变更：导入并初始化 `VehicleCropper`，在 `predict_similarity` 中对两张图片先 `process_pil` 后再送入 Siamese 比对。

- **[修复] 概率格式化报错**
  - 文件：`Siamese-pytorch-master/my_predict_gui.py`
  - 变更：将 `detect_image` 的返回 `Tensor` 转为 `float` 再比较/格式化，避免 “unsupported format string passed to Tensor.__format__”。

- **[版本控制] 放开日志文件追踪**
  - 文件：`.gitignore`
  - 变更：新增 `!开发日志.md`，允许将开发日志提交到 Git。

## 2025-11-13

- **[新增] 批量套牌检测脚本**
  - 文件：`Siamese-pytorch-master/detect_clone_plates.py`
  - 功能：按车牌分组、仅向过去寻找“最近一趟”有图记录进行相似度比对；当前行无图=不可判定；仅使用 `过皮部位1图片URL`；默认阈值 `0.3`；输出 `clone_check_report.csv`。
  - 复用：`siamese.Siamese` 与 `data_chuli.cropper.VehicleCropper` 的预处理/推理流程。

- **[新增] 可视化批处理 GUI**
  - 文件：`Siamese-pytorch-master/clone_checker_gui.py`
  - 功能：选择 CSV、一键运行、展示当前/参考图片信息、相似度与判定结果；支持阈值调整；结果保存路径提示。
  - 复用：与命令行一致的模型与裁剪流程。

- **[修复] 导入路径**
  - 文件：`Siamese-pytorch-master/clone_checker_gui.py`
  - 变更：修正为从同目录导入 `detect_from_csv`，避免包路径错误。

- **[新增] 数据统计小工具**
  - 文件：`data_chuli/data_tongji.py`
  - 功能：按 `车号` 统计出现次数，输出 `plate_counts.csv` 与 `duplicate_rows.csv`，用于快速查看重复车牌与明细。

 
 ## 2026-01-05
 
 ### 阈值调整（疑似套牌）
 - **变更内容**：将“疑似套牌”头部相似度阈值默认值调整为 `0.8`。
 - **判定规则**：`head_prob < 0.8` 判定为疑似套牌（等于 `0.8` 不判定为低）。
 - **同步范围**：主批量检测逻辑 + “本地两图比对”工具页。
 
 ### 数据库批量检测：从指定日期开始（按 TASK_ID）
 - **入口**：“从数据库批量检测”弹窗新增模式 `从指定日期开始...`。
 - **日期选择**：日历选择起始日期（包含当天）。
 - **过滤规则**：按 `TASK_ID` 前 6 位 `yyMMdd` 解析日期，过滤 `TASK_YYMMDD >= 起始日期yyMMdd`。
 - **清理策略**：自动删除 `D:\output` 下起始日期之前的结果文件夹（按文件夹名中的 `TASK_ID` 前缀判断）。
 - **结果更新**：覆盖写入默认 CSV（无疑似结果时也覆盖为空表头，避免旧结果残留）。
 - **状态更新**：`last_task_id` 使用本次参与检测数据的最大 `TASK_ID` 更新。

 ## 2026-01-15

 - **[新增] Flask 后端推理服务（两图比对：车头/车尾）**
   - 文件：`Siamese-pytorch-master/my_predict_gui_new1.py`
   - 接口：`GET /health`、`POST /predict`
   - 输出：`head_prob` / `tail_prob` 与 `case_type`

 - **[新增] 图片路径校验与安全控制**
   - 规则：必须绝对路径、必须存在且是图片扩展名（`.jpg/.jpeg/.png/.bmp/.webp`）
   - 白名单：支持 `ALLOWED_BASE_DIRS` 限制可访问目录

 - **[新增] 本地缺失文件的远程拉取（可关闭）**
   - 组件：`data_tran.image_resolver.ImagePathResolver`
   - 开关：`REMOTE_FETCH_ENABLED`（关闭后本地不存在直接报错）

 - **[新增] 车头/车尾部位裁切**
   - 方式：YOLO 检测车头/车尾框并裁切最高置信度目标（`cls_id=0` 车头、`cls_id=1` 车尾）
   - 模型：`HEADTAIL_MODEL_PATH`

 - **[增强] 并发保护**
   - 初始化：`_INIT_LOCK`
   - 推理：`_INFER_LOCK`

 - **[新增] Web 前端页面（远端浏览器访问）**
   - 页面：`GET /ui`
   - 前端文件：`Siamese-pytorch-master/templates/ui.html`、`Siamese-pytorch-master/static/ui.css`、`Siamese-pytorch-master/static/ui.js`
   - 功能：支持“路径/链接预测”和“本地上传预测”，并提供结果下载（JSON/CSV）按钮

 - **[新增] 本地上传预测接口（适配远端电脑图片在本机）**
   - 接口：`POST /predict_upload`（`multipart/form-data`）
   - 字段：`file1`、`file2`

 - **[增强] /predict 支持 http(s) 图片链接**
   - 说明：当 `path1/path2` 为 `http(s)://...` 时，服务端先拉取到本地再推理

 - **[新增] 预览推理接口（返回 6 张裁切图）**
   - 接口：`POST /predict_preview`、`POST /predict_upload_preview`
   - 返回：在原有 `head_prob/tail_prob/case_type` 基础上增加 `previews`（6 张图的 base64 dataURL）

 - **[增强] /ui 页面简洁改版 + 使用教程**
   - 标题：`过磅车辆智能识别系统 v4.2`
   - 风格：白底简洁
   - 功能：新增“使用教程”弹窗，结果区支持 6 图可视化与概率进度条

 ## 2026-01-22

 - **[修复] Ultralytics 数据集切分输出结构**
   - 文件：`truck_detect/split_train_val.py`
   - 变更：输出目录固定为 `images/train|val` + `labels/train|val`（YOLO txt）；不再导出/复制 XML 到 split 数据集。

 - **[更新] 检测权重默认切换为训练得到的 best.pt**
   - 文件：`truck_detect/truck_detect.py`、`truck_detect/export_labelimg_xml.py`
   - 变更：默认模型路径改为自训 `best.pt`，并保留找不到时回退到旧 `yolo26m.pt`。

 - **[修复] 自训模型类别过滤导致“检测不到框”**
   - 文件：`truck_detect/truck_detect.py`、`truck_detect/export_labelimg_xml.py`
   - 变更：不再硬编码 COCO 类别；根据 `model.names` 自动推断类别（自训仅 0 类时可正常出框）。

 - **[更新] 后端车辆裁切改用自训车辆检测权重（仅 0 类）**
   - 文件：`data_chuli/cropper.py`
   - 变更：默认车辆检测权重切换为自训 `best.pt`；类别默认 `[0]`；裁切策略改为“取最靠近中心的检测框”。

 - **[新增] 车辆先裁切再车牌打码（可视化测试脚本）**
   - 文件：`data_chuli/plate_mask_yolo.py`
   - 功能：先用车辆检测框裁切车辆，再用车牌模型检测框打码（黑块/模糊）；封装 `PlateMasker` 类，运行脚本只需改 `DEMO_IMAGE_PATH` 即可弹窗可视化；可视化自动缩放适配屏幕。

 # 后端服务（Flask）

 ## 服务入口

 - 文件：`Siamese-pytorch-master/my_predict_gui_new1.py`
 - 框架：Flask
 - 功能：对两张车辆图片做预处理与裁切，然后分别计算“车头相似度/车尾相似度”，并给出简单分类结果。

 ## 核心流程（服务内部）

 - **车辆裁切预处理**：`data_chuli.cropper.VehicleCropper().process_pil()`
 - **部位裁切**：使用 YOLO 检测车头/车尾框（`ultralytics.YOLO`）
   - `cls_id=0`：车头
   - `cls_id=1`：车尾
 - **相似度模型**：分别用两套 Siamese 模型计算
   - 车头：`Siamese(model_path=HEAD_MODEL_PATH)`
   - 车尾：`Siamese(model_path=TAIL_MODEL_PATH)`
 - **并发控制**：初始化使用 `_INIT_LOCK`，推理使用 `_INFER_LOCK`，避免多线程并发导致模型状态异常。

 ## 接口说明

 ### `GET /`

 - 返回：可用 endpoints 列表

 ### `GET /health`

 - 返回：`{"status":"ok"}`

 ### `GET /ui`

 - 返回：Web 前端页面（浏览器访问入口）
 - 说明：前端静态资源位于 `Siamese-pytorch-master/static/`

 ### `POST /predict`

 - **Content-Type**：`application/json`
 - **请求体**：
   - `path1`：图片1的绝对路径或 `http(s)` 图片链接
   - `path2`：图片2的绝对路径或 `http(s)` 图片链接
 - **路径校验规则**：
   - 传入本地路径时：必须是绝对路径
   - 文件必须存在且为图片格式（`.jpg/.jpeg/.png/.bmp/.webp`）
   - 如果设置了 `ALLOWED_BASE_DIRS`，则路径必须落在白名单目录内

 ### `POST /predict_upload`

 - **Content-Type**：`multipart/form-data`
 - **请求体**：
   - `file1`：图片1
   - `file2`：图片2
 - **说明**：适用于远端电脑图片在本机、不在服务器磁盘的场景

 ### `POST /predict_preview`

 - **Content-Type**：`application/json`
 - **请求体**：同 `/predict`
 - **返回字段**：同 `/predict`，并额外包含：
   - `previews`：预览图（base64 dataURL）
     - `vehicle1` / `vehicle2`：车辆裁切预处理后的图
     - `head1` / `head2`：车头裁切图
     - `tail1` / `tail2`：车尾裁切图

 ### `POST /predict_upload_preview`

 - **Content-Type**：`multipart/form-data`
 - **请求体**：同 `/predict_upload`
 - **返回字段**：同 `/predict_preview`

 - **响应字段**：
   - `ok`：是否推理成功（`case_type != "abnormal"`）
   - `case_type`：分类结果（见下）
   - `head_prob`：车头相似度（float）
   - `tail_prob`：车尾相似度（float）
   - `error`：异常信息（可选）

 ## 分类规则（`case_type`）

 - `abnormal`
   - 模型初始化失败或推理异常（如文件打不开、模型路径错误等）
 - `fake_plate`
   - `head_prob < HEAD_LOW_TH`（默认 `0.8`）
 - `change_trailer`
   - `head_prob > HEAD_SAME_TH`（默认 `0.3`）且 `tail_prob <= TAIL_LOW_TH`（默认 `0.3`）
 - `normal`
   - 其余情况

 ## 环境变量配置

 - `HOST`
   - 默认：`0.0.0.0`
 - `PORT`
   - 默认：`8001`
 - `HEAD_MODEL_PATH`
   - 车头 Siamese 权重路径
   - 默认（脚本内置）：`Siamese-pytorch-master/logs/head/1211/best_epoch_weights.pth`
 - `TAIL_MODEL_PATH`
   - 车尾 Siamese 权重路径
   - 默认（脚本内置）：`Siamese-pytorch-master/logs/weibu/1211/best_epoch_weights.pth`
 - `HEADTAIL_MODEL_PATH`
   - YOLO 检测模型路径（用于裁切车头/车尾）
   - 默认（脚本内置）：`D:\data2\runs\detect\train\weights\best.pt`
 - `ALLOWED_BASE_DIRS`
   - 图片路径白名单；多个目录用英文分号 `;` 分隔
   - 示例：`D:\images;D:\dataset\capture`
 - `REMOTE_FETCH_ENABLED`
   - 远程拉取开关（当 `/predict` 传入 `http(s)` 链接或本地文件缺失时）
   - 默认：开启（`1`）；关闭示例：`0/false/no/off`
 - `PREVIEW_MAX_SIZE`
   - 预览图片最大边长（用于 `/predict_preview` 与 `/predict_upload_preview` 返回的 6 图）
   - 默认：`640`
 - `HEAD_LOW_TH` / `HEAD_SAME_TH` / `TAIL_LOW_TH`
   - 分类阈值，默认分别为 `0.8 / 0.3 / 0.3`

 ## 启动方式（Windows 示例）

 - 直接启动（使用脚本默认模型路径）：
  - `python Siamese-pytorch-master\my_predict_gui_new1.py`

 - 指定端口与模型路径（PowerShell）：
  - `$env:PORT="8001"; $env:HEAD_MODEL_PATH="D:\\path\\head.pth"; $env:TAIL_MODEL_PATH="D:\\path\\tail.pth"; $env:HEADTAIL_MODEL_PATH="D:\\path\\best.pt"; python Siamese-pytorch-master\my_predict_gui_new1.py`

 ## 远端访问注意事项（局域网）

 - 远端电脑访问时不要使用 `127.0.0.1/localhost`，应使用运行服务机器的局域网 IPv4（常见为 `172.*` 或 `10.*`）。
 - 若远端浏览器一直“连接中”，优先检查：
   - Windows 防火墙是否放行入站 `TCP 8001`
   - 远端是否能连通端口：`Test-NetConnection -ComputerName <服务器IP> -Port 8001`

 ## 调用示例

 - 请求：
```json
{
  "path1": "D:\\images\\a.jpg",
  "path2": "D:\\images\\b.jpg"
}
```

 - 响应示例：
```json
{
  "ok": true,
  "case_type": "normal",
  "head_prob": 0.91,
  "tail_prob": 0.88
}



## 2026-04-13（当前版本）

### 后端服务全面升级（`my_predict_gui_new.py`）

**当前系统版本：过磅车辆智能识别系统 v5.0**

本次升级实现了完整的生产级后端服务，核心功能包括：

#### 1. 双层鉴别分类架构
- **第一层（Siamese快速筛选）**
  - 车头相似度 < 0.3：明确判定为套牌车
  - 车头相似度 > 0.8：判定为正常车头
  - 车尾相似度 < 0.3：判定为换挂车
  - 车尾相似度 > 0.8：判定为车尾一致
  
- **第二层（AI视觉大模型精细鉴别）**
  - 触发条件：相似度在 0.3~0.8 不确定区间
  - 模型：Qwen3.5:9b（可配置）
  - 功能：对车头/车尾分别进行视觉对比判断
  - 开关：`AI_SECOND_JUDGE_ENABLED`（默认开启）
  - 耗时：约 500-2000ms（异步处理，不阻塞主流程）

#### 2. 完整的记录管理系统

**数据存储结构**
- 自动保存8张图片/记录：
  - 2张原始图片（original1.jpg, original2.jpg）
  - 6张处理后图片（vehicle1/2, head1/2, tail1/2）
- 元数据存储：JSON格式，包含相似度、判定结果、时间戳等
- 存储路径：`stats_logs/images/YYYYMMDD/{record_id}/`

**记录管理API**
- 记录查询：支持按日期范围、类型筛选、分页
- 单条记录详情：完整元数据+图片访问接口
- 图片访问：`/api/record/{id}/image/{name}`
- 批量删除：支持软删除和硬删除两种模式

**数据保护机制**
- 自动清理策略：
  - 正常车辆：90天后自动清理
  - 套牌/换挂车：受保护记录不会被自动清理
  - 保护标记：可手动设置重要记录为受保护状态
- 手动删除限制：正常车辆记录不允许手动删除

#### 3. 人工复核系统

**复核功能**
- 提交复核：`POST /api/record/{id}/review`
- 撤销复核：`DELETE /api/record/{id}/review`
- 复核字段：复核类型、复核理由、复核人员、置信度
- 复核历史：支持多次复核，保留完整历史记录
- 复核统计：`/api/records/review_stats`
  - 总记录数、已复核数、复核率
  - 按类型统计（确认正确/被修正）
  - 修正流向分析（套牌→正常、正常→换挂等）

**复核页面**
- 独立页面：`/review_stats`
- 可视化展示复核统计图表

#### 4. 数据导出系统

**单条导出**
- 接口：`POST /api/record/{id}/export`
- 导出内容：图片+元数据+info.txt说明文件
- 可选图片类型：支持选择导出原始图、裁切图、部件图

**批量导出**
- 接口：`POST /api/records/batch_export`
- 分组方式：按类型(case_type)分组或无分组
- 汇总文件：自动生成 export_summary.csv 和 export_log.txt
- 图片类型预设：全部、仅原始图、仅处理后、仅车头、仅车尾等

**导出配置**
- 可用图片类型接口：`GET /api/export/image_types`
- 支持自定义导出路径

#### 5. 监控统计系统

**实时统计**
- `/stats`：当前服务状态快照
  - 服务启动时间、总请求数、成功率
  - 按端点统计（请求数、错误数、P95延迟）
  - 各类型案件统计（套牌、换挂、正常）
  
- `/stats/recent?n=200`：最近n条请求记录
- `/stats/summary?days=7`：小时级趋势数据（用于图表）

**仪表板页面**
- 独立页面：`/dashboard`
- 可视化展示系统运行状态

#### 6. Web界面（`/ui` 页面功能增强）

**现有功能**
- 路径预测：支持本地路径和HTTP图片链接
- 上传预测：支持本地上传两张图片
- 结果展示：6张预览图可视化+概率进度条
- 结果下载：JSON/CSV格式

**新增功能**
- AI判断结果显示：显示是否使用了AI二次判断
- AI判断耗时：独立显示AI判断耗时
- 记录ID返回：每次检测生成唯一记录ID

#### 7. 性能优化与稳定性

**并发控制**
- 模型初始化锁：`_INIT_LOCK`，防止并发初始化
- 推理管道锁：`_PIPELINE_LOCK`，保证单线程推理（模型线程安全）

**性能监控**
- 分阶段耗时统计：验证、打开、计算、AI判断
- P95延迟指标统计
- 自动记录每条请求的详细耗时

**路径安全**
- 绝对路径强制校验
- 白名单目录控制：`ALLOWED_BASE_DIRS`
- 远程拉取开关：`REMOTE_FETCH_ENABLED`

#### 8. 环境变量配置（新增）

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `AI_SECOND_JUDGE_ENABLED` | 启用AI二次判断 | `1` |
| `AI_JUDGE_MODEL` | AI判断模型名称 | `qwen3.5:9b` |
| `HEAD_AI_LOW_TH` | AI车头低阈值 | `0.3` |
| `HEAD_AI_HIGH_TH` | AI车头高阈值 | `0.8` |
| `TAIL_AI_LOW_TH` | AI车尾低阈值 | `0.3` |
| `TAIL_AI_HIGH_TH` | AI车尾高阈值 | `0.8` |

#### 9. API端点汇总

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 可用端点列表 |
| `/health` | GET | 健康检查 |
| `/ui` | GET | Web界面 |
| `/dashboard` | GET | 仪表板页面 |
| `/records` | GET | 记录查询页面 |
| `/review_stats` | GET | 复核统计页面 |
| `/stats` | GET | 统计快照 |
| `/stats/recent` | GET | 最近记录 |
| `/stats/summary` | GET | 小时汇总 |
| `/predict` | POST | 路径预测 |
| `/predict_preview` | POST | 路径预测+预览图 |
| `/predict_upload` | POST | 上传预测 |
| `/predict_upload_preview` | POST | 上传预测+预览图 |
| `/api/records` | GET | 查询记录列表 |
| `/api/record/{id}` | GET | 单条记录详情 |
| `/api/record/{id}/image/{name}` | GET | 获取记录图片 |
| `/api/record/{id}` | DELETE | 删除记录 |
| `/api/records/batch_delete` | POST | 批量删除 |
| `/api/record/{id}/protect` | POST | 设置保护状态 |
| `/api/record/{id}/export` | POST | 导出单条记录 |
| `/api/records/batch_export` | POST | 批量导出 |
| `/api/export/image_types` | GET | 获取图片类型列表 |
| `/api/record/{id}/review` | POST | 提交复核 |
| `/api/record/{id}/review` | DELETE | 撤销复核 |
| `/api/records/review_stats` | GET | 复核统计 |

---

### 当前项目整体进度总结

**核心功能完成度：95%**

| 模块 | 状态 | 说明 |
|------|------|------|
| 车辆检测与裁切 | ✅ 完成 | YOLOv8 + 自训权重 best.pt |
| 车头/车尾识别 | ✅ 完成 | YOLO检测 + Siamese相似度计算 |
| 套牌判定逻辑 | ✅ 完成 | 双层鉴别：Siamese + AI二次判断 |
| 后端推理服务 | ✅ 完成 | Flask + 多线程 + 并发控制 |
| Web管理界面 | ✅ 完成 | 检测界面 + 记录管理 + 仪表板 |
| 记录管理系统 | ✅ 完成 | 完整CRUD + 图片存储 + 数据保护 |
| 人工复核系统 | ✅ 完成 | 提交/撤销复核 + 复核统计 |
| 数据导出系统 | ✅ 完成 | 单条/批量导出 + CSV汇总 |
| 监控统计系统 | ✅ 完成 | 实时统计 + 趋势分析 |
| 数据清理策略 | ✅ 完成 | 自动清理 + 保护机制 |

**待完善功能（5%）**
- 数据库持久化：当前使用JSONL文件存储，可扩展至关系型数据库
- 分布式部署：当前单机多线程，可扩展至微服务架构
- 实时告警：异常案件自动通知功能

**系统访问地址**
- 本地测试：`http://127.0.0.1:8001/ui`
- 局域网访问：`http://{服务器IP}:8001/ui`
- 仪表板：`http://{服务器IP}:8001/dashboard`
- 记录管理：`http://{服务器IP}:8001/records`

---

