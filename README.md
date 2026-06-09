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
   - `ai_judge_used`：是否触发 AI 二次判断（可选）
   - `ai_head_result`：AI 对车头的复核结果，取值 `fake_plate/normal`（可选）
   - `ai_tail_result`：AI 对车尾的复核结果，取值 `change_trailer/normal`（可选）
   - `ai_ms`：AI 二次判断耗时，毫秒（可选）
   - `diff_desc`：AI 复核说明或差异描述；一阶段异常但 AI 改判正常时也可能返回（可选）
   - `diff_analyzed_part`：AI 分析的部位，如 `head`、`tail`、`head+tail`（可选）
   - `ai_diff_ms`：AI 差异分析耗时，毫秒（可选）
   - `error`：异常信息（可选）

 ## 分类规则（`case_type`）

 - `abnormal`
   - 输入校验失败、图片打开失败、模型初始化失败或推理异常时返回。
 - `fake_plate`
   - 一阶段规则：`head_prob <= head_threshold` 时，进入车头 AI 二次复核。
   - 最终返回条件：车头 AI 复核结果为 `fake_plate`，或 AI 无法有效判断时回退到一阶段 `fake_plate`。
 - `change_trailer`
   - 一阶段规则：`head_prob > head_threshold` 且 `tail_prob <= tail_threshold` 时，进入车尾 AI 二次复核。
   - 最终返回条件：车尾 AI 复核结果为 `change_trailer`，或 AI 无法有效判断时回退到一阶段 `change_trailer`。
 - `normal`
   - `head_prob > head_threshold` 且 `tail_prob > tail_threshold` 时，直接判定为 `normal`。
   - 其余进入 AI 复核的样本，如果 AI 最终判为正常，也返回 `normal`。

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
 - `HEAD_THRESHOLD_DEFAULT` / `TAIL_THRESHOLD_DEFAULT`
   - 一阶段直通阈值默认值，默认均为 `0.8`
 - `AI_SECOND_JUDGE_ENABLED`
   - 是否启用 AI 二次判断，默认：开启（`1`）
 - `AI_JUDGE_MODEL`
   - AI 判断模型名称，默认：`qwen3.5:9b`

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



## 2026-04-13（当前服务基线）

- **[升级] 当前 Flask 服务主入口切换为 `my_predict_gui_new.py`**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **说明**：当前实际维护的服务入口为 `my_predict_gui_new.py`，接口、日志、记录管理、导出、复核、统计页面均以该文件为准。

- **[保留] 双阶段判定主链路**
  - **第一阶段**：先计算 `head_prob`、`tail_prob`
  - **第二阶段**：非“双高”样本进入 AI 二次判断
  - **正常直通规则**：`head_prob > head_threshold` 且 `tail_prob > tail_threshold` 时直接返回 `normal`

- **[提供] 完整服务能力**
  - 接口：`/predict`、`/predict_preview`、`/predict_upload`、`/predict_upload_preview`
  - 页面：`/ui`、`/dashboard`、`/records`、`/review_stats`
  - 管理能力：日志、图片留档、导出、人工复核、统计汇总

## 当前服务说明（以 `my_predict_gui_new.py` 为准）

### 服务入口

- 文件：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
- 默认地址：`http://127.0.0.1:8001`

### 当前接口总览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 查看可用端点 |
| `/health` | GET | 健康检查 |
| `/ui` | GET | 检测前端页面 |
| `/dashboard` | GET | 统计仪表板 |
| `/records` | GET | 记录查询页面 |
| `/review_stats` | GET | 复核统计页面 |
| `/predict` | POST | 路径预测 |
| `/predict_preview` | POST | 路径预测并返回预览图 |
| `/predict_upload` | POST | 上传图片预测 |
| `/predict_upload_preview` | POST | 上传图片预测并返回预览图 |
| `/stats` | GET | 服务统计快照 |
| `/stats/recent` | GET | 最近请求列表 |
| `/stats/summary` | GET | 小时级汇总 |
| `/stats/reset` | POST | 重置内存统计 |
| `/api/records` | GET | 查询记录列表 |
| `/api/record/{id}` | GET | 获取记录详情 |
| `/api/record/{id}/image/{name}` | GET | 获取记录图片 |
| `/api/record/{id}` | DELETE | 删除记录 |
| `/api/records/batch_delete` | POST | 批量删除 |
| `/api/record/{id}/protect` | POST | 设置保护状态 |
| `/api/record/{id}/export` | POST | 导出单条记录 |
| `/api/records/batch_export` | POST | 批量导出记录 |
| `/api/export/image_types` | GET | 获取可导出图片类型 |
| `/api/record/{id}/review` | POST | 提交复核 |
| `/api/record/{id}/review` | DELETE | 撤销复核 |
| `/api/records/review_stats` | GET | 获取复核统计 |
| `/thresholds` | GET/POST | 获取或更新阈值 |

### 当前判定逻辑

#### 1. 两地址模式

- 请求只传 `path1/path2`，或上传只传 `file1/file2`
- 完全沿用原方案：
  - 先算 `head_prob`、`tail_prob`
  - 若双高则直接 `normal`
  - 否则根据一阶段分流进入车头或车尾 AI 二次判断
  - 最终输出 `normal / fake_plate / change_trailer`

#### 2. 四地址模式

- 路径模式支持额外传入 `path3/path4`
- 上传模式支持额外传入 `file3/file4`
- 前两张仍是主判定图，后两张仅用于“尾部原图二次确认”
- 当前真实顺序为：
  1. 先完整执行原方案，得到 `stage1_case_type`
  2. 仅当原方案结果为 `change_trailer` 时
  3. 再调用 `qwen_vl/predict_ai_shijiao2.py` 做尾部原图确认
  4. 若尾部原图确认返回“正常”，最终结果改判为 `normal`
  5. 若尾部原图确认返回“换挂”，最终保持 `change_trailer`

#### 3. 尾部原图确认规则

- 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai_shijiao2.py`
- 类：`TailVehicleCheck`
- 核心规则：
  - 只看两张原图中央车辆
  - 优先比对车号、车身编号、放大号
  - 编号一致，直接判 `正常`
  - 编号不一致、单边可见单边缺失、被遮挡、无法互相确认，直接判 `换挂`
  - 只有在编号无法稳定确认且不能直接下结论时，才补看尾门、栏杆、尾灯、车厢结构等特征

## 2026-04-28

- **[调整] `my_predict_gui_new.py` 一级分流规则**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：将车辆判定逻辑调整为“车头相似度和车尾相似度均大于 `0.8` 时直接判定为 `normal`”。
  - **新规则**：
    - `head_prob > 0.8` 且 `tail_prob > 0.8`：直接判定为 `normal`
    - `head_prob > 0.8` 且 `tail_prob <= 0.8`：进入车尾二级判断，确认是否为 `change_trailer`
    - `head_prob <= 0.8`：进入车头二级判断，确认是否为 `fake_plate`
  - **说明**：本次调整取消了一级阶段“低分直接判异常”的分支，改为仅对“双高”样本直接放行，其余样本按疑似类型进入 AI 二级判断。

## 2026-05-05

- **[修复] `my_predict_gui_new.py` AI 无法判断时回退一阶段判定结果**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：在 `_classify_with_ai_second_judge()` 中新增 `stage1_case_type`，显式保存一阶段判定结果，并在 AI 不可用时直接回退到一阶段结论。
  - **效果**：当 AI 因图片质量差、输出异常或服务不可用而无法继续判断时，最终 `case_type` 保持与一阶段 Siamese 结果一致。

- **[增强] AI 判定值有效性校验**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：对车头 AI 只接受 `fake_plate/normal`，对车尾 AI 只接受 `change_trailer/normal`；空值或非预期字符串统一视为无效结果。
  - **效果**：避免异常 AI 输出直接污染最终判定，减少误判和结果漂移。

- **[增强] AI 二次判断返回结构增加理由文本**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：车头与车尾复核从 `check_head/check_tail` 调整为 `check_head_with_reason/check_tail_with_reason`，除标签外同步接收 `reason` 字段。
  - **效果**：接口和页面可展示更明确的 AI 复核依据，便于人工核查和业务解释。

- **[增强] AI 返回无效值或裁切保存失败时统一按一阶段结果兜底**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：新增 `ai_invalid` 标记；当 AI 返回空值、非预期标签，或临时裁切图保存失败时，统一回退到 `stage1_case_type`。
  - **效果**：保证最终结果至少与一阶段 Siamese 判断一致，提升异常场景稳定性。

- **[增强] 一阶段异常但 AI 改判正常时保留说明文本**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：将 `diff_desc` 的生成条件从“最终结果为异常”调整为“一阶段曾经判定为异常”；当 AI 复核后最终改判 `normal` 时，保留 `AI复核后判为正常` 或对应 `reason`。
  - **效果**：页面与接口在“异常改判正常”场景下仍保留复核说明，方便后续复盘。

## 2026-05-08

- **[新增] 四地址模式下的尾部原图二次确认方案**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：路径预测接口在保留 `path1/path2` 的基础上，新增可选 `path3/path4`。其中：
    - 仅传 `path1/path2` 时，仍沿用原有两地址方案；
    - 同时传入 `path3/path4` 时，进入四地址模式，后两张图仅用于尾部原图复核。
  - **效果**：兼容旧调用方式，不影响现有两地址业务，同时为换挂复核提供额外视角。

- **[调整] 四地址模式判定顺序改为“原方案先判，原图方案后确认”**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：
    - 先完整执行原有 Siamese + 车头/车尾 AI 二次判断流程，得到原方案最终结果；
    - 仅当原方案最终结果为 `change_trailer` 时，才调用 `qwen_vl/predict_ai_shijiao2.py` 中的尾部原图方案，对 `path3/path4` 进行进一步确认；
    - 若尾部原图复核结果为“正常”，则将最终结果从 `change_trailer` 改判为 `normal`；
    - 若尾部原图复核结果为“换挂”，则保持 `change_trailer` 不变。
  - **效果**：新方案不再提前接管尾部分支，而是作为换挂确认器使用，更符合现有业务流程。

- **[新增] 尾部原图 AI 复核脚本**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai_shijiao2.py`
  - **变更内容**：新增 `TailVehicleCheck`，直接对两张原图中的中央车辆尾部进行比对，优先比较车号/车身编号/放大号，在无法确认时再比对尾门、栏杆、尾灯、车厢结构等稳定特征。
  - **输出结构**：返回结构化字段，包括 `label`、`reason`、`plate_or_number_consistency`、`structure_consistency`。
  - **效果**：为四地址模式中的换挂确认提供更明确的尾部业务规则。

- **[增强] 接口返回与日志记录增加“原方案结果/二次确认结果”链路信息**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **新增字段**：
    - `input_mode`
    - `tail_ai_mode`
    - `stage1_case_type`
    - `tail_second_check_used`
    - `tail_second_check_result`
    - `tail_second_check_reason`
  - **效果**：接口响应、`stats_logs/*.jsonl`、图片目录下的 `meta.json`、导出 `info.txt` 均可追踪“原方案先判什么、尾部原图是否复核、复核后是否改判”。

- **[增强] `/ui` 页面适配两地址/四地址业务模式**
  - **变更文件**：
    - `data_chuli/demo/demo/Siamese-pytorch-master/templates/ui.html`
    - `data_chuli/demo/demo/Siamese-pytorch-master/static/ui.js`
    - `data_chuli/demo/demo/Siamese-pytorch-master/static/ui.css`
  - **变更内容**：
    - 路径预测页面新增 `path3/path4` 输入框；
    - 前端提交时对 `path3/path4` 做成对校验；
    - 结果区新增 `input_mode`、`tail_ai_mode` 展示；
    - 下载 JSON/CSV 时同步写入四地址相关字段。
  - **效果**：前端与后端业务保持一致，便于现场联调和人工确认本次调用走的是哪条判定链路。

### 2026-05-08 接口使用示例

#### 1. 两地址模式：完全沿用原方案

- **适用场景**：
  - 只有两张主图；
  - 不启用尾部原图二次确认；
  - 行为与历史版本保持一致。

- **请求示例**

```json
{
  "path1": "D:\\images\\car_a_1.jpg",
  "path2": "D:\\images\\car_a_2.jpg"
}
```

- **调用接口**
  - `POST /predict`
  - `POST /predict_preview`
  - `POST /predict_upload`
  - `POST /predict_upload_preview`

- **返回示例**

```json
{
  "ok": true,
  "case_type": "normal",
  "head_prob": 0.9132,
  "tail_prob": 0.8741,
  "input_mode": "2_paths",
  "tail_ai_mode": "none",
  "stage1_case_type": "normal",
  "tail_second_check_used": false,
  "tail_second_check_result": null,
  "tail_second_check_reason": null,
  "diff_desc": null,
  "diff_analyzed_part": null
}
```

- **说明**
  - `input_mode = "2_paths"`：表示本次只使用前两张图；
  - `tail_second_check_used = false`：表示没有启用第二种尾部原图确认方案；
  - 其余判定逻辑与原方案一致。

#### 2. 四地址模式：原方案先判，尾部原图后确认

- **适用场景**：
  - `path1/path2` 为主图；
  - `path3/path4` 为额外尾部原图；
  - 仅当原方案先判为 `change_trailer` 时，才触发尾部原图二次确认。

- **请求示例**

```json
{
  "path1": "D:\\images\\main_view_1.jpg",
  "path2": "D:\\images\\main_view_2.jpg",
  "path3": "D:\\images\\tail_view_1.jpg",
  "path4": "D:\\images\\tail_view_2.jpg"
}
```

- **调用接口**
  - `POST /predict`
  - `POST /predict_preview`
  - `POST /predict_upload`
  - `POST /predict_upload_preview`

- **返回示例 A：原方案先判换挂，尾部原图确认后仍为换挂**

```json
{
  "ok": true,
  "case_type": "change_trailer",
  "head_prob": 0.8924,
  "tail_prob": 0.4217,
  "input_mode": "4_paths",
  "tail_ai_mode": "original_tail_confirm",
  "stage1_case_type": "change_trailer",
  "tail_second_check_used": true,
  "tail_second_check_result": "change_trailer",
  "tail_second_check_reason": "中央车辆尾部编号无法一致确认，且尾灯与栏杆结构存在明显不一致。",
  "ai_tail_result": "change_trailer",
  "diff_desc": "中央车辆尾部编号无法一致确认，且尾灯与栏杆结构存在明显不一致。",
  "diff_analyzed_part": "tail",
  "ai_diff_ms": 0.0
}
```

- **返回示例 B：原方案先判换挂，尾部原图确认后改判正常**

```json
{
  "ok": true,
  "case_type": "normal",
  "head_prob": 0.9018,
  "tail_prob": 0.4675,
  "input_mode": "4_paths",
  "tail_ai_mode": "original_tail_confirm",
  "stage1_case_type": "change_trailer",
  "tail_second_check_used": true,
  "tail_second_check_result": "normal",
  "tail_second_check_reason": "中央车辆尾部放大号一致，结构特征未发现明显差异。",
  "ai_tail_result": "normal",
  "diff_desc": null,
  "diff_analyzed_part": null,
  "ai_diff_ms": 0.0
}
```

#### 3. 上传模式补充说明

- `POST /predict_upload`
  - 必传：`file1`、`file2`
  - 可选：`file3`、`file4`
  - 规则：`file3/file4` 必须成对出现

- `POST /predict_upload_preview`
  - 规则与 `/predict_upload` 一致
  - 额外返回 `previews`
    - `vehicle1`、`vehicle2`
    - `head1`、`head2`
    - `tail1`、`tail2`

#### 4. 当前返回字段说明

- `ok`
  - 是否成功完成本次判定
  - `true` 表示接口执行成功并得到了业务结论
  - `false` 一般表示参数错误、图片打开失败或内部异常

- `case_type`
  - 最终业务结论
  - `normal`：正常
  - `fake_plate`：套牌
  - `change_trailer`：换挂
  - `abnormal`：异常请求或异常处理结果

- `head_prob`
  - 前两张主图的车头相似度
  - 值越高，表示车头越像同一辆车

- `tail_prob`
  - 前两张主图的车尾相似度
  - 值越高，表示车尾越像同一辆车

- `input_mode`
  - `2_paths`：只使用两张输入图
  - `4_paths`：使用四张输入图，后两张用于尾部原图确认

- `ai_judge_used`
  - 是否触发过原方案中的 AI 二次判断
  - 这是“原方案 AI”是否参与，不等同于尾部原图确认是否触发

- `ai_head_result`
  - 原方案中车头 AI 的复核结果
  - 常见值：`fake_plate`、`normal`
  - 未触发时为 `null`

- `ai_tail_result`
  - 原方案中车尾 AI 的复核结果，或四地址模式下尾部原图确认后的最终尾部结论
  - 常见值：`change_trailer`、`normal`
  - 未触发时为 `null`

- `ai_ms`
  - 原方案 AI 二次判断耗时，单位毫秒
  - 只统计车头/车尾旧 AI 复核阶段

- `tail_ai_mode`
  - `none`：未走尾部 AI
  - `legacy_crop`：走了原有“裁切尾图 + 旧 AI”方案
  - `original_tail_confirm`：在四地址模式下又走了“尾部原图确认”方案

- `stage1_case_type`
  - 原方案完整执行后的结果
  - 这是四地址模式里非常关键的字段
  - 如果最终被尾部原图改判为 `normal`，这里仍可能保留 `change_trailer`

- `tail_second_check_used`
  - 是否触发了第二种方法，也就是尾部原图确认
  - `true` 表示四地址模式下已经执行
  - `false` 表示未执行

- `tail_second_check_result`
  - 第二种方法本身给出的结论
  - 常见值：`change_trailer`、`normal`
  - 未触发时为 `null`

- `tail_second_check_reason`
  - 第二种方法给出的中文说明
  - 主要用于人工核查“为什么判换挂”或“为什么改判正常”

 - `diff_desc`
  - 一句话差异总结
  - 当最终结论为 `fake_plate` 或 `change_trailer` 时，通常返回具体差异说明
  - 当最终结论为 `normal` 时，当前代码统一返回 `null`

- `diff_analyzed_part`
  - 差异分析针对的部位
  - 常见值：`head`、`tail`、`head+tail`
  - 正常时通常为 `null`

- `ai_diff_ms`
  - 差异分析耗时，单位毫秒
  - 若是尾部原图确认直接给出结论，当前代码一般返回 `0.0`

- `record_id`
  - 本次请求生成的唯一记录 ID
  - 可用于后续查询记录、查看图片、导出、人工复核

- `error`
  - 仅在请求失败或部分处理异常时返回

#### 5. 用户示例返回逐字段解读

针对如下示例：

```json
{
  "ai_diff_ms": 0.0,
  "ai_head_result": null,
  "ai_judge_used": true,
  "ai_ms": 28958.9,
  "ai_tail_result": "change_trailer",
  "case_type": "change_trailer",
  "diff_analyzed_part": "tail",
  "diff_desc": "两张图中车辆的车牌号（桂B·A4886与桂B·W0143）不一致，且车头品牌（CENLYON与东风柳汽）及车身标识均不同，确认为不同车辆。",
  "head_prob": 0.9989994168281555,
  "input_mode": "4_paths",
  "ok": true,
  "record_id": "20260508_115644_4662c13c",
  "stage1_case_type": "change_trailer",
  "tail_ai_mode": "original_tail_confirm",
  "tail_prob": 0.007520087528973818,
  "tail_second_check_reason": "两张图中车辆的车牌号（桂B·A4886与桂B·W0143）不一致，。",
  "tail_second_check_result": "change_trailer",
  "tail_second_check_used": true
}
```

- `head_prob = 0.9989`
  - 前两张主图车头非常相似，所以这次不是车头问题

- `tail_prob = 0.0075`
  - 前两张主图车尾相似度极低，因此原方案会怀疑换挂

- `stage1_case_type = "change_trailer"`
  - 原方案完整执行后，先给出的结论就是换挂

- `input_mode = "4_paths"`
  - 这次不是传统两图，而是四图模式

- `tail_second_check_used = true`
  - 因为原方案先判成了换挂，所以继续触发了尾部原图二次确认

- `tail_ai_mode = "original_tail_confirm"`
  - 表示最后采用的是新增的“尾部原图确认”链路

- `tail_second_check_result = "change_trailer"`
  - 第二种方法复核后，仍然判定为换挂

- `tail_second_check_reason`
  - 第二种方法给出的核心依据
  - 本例直接指出车牌号、品牌、车身标识不一致

- `case_type = "change_trailer"`
  - 因为二次确认没有推翻原结论，所以最终结果仍然是换挂

- `diff_desc`
  - 给前端和接口使用的一句话差异总结
  - 本例返回的是“哪里不同、为什么判换挂”

- `diff_analyzed_part = "tail"`
  - 表示这条差异总结是从车尾链路得出的

- `ai_judge_used = true`
  - 原方案里确实调用了 AI 二次判断

- `ai_head_result = null`
  - 这次没有走车头 AI 复核

- `ai_tail_result = "change_trailer"`
  - 当前最终尾部 AI 结论为换挂

- `ai_ms = 28958.9`
  - 原方案 AI 二次判断耗时约 28.96 秒

- `ai_diff_ms = 0.0`
  - 这次差异结论直接来自尾部原图确认，没有再单独跑额外差异分析耗时

- `record_id`
  - 可用于回查本次留档记录、图片与导出结果

#### 6. 日志与留档说明

- 日志目录：`data_chuli/demo/demo/Siamese-pytorch-master/stats_logs/`
- 每日日志：`stats_YYYYMMDD.jsonl`
- 图片目录：`stats_logs/images/YYYYMMDD/{record_id}/`
- 记录元数据会同步保存：
  - `input_mode`
  - `tail_ai_mode`
  - `stage1_case_type`
  - `tail_second_check_used`
  - `tail_second_check_result`
  - `tail_second_check_reason`
  - `diff_desc`
  - `diff_analyzed_part`
  - `ai_diff_ms`
现有判别逻辑是：系统先用前两张主图做车辆裁切、车头车尾部位裁切，并计算 head_prob 和 tail_prob；如果车头和车尾相似度都高于阈值，就直接判定为 normal，否则进入原有 AI 二次判断，其中车头分支用于判断是否 fake_plate，车尾分支用于判断是否 change_trailer。如果本次是四地址模式，并且原方案最终先判成了 change_trailer，系统才会再使用后两张尾部原图做一次尾部确认：优先比对中央车辆的车号、车身编号、放大号，无法确认时再比对尾门、栏杆、尾灯、车厢结构等特征；如果二次确认仍判换挂，则最终结果保持 change_trailer，如果二次确认判为正常，则最终改判为 normal。

  主流程

读取主视角两张图 img1/img2
如果有，再读取尾部视角两张图 img3/img4
主视角做整车裁切
从整车里裁出：
head1/head2
tail1/tail2
Siamese 计算：
head_prob
tail_prob
第一层分类

如果 head_prob is None 或 tail_prob is None
返回 abnormal

如果 head_prob < head_threshold
一阶段结果记为 fake_plate

如果 head_prob >= head_threshold 且 tail_prob <= tail_threshold
一阶段结果记为 change_trailer

否则
一阶段结果记为 normal

OCR 预检

先对 head1/head2 做 OCR
如果 OCR 没拿到有效文本
继续后续 AI 判断，不直接改结果
如果 OCR 文本一致
继续后续 AI 判断，不直接改结果
如果 OCR 文本不一致：
如果 head_prob > 0.8
进入“强制车头 AI 复核”
否则
直接返回 fake_plate
AI 总入口

如果 head_prob < 0.1
直接返回 fake_plate
这里会跳过所有 AI

如果不是 OCR 强制复核，并且：

head_prob > head_threshold
tail_prob > tail_threshold
直接返回 normal
否则进入 AI 二次判断

车头 AI

如果满足下面任一条件，车头需要 AI：
head_prob <= head_threshold
OCR 不一致且 head_prob > 0.8，触发了强制复核
车头 AI 输入：
head1/head2
车头 AI 输出：
fake_plate
normal
其他无效结果
如果车头 AI 输出无效
回退一阶段结果
车尾 AI

如果 tail_prob > tail_threshold
车尾不需要 AI

如果 tail_prob <= tail_threshold
车尾需要 AI

如果提供了 img3/img4
先准备 3/4 视角尾部裁切图：

tail_view_crop3
tail_view_crop4
先跑 3/4 视角尾部 AI

3/4 视角尾部 AI

先检查两张图是否都有足够尾部信息
如果尾部信息不足
返回 无法判断
如果两张图尾部编号明确一致
返回 正常
如果两张图尾部编号明确不一致
返回 换挂
如果编号无法确认，但尾部结构可比较
再看结构：
结构明显不一致 -> 换挂
结构无明显不一致 -> 正常
3/4 视角结果分流

如果 3/4 视角 AI 返回 正常
车尾判正常，结束车尾判断

如果 3/4 视角 AI 返回 换挂
车尾判换挂，结束车尾判断

如果 3/4 视角 AI 返回 无法判断
回退到主视角车尾裁切图 AI

主视角车尾 AI 回退

输入：
tail1/tail2
AI 输出：
change_trailer
normal
无效
如果主视角车尾 AI 也无效
回退一阶段结果
最终合成

如果 AI 过程中关键结果无效
最终结果 = 一阶段结果

否则如果车头 verdict = fake_plate
最终结果 = fake_plate

否则如果车尾 verdict = different
最终结果 = change_trailer

否则
最终结果 = normal

你现在可以把它理解成一句最短版

先用 Siamese 做头尾相似度初筛
车头先做 OCR
OCR 判套牌但车头又很像时，强制加一次头部 AI 复核
车尾低相似度时，优先看 3/4 视角尾部 AI
3/4 视角信息不足，再回退主视角车尾裁切 AI
最后综合成 normal / fake_plate / change_trailer

# 修改记录

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


## 2026-05-11

- **[调整] 车头 OCR 预处理链路重构为“先提字、后比对”**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：
    - 前置 OCR 仅针对 1/2 视角车辆检测后的车头裁切图 `h1/h2` 执行；
    - 主流程分别调用两次 `MaxBoxOCR.get_max_text()` 提取两张车头图的最大有效文字；
    - 再调用 `compare_texts()` 比较两边 OCR 文本，不再直接用整段结构体字符串做匹配。

- **[新增] 车头 OCR 置信度与面积双门槛**
  - **变更文件**：
    - `data_chuli/demo/demo/Siamese-pytorch-master/paddle_ocr/ocr_detect.py`
    - `data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：
    - `get_max_text()` 默认只保留 `score >= 0.6` 的 OCR 候选；
    - 在候选中选取面积最大的文字；
    - 主流程新增 `HEAD_OCR_MIN_AREA`，默认值 `20000`；
    - 仅当 `area > 20000` 时才认为该 OCR 结果有效，否则按“无有效文字”处理。

- **[调整] 车头 OCR 文本比对规则改为字符命中策略**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/paddle_ocr/ocr_detect.py`
  - **变更内容**：
    - 长文本要求存在“连续两个字符一致”才允许放行；
    - 当短文本长度不超过 2 个字符时，只要存在 1 个字符一致即可放行；
    - 若两边文本完全一致、标准化后一致、易混字符归一后相同，或数字部分一致，也允许放行；
    - 否则判定为 OCR 不一致。

- **[调整] 空 OCR 结果不再直接拦截为套牌**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：
    - 若两张车头裁切图都未识别到有效文字，则前置 OCR 不直接判 `fake_plate`；
    - 这类样本按 “OCR 无法提供有效结论” 处理，继续进入后续判别流程。

- **[增强] 前置 OCR 控制台日志**
  - **变更文件**：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - **变更内容**：新增终端日志，输出 `text1/text2`、`area1/area2`、`score1/score2`、`match`、`similarity`、`reason`，便于现场排查 OCR 预处理结果。
  
 ## 2026-05-13

- **[前端调整] 记录详情页隐藏“备注”展示**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/templates/records.html`
  - 变更：记录详情弹窗“基本信息”区域不再显示 `备注` 字段，避免无关键值占用页面空间。

- **[前端调整] 记录详情页隐藏“阶段耗时”展示**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/templates/records.html`
  - 变更：记录详情弹窗“判定链路”区域不再显示 `stage_ms` 的 JSON 明细，仅保留总耗时、AI 判断耗时、差异分析耗时等汇总信息。

- **[定位] 车头 OCR 不一致且高相似度时的 AI 复核输入**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - 结论：车头 OCR 预比对与车头 AI 复核默认都使用主视角整车裁切后再裁出的车头图（`h1/h2`）；若车头部位检测失败，则会回退为整车裁切图。

- **[增强] 车头 AI 复核提示词**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 变更：补充车头 AI 复核规则，明确要求忽略环境光、反光、阴影、污渍、轻微角度变化等干扰；加强对车头/车门文字区域、引擎盖装饰、品牌标识差异的关注。
  - 说明：由于原文件中旧版 `_build_head_prompt` 段落存在编码显示问题，本次通过在后文追加同名函数的方式覆盖旧实现；运行时以后定义版本为准。

## 2026-05-16

- **[前端修复] 预测页差异卡片优先显示最终差异总结**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/static/ui.js`
  - 变更：预测页右侧差异卡片改为优先读取 `final_diff_summary`，无值时再回退 `diff_desc`。
  - 效果：避免将 OCR 复核触发说明误当作最终异常结论展示。

- **[前端修复] 记录页主视角尾部 AI 结果展示受真实触发开关控制**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/templates/records.html`
  - 变更：记录详情页“主视角尾部AI结果”仅在 `main_tail_ai_used=true` 时显示结果，否则显示 `-`。
  - 效果：避免未触发主视角尾部 AI 时仍误显示 `change_trailer/normal`。

- **[后端修复] 3/4 视角尾部 AI 不再复用主视角尾部结果字段**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - 变更：
    - `tail_second_check_*` 明确仅表示 3/4 视角尾部 AI 优先判定结果；
    - `ai_tail_*` 明确仅表示主视角车尾裁切图 AI 结果；
    - 3/4 视角尾部 AI 返回 `正常/换挂` 时，不再写入 `ai_tail_result/ai_tail_reason`。
  - 效果：彻底拆开两条尾部 AI 链路，避免字段语义污染。

- **[文案调整] 记录详情页 AI 字段名称与理由标题对齐业务口径**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/templates/records.html`
  - 变更：
    - “头部AI结果”改为“头部视角车头AI结果”
    - “3/4尾部AI结果”改为“尾部视角车尾AI结果”
    - “主视角尾部AI结果”改为“头部视角AI结果”
    - 理由区标题同步调整
  - 说明：仅修改前端展示名称，不改后端变量名与返回字段。

- **[前端优化] 记录详情页头部视角车头/车尾裁切图改为完整显示**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/templates/records.html`
  - 变更：仅对 `head1/head2/tail1/tail2` 这 4 张裁切图增加 `contain` 展示样式，其余图片仍保持原有 `cover`。
  - 效果：头部视角车头图、车尾图在记录详情页中不再被固定比例裁掉，便于人工复核。

## 2026-05-18

- **[后端优化] 车头 OCR 增加工业相机叠字与时间模板过滤**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/paddle_ocr/ocr_detect.py`
  - 变更：
    - 新增“车头抓拍 / 车型抓拍车头 / 抓拍车头”等监控叠字黑名单；
    - 新增时间日期模板词过滤，如 `月 / 日 / 星期 / HH:MM(:SS)`；
    - 兼容裁剪后只剩半截叠字的情况，长度达到 3 个字的模板片段也会被过滤；
    - 过滤逻辑同时作用于 `get_max_text()` 候选选择和 `compare_texts()` 比对入口。
  - 效果：避免工业相机角标文字参与车头 OCR 一致性判断，减少因裁剪差异导致的误判套牌。

- **[提示词重构] 车头视角 AI 提示词按模块重写并压缩重复规则**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 变更：将原先平铺规则重组为“总原则 / 高优先级观察项 / 低优先级或排除项 / 特殊判读规则 / 思考顺序 / 输出要求”。
  - 效果：在不改变业务边界的前提下，提升提示词层次和稳定性，降低模型对重复规则的注意力分散。

- **[提示词优化] 车头视角 AI 明确固定标识、导流罩字样与后视镜总成属于有效差异**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 变更：
    - 明确引擎盖固定标识、导流罩长期喷涂文字、车门固定编号区、后视镜总成配色与造型属于稳定标识或主体部件；
    - 不再默认把这类差异降级为“普通装饰细节”。
  - 效果：修正 `WRC` 标识、导流罩文字、后视镜差异被模型误忽略的问题。

- **[提示词优化] 车头视角 AI 排除货物编号牌与车牌打码黑块干扰**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 变更：
    - 明确橙色/黄色纯数字编号牌、危险品或货物标识牌不属于车辆身份标识；
    - 明确程序打在真正车牌区域上的黑色矩形框只是预处理结果，不属于车辆结构或稳定标识。
  - 效果：避免模型将货物编号牌数字差异或打码黑块差异误判为套牌依据。

- **[提示词优化] 尾部视角车尾 AI 拆分“挂车身份编号”与“货物标识代码”**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai_shijiao2.py`
  - 变更：
    - 明确挂车号牌、放大号、车架号等才属于强身份信息；
    - 明确危险品/货物标识代码如两行纯数字编码，不属于挂车身份编号；
    - 货物标识代码不同不能单独作为换挂依据。
  - 效果：避免把 `60 2874 / 33 1114` 这类货物标识误当成换挂证据。

- **[提示词优化] 尾部视角车尾 AI 降低小车牌、遮挡车牌和颜色差异的误判权重**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai_shijiao2.py`
  - 变更：
    - 明确车牌区域过小、被车体遮挡、过暗、过曝、反光、只能猜字符时，编号证据一律视为不可靠；
    - 明确编号不可靠时优先转结构比对，结构也不可靠时再回退主视角车尾 AI；
    - 明确积灰、泥污、锈蚀、掉漆、补漆会改变尾门和保险杠表观颜色，`红/灰/深/浅` 不能单独作为换挂依据。
  - 效果：降低尾部视角中因小车牌误读、颜色表观变化导致的换挂误判。

## 2026-05-20

- **[后端调整] 车头 AI 触发链路改为“先看相似度，再用 OCR 兜底触发”**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - 变更：
    - 保留车头 OCR 预检，但不再因为 `ocr_match=false` 直接终判套牌；
    - 车头 AI 触发条件统一改为：`head_prob <= head_threshold`，或 `head_prob > head_threshold` 且车头 OCR 不一致；
    - 只有 `head_prob > head_threshold` 且 OCR 一致时，车头才不进入 AI。
  - 效果：减少“OCR 一次误识别直接套牌”的硬拦截，让车头 AI 真正承担复核职责。

- **[后端调整] 移除“车头相似度低于 0.1 直接套牌并跳过所有 AI”短路链路**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - 变更：
    - 删除 `head_prob < DIRECT_FAKE_PLATE_HEAD_THRESHOLD` 时直接返回 `fake_plate` 的逻辑；
    - 同步移除最终差异摘要里“车头相似度过低，直接判定为套牌”的旧文案分支。
  - 效果：避免极低相似度样本被过早终判，减少这条短路链路带来的误检。

- **[提示词优化] 车头视角 AI 收紧导流罩/遮阳板文字的有效证据条件**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 变更：明确导流罩、引擎盖顶部遮阳板、车头文字区域、喷涂标识区域，只有在两张图该区域都清晰可见，且未被强反光、过曝、发白、眩光、污渍或阴影遮盖时，才可依据文字内容差异判定 `fake_plate`。
  - 效果：降低“白字一边清晰、一边被反光洗掉”这类样本被误判套牌的概率。

- **[提示词优化] 车头视角 AI 明确套牌依据只能从车体本身寻找**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 变更：
    - 明确过磅自助机、建筑物、背景牌子、地磅设备、路面设施等非车辆对象，不能拿来与另一张图中的车头做结构差异比较；
    - 将“其中一张图没有清晰车头主体，或主要拍到非车辆对象”统一归入“输入图片质量太差”的情况。
  - 效果：避免模型拿场景设备去和车头做 `fake_plate` 比较，减少明显脏样本的误判。

- **[提示词与回退口径对齐] 车头 AI 仅输出 `fake_plate/normal`，图片质量太差时按相似度阈值给出解释性结论**
  - 文件：
    - `data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
    - `data_chuli/demo/demo/Siamese-pytorch-master/my_predict_gui_new.py`
  - 变更：
    - 车头 AI 提示词不再要求输出 `unknown`；
    - 当输入图片质量太差、长时间无法稳定判断或无法形成可靠车头结论时，统一使用解释性兜底文案：
      - `输入图片质量太差，AI无法判断，车头相似度低于或等于阈值，判断为套牌`
      - `输入图片质量太差，AI无法判断，车头相似度大于阈值，判断为正常`
    - 最后一行仍只输出 `fake_plate` 或 `normal`。
  - 效果：统一车头 AI 无法稳定判别时的业务口径，避免提示词里暴露“给定兜底结论”这类内部措辞。


## 2026-05-25

- **[后端修复] 车头 AI 判定结果提取逻辑改进，避免否定句式误判**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/qwen_vl/predict_ai.py`
  - 问题：当 AI 返回文本同时包含多个关键词时（如："这两张图片是 **normal**（正常），并非 **fake_plate**"），原有的简单关键词匹配会按列表顺序先匹配到 `fake_plate`，导致误判。
  - 变更：
    - 优先从最后几行提取结论（AI 通常在最后输出标签）
    - 识别否定句式（"并非 xxx"、"不是 xxx"），排除被否定的关键词
    - 优先匹配肯定句式（"是 xxx"、"判定为 xxx"、"属于 xxx"）
    - 简单关键词匹配作为兜底策略
  - 效果：修复了 AI 明确判定为 `normal` 但系统最终误判为 `fake_plate` 的问题，提升判定准确性。

- **[工具] 新增启动脚本 `启动程序.bat`**
  - 文件：`data_chuli/demo/demo/Siamese-pytorch-master/启动程序.bat`
  - 功能：自动激活 test2 环境并启动主程序，无需手动输入命令
  - 使用：双击运行即可
  
## 2026-05-29

### 尾部视角（3/4 原图，`predict_ai_shijiao2.py`）

- **[更新] 换挂结构比对：第二优先级 Tier-A/B**
  - 重写为 5 步：同位分区 → Tier-A（后开口/侧围）→ 颜色交叉校验 → Tier-B → 结论；号牌不可靠时必须先走 Tier-A，禁止空泛“栏板式一致”。

- **[增强] 货物误判防护（2 条超短硬规则）**
  - 禁止将堆料/篷布轮廓当作尾门或侧围；无固定门板/立柱证据不得判有尾门/侧挡板。

- **[增强] 号牌一致即正常（最高优先级）**
  - 两侧挂车号牌/尾部放大号清晰且关键位一致 → 直接判「正常」，`structure_consistency` 填「未检验」，不再做结构比对。
  - 禁止「号牌一致但因篷布遮挡结构」仍输出「无法判断」。

### 主视角/头部视角车尾（tail1/tail2，`predict_ai.py`）

- **[重构] 与尾部视角提示词分离**
  - `_build_tail_prompt()` 专用于车头方向下的车尾裁切（侧挡板、轴数、挡泥板、侧挂附件等可见项）。
  - 废弃此前与 `predict_ai_shijiao2` 同构的尾门/Tier-A/尾灯/号牌比对；不写本视角通常不可见的正后方部件。

### 车头二次复核（`predict_ai.py`）

- **[增强] 输出格式与同位自证**
  - 理由与标签分两段，最后一行仅 `fake_plate` 或 `normal`；禁止 `fake_plate: normal reason:` 混写。
  - 涉及文字/标识差异时须写「子区域、图1/图2、同部位对齐=是/否」；未对齐则文字差异无效，不得单凭文字判套牌。

### 代码清理（`predict_ai.py`）

- **[移除] 未再使用的接口与 prompt**
  - 删除 `_build_prompt()`、`check_vehicle()`（整车单轮三分类，主流程已改为分部位 `check_head` / `check_tail`）。
  - 删除 `_build_diff_analysis_prompt()`、`analyze_differences()`（差异描述改由 `ai_*_reason` 承担，GUI 不再调用）。
  - 保留 `_build_tail_prompt()`、`check_tail_with_reason()`（主视角车尾回退 AI 仍在用）。
  
### 2026-05-31

**差异总结展示（`my_predict_gui_new.py`）**
- 移除 `最终差异总结` 的 `_shorten_reason_text` 截断逻辑，套牌/换挂场景下输出完整 AI 理由，与上方「AI 判定理由」保持一致。

**车头 AI（`qwen_vl/predict_ai.py`）**
- 输出格式由「最后一行英文结论」改为 **JSON**（`label` + `reason` 分离），避免 reason 中出现「不作为 fake_plate 依据」等表述时被全文子串误匹配成套牌。
- 提示词新增 **顶棚阴影与顶边灯**、**挡风玻璃与雨刷** 专节：单侧阴影不可写「图1无、图2有」；雨刷位置不得作为套牌依据；顶边灯使用子区域 `deflector_top_lamp_strip`。

**尾部视角车尾 AI（`qwen_vl/predict_ai_shijiao2.py`）**
- 新增 **第 0 步成对可比对性审查**：任一侧未拍到挂车尾部时，必须输出「无法判断」并回退主视角车尾 AI，禁止用牵引车头与挂车颜色差异判换挂。
- JSON 增加 `img1_trailer_rear_visible`、`img2_trailer_rear_visible`、`pair_comparable` 字段；`structure_consistency` 支持「无法确认」。
- 解析层 `_apply_comparability_rules`：当不可比对、或换挂但号牌/结构无明确不一致证据时，自动降级为「无法判断」，触发主视角车尾 AI 回退。

## 2026-06-02

### 车头 AI（`qwen_vl/predict_ai.py`）

- **[提示词重构] 引入一票否决 V1–V7（最高优先级）**
  - 明确「一侧有字一侧无字 / 字符看不清」不得单独判套牌，须先归因为过曝、反光、阴影、背光或时段光照；
  - 任一侧文字区强反光、过曝、深阴影时该子区域文字证据作废；
  - 格栅/大灯/保险杠/车标等硬结构整体一致时默认 `normal`，禁止仅凭文字可见性改判套牌；
  - `reason` 若写「图1无、图2有」类表述，必须同时写明两侧可读性，否则 `label` 必须为 `normal`。
  - 效果：针对过磅现场清晨顶光、夜间点光源、大光比背光下导流罩「单侧可见文字」误判套牌的问题做硬性约束。

- **[提示词优化] 规则压缩与四步思考顺序**
  - 将原先分散的强光、色号漂移、顶棚阴影、雨刷等专节合并为「光照与成像（摘要）」；
  - 思考顺序固定为：步骤1 硬结构 → 步骤2 子区域对齐 → 步骤3 可读性 → 步骤4 定案，禁止跳步。

- **[输出要求] JSON `reason` 结构化模板**
  - 要求优先填写：子区域、图1/图2 可读性（清晰/过曝/阴影/反光）、硬结构一致与否、文字证据采纳或作废原因；
  - 触发 V1–V2 或同部位未对齐时 `label` 必须为 `normal`；
  - `reason` 禁止出现 `fake_plate`、`normal` 等英文 label 词，避免与解析层冲突。

### 主视角车尾 AI（`qwen_vl/predict_ai.py`）

- **[输出格式] 主视角车尾裁切 AI 同步改为 JSON**
  - `_build_tail_prompt()` 输出 `label`（`change_trailer` / `normal`）+ `reason` 分离；
  - 新增 `_parse_tail_response`、`_call_tail_model_with_reason`，与车头解析路径拆分，不再共用英文末行关键词提取。

- **[解析层] 车头/车尾分路调用**
  - `check_head` / `check_head_with_reason` 走 `_call_head_model_with_reason` + `_parse_head_response`；
  - `check_tail` / `check_tail_with_reason` 走 `_call_tail_model_with_reason` + `_parse_tail_response`；
  - 效果：车头、主视角车尾均统一为 JSON 解析，降低 reason 正文子串误匹配 label 的风险。

### 尾部视角车尾 AI（`qwen_vl/predict_ai_shijiao2.py`）

- **[提示词优化] 新增“单侧编号不可用于换挂”硬约束并强调结构优先**
  - 新增“编号可用性对称规则”：仅当两侧挂车号牌/放大号都清晰完整可读时，才可用编号一致/不一致定案；
  - 任一侧不可读、缺失、眩光、过曝或仅见局部字符时，`plate_or_number_consistency` 必须填“无法确认”，放弃编号比较并转结构链路；
  - 新增“编号来源白名单”：编号仅可来自挂车本体合法区域（号牌安装区、尾部放大号规范区域、车架正式编号区）；侧板喷字、货台喷字、背景指示屏/道闸屏/建筑牌编号一律排除。

- **[反例约束] 压制“单侧有号牌 + 单侧侧板喷字”误判换挂**
  - 明确：若图1号牌可读、图2号牌不可见，且图2仅出现侧板喷字或其他非号牌文字（如 `桂BA0596`），不得写成“两侧放大号不一致”；
  - 必须写明“编号证据不对称，已放弃编号比较，转结构比对”。

- **[链路强化] 编号不可靠时，Tier-A 成为主判断依据**
  - 在 `Step1` 明确“单侧编号禁止定换挂”；
  - 在 `Step2` 明确“后开口 + 侧围”为编号不可靠时的主判断依据，着重比较侧挡板、前/后挡板、顶棚/顶架等稳定结构。

### 备份整理

- 删除 `备份/0412`、`0507`、`0508`、`0510`、`0511` 等过期快照；
- 新增 `备份/0531/`（`my_predict_gui_new.py`、`predict_ai.py`、`predict_ai_shijiao2.py`），作为 5 月末基线留存。

## 2026-06-05

### 尾部视角车尾 AI（`qwen_vl/predict_ai_shijiao2.py`）

- **[硬约束] 编号比对细化（H2/H4/H8/H9）**
  - H2/H4 统一为「同类编号」才可比较（均为号牌或均为放大号）；
  - H8：放大号与号牌禁止跨类混比（如 BG136 vs G1966），跨类时 `plate_or_number_consistency=无法确认` 并转结构比对；
  - H9：结构相似时禁止强读编号，单侧字符碎片/眩光/积灰不得强行定不一致。

- **[Step1] 编号定案与结构切换**
  - 补充放大号≠号牌反例、结构相似禁止强读编号、编号确凿不同无需结构证实（双侧同类编号清晰且关键位不同 → 直接换挂）；
  - 单侧编号仍禁止单独定换挂。

- **[Step2 Tier-A] 光照与几何防误判**
  - **色相不可信**：任一侧存在夜间欠曝、顶棚阴影、积灰泛白、强反光 → 该侧 `body_hue` 标记「色相不可信」，不得单独定换挂；
  - **几何优先**：先数金属外框线、开口/窗洞数量、立柱/横梁布局，再判实心/镂空；禁止凭洞内或板面黄/黑/亮/暗填色定结构型；
  - **镂空窗洞透光规则**：同位置 N 个矩形外框一致时，洞内黄/黑/亮/暗、侧光填满或进深阴影仅属光照差，不得判实心 vs 镂空或定换挂；
  - 针对「四窗洞一黄一黑误判换挂」「放大号一致仍被结构填色带偏」等场景。

- **[特殊防误判] 新增 5–7 短版**
  - 混比防护、强读防护、编号确凿防护（双侧同类编号清晰且关键位不同 → 直接换挂，不必等结构二次证实）。

### 主视角车尾 AI（`qwen_vl/predict_ai.py`）

- **[提示词重构] Step0–Step4 主流程（禁止跳步）**
  - **Step0** 每图光照归因 → `panel_hue_reliable`：顶棚欠曝、夜间/暗环境、外来强光/车灯、栏板光滑反光、积灰泛白任一项 → 该侧栏板色相不可信；
  - **Step1** 几何优先：竖筋/立柱、栏高、镂空笼 vs 实心栏、轴数与轮组；禁止凭表观色推断材质（金属/木质）；
  - **Step2** 安装件须双侧清晰：反光条/挡泥板/侧挂箱；阴影侧「看不见」= 无法确认；附件颜色 alone 不定换挂；
  - **Step3** 栏板色末位：仅几何+安装件一致且无编号可比对、且两侧 `panel_hue_reliable` 均为是时才比栏板色；
  - **Step4** 兜底：任一侧色相不可信且仅有颜色/材质表观差 → 必须 `normal`。
  - 效果：抑制「一红一褐/金属 vs 木/反光条一侧看不见/附件颜色不同」等昼夜与阴影导致的换挂误判。

- **[输出要求] reason 约束**
  - 栏板色相不可信时须写几何/安装件一致或证据不足，禁止仅以颜色/材质定换挂。

## 2026-06-09

### 判定链路（`my_predict_gui_new.py`）

- **[裁切状态] 新增 `crop_status` 全链路透传**
  - `_build_crop_status` 逐层检测整车/车头/主视角车尾裁切是否成功（比较 parent vs child PIL 像素是否变化）；
  - 输出 `vehicle1/2_ok`、`head1/2_ok`、`main_tail1/2_ok`，以及 `head_ai_asymmetric`、`main_tail_ai_asymmetric` 不对称标记；
  - 不对称时打印 `[predict] crop_status` 日志；`crop_status` 写入 API 响应与 `_record_metric` 指标记录。

- **[裁切守护] 车头/主视角车尾 AI 定案分路解析**
  - `_resolve_head_ai_with_crop_guard`：reason 含「裁切失败侧无目标车辆」→ 直接 `fake_plate`（`crop_no_vehicle`）；
  - AI 返回 `unknown` 或 reason 含「图片质量太差/AI无法判断」→ 按车头相似度与阈值比较回退（`similarity_fallback`）；
  - `_resolve_main_tail_ai_with_crop_guard`：主视角车尾 AI 同理，`unknown`/质量太差时按车尾相似度回退；
  - 新增 `head_ai_decision_source`、`main_tail_ai_decision_source` 字段（`ai` / `crop_no_vehicle` / `similarity_fallback` / `invalid`），便于追溯定案来源。

- **[链路优化] 车头已正常时的车尾回退策略**
  - `_head_ai_cleared_normal`：车头 AI 已判 `normal` 后，主视角车尾 AI 无有效结论（invalid/裁切保存失败/全流程仍 inconclusive）时，不再回退 stage1，改为 `_apply_main_tail_similarity_fallback` 按车尾相似度定案；
  - 效果：车头已排除套牌后，车尾 AI 因裁切/质量原因无法判断时，用相似度阈值给出明确结论，减少「维持原结论」的模糊输出。

- **[H2 二次校验] 尾部视角 GUI 层硬拦截**
  - `_apply_tail34_h2_guard`：3/4 尾部视角 AI 返回后，若 `plate_or_number_consistency=一致` 或 reason 含号牌一致表述但 label 为换挂，GUI 强制改为「正常」、`structure_consistency=未检验`；
  - 与 `predict_ai_shijiao2.py` 解析层 `_apply_h2_plate_match_guard` 形成双保险。

### 车头 AI（`qwen_vl/predict_ai.py`）

- **[提示词] 注入系统裁切状态与不对称裁切规则**
  - 新增 `_build_head_crop_context` / `_build_main_tail_crop_context`，将 `crop_status` 写入提示词供模型采信；
  - **C1–C3 不对称裁切**：仅一侧裁切失败时，先观察失败侧是否仍有目标车辆；无车 → `fake_plate`（车头）或 `unknown`（主视角车尾）；有车但为全景 → 进入全景 vs 特写规则；
  - **P1–P3 全景 vs 特写**：禁止比对小结构（格栅条纹、栏板纹理等），只比整体轮廓/布局；仍不可比 → reason 写「输入图片质量太差，AI无法判断」、`label=unknown`，禁止强行定案。

- **[输出格式] 新增 `unknown` label**
  - 车头 label 支持 `fake_plate` / `normal` / `unknown`；主视角车尾支持 `change_trailer` / `normal` / `unknown`；
  - `check_head_with_reason` / `check_tail_with_reason` 新增 `crop_status` 参数透传。

### 尾部视角车尾 AI（`qwen_vl/predict_ai_shijiao2.py`）

- **[H2 硬拦截] 解析层 `_apply_h2_plate_match_guard`**
  - 双侧挂车尾部可见且 `plate_or_number_consistency=一致` 时，无论模型原结论如何，强制 `label=正常`、`structure_consistency=未检验`；
  - 原换挂/无法判断结论写入 reason 附注，打印 `[tail-ai] H2 guard adjusted label` 日志。

- **[提示词] Tier-A 与 H2 优先级对齐**
  - Step2 Tier-A 明确为「仅当 `plate_or_number_consistency=无法确认` 时作为主判断依据」；H2 已成立（编号一致）时 Tier-A 结构结论一律无效；
  - Step3 色相交叉校验、Step5 结论均补充「若 `plate_or_number_consistency=一致`，结构差异不适用」；
  - Step5 新增 H2 优先条目：双侧同类编号关键位一致 → 直接正常，禁止再引用 Tier-A/Tier-B 结构差异。

### 备份整理

- 更新 `备份/0607/`（`my_predict_gui_new.py`、`predict_ai.py`、`predict_ai_shijiao2.py`），同步裁切守护与 H2 拦截逻辑。

### 工具脚本

- 新增 `scripts/gen_summary_0522_0531.py`：生成 0522–0531 阶段总结 Word 文档。

