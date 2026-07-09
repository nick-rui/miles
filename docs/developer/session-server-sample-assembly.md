# Worker 内样本装配:records 不再离开 session server

状态:proposal(multi-process session server stack 的后续,见 `docs/developer/multi-process-session-server.md`)。读者:该 stack 的 reviewer。读完你应当能判断:所有权边界的改动是否正确、新端点契约是否可靠、宣称的 parity 是否真正可验证。

## 动机

今天训练路径把**原始 session records** 从 session server 拉出来,在 Miles 侧装配训练 `Sample`(`agentic_tool_call.generate` → `collect_records()` → `compute_samples_from_openai_records` → `truncate_samples_by_total_tokens` → `merge_samples`)。在 R3(`routed_experts` / `indexer_topk`)下这有两笔实测/结构性代价,外加一个已在发生的 bug:

- **records dump 是全系统最重的对象,且要跨越每一层边界。** 生产 shape(32 sessions × 50 turns)下一次全量 `GET /sessions/{id}` 实测 ~3.15 GiB。owning worker 对它做两趟序列化(`model_dump` + `json.dumps`,同步占住 event loop),它作为单帧跨越 IPC(实测:collect 期间 chat 回复 p99 崩塌 ~33×,GET 平均 ~103 s,router `/health` 在 worker 被卡住时假 503),然后 rollout driver 再把它全部解析回来(`json.loads` + pydantic 校验 + 逐 record 的 R3 → numpy 解码)。
- **Miles 侧装配跑在单个解释器里。** 32 条并发轨迹的 GiB 级 parse + decode + merge 全部串行在 rollout driver 的一个 GIL 上——恰好是 multi-process session server 被造出来要消灭的瓶颈形状,在下游一跳处被原样复活。
- **collect 尾部已经超过它自己的 timeout——今天就在静默丢数据**(方向层 review 发现)。`collect_records` 最多等 `_SESSION_REQUEST_TIMEOUT = 120` s(`openai_endpoint_utils.py:20`),而 records GET 在生产 shape 下平均 ~103 s,尾部 collect 会超时、返回 `[], {}`(`openai_endpoint_utils.py:63-76`),`agentic_tool_call.py:93-97` 把它转成静默 ABORTED 样本。决定在案:不出临时缓解 PR;本重构移除这条路径(切换 PR)。

消解这一切的观察:**records 的唯一消费用途就是装配 `Sample`,而每一项装配输入都已经在 owning worker 手里**(records、`accumulated_token_ids`、`max_trim_tokens`,以及与 rollout 侧 `GenerateState.tokenizer` 构造完全相同的 `load_tokenizer(args.hf_checkpoint, ...)` tokenizer)。外界只需要装配完的 `Sample`。所以改所有权边界:**session server 装配并返回 `Sample`;原始 records 是内部状态,永不外泄**(仅保留一个 debug-only 的 dump 端点)。

收益不只是把 CPU 挪到解释器可扩展的地方:默认(merge)路径下,`merge_samples` 只保留**最后保留轮**的 R3 数组(`sample_utils.py:123`;注意 merge 在第一个非 COMPLETED 轮停止,`sample_utils.py:15-17`,所以"保留轮"可能是轨迹中段的),跨边界的字节量从 ~3.15 GiB JSON 降为一个 merged `Sample`——该单轮 R3 以 int32 二进制计,**生产 shape 下约 ~100 MiB**(134 MiB 末轮 body 主要是 base64 的 R3,`generate_endpoint_utils.py:100`;int32 ≈ base64 文本的 0.75×)。即 **~30× 的传输缩减**,诚实版本:每 session 仍有一个大回复帧,IPC head-of-line 是减轻而非消除;且 `generate_multi_samples=True` 保留每轮 R3 数组,只赚 parse-offload,不赚传输。

## 约束

硬约束:

- **Sample parity。** 相同的 records 与输入下,新路径产出的 `Sample` 必须与现有 Miles 侧管线**逐字段一致**(tokens、`loss_mask`、logprobs、R3 数组、status、metadata 内容**及** metadata 优先序)。现有单测(`tests/fast/rollout/generate_utils/test_openai_endpoint_utils.py`)随函数迁移,断言不变。
- **响亮失败。** 装配断言(trim/cursor 一致性、merge 前缀链)今天会立刻炸掉 rollout driver。它们必须保持立即:确定性装配失败不得被客户端的盲重试循环掩盖(`http_utils._post` 对任何错误重试至多 60 × 1 s)。
- **Router 保持 payload 无关。** 它像对待其他所有回复一样原样中继 samples 回复(`_reply_to_response`);从不 unpickle,也不 import torch 或 tokenizer 栈。
- **不新增 per-worker 内存底座。** 实测(2026-07-08):worker import 闭包本就是每进程 ~954 MB private——本环境中 `transformers` eager import `torch`(~490 MB,`processing_utils.py:10`),且 `chat_template_utils` 拉入 `sglang`——所以 worker import `miles.utils.types.Sample` 的增量可忽略。(早期草案基于只扫 miles 文件的 AST 遍历声称闭包无 torch;直接测量推翻了它。)
- **其余一切不动。** sticky hash 路由、chat 路径、DELETE 语义、supervisor/health 层、其余端点的可观测行为均不变。

软约束:新增协议面最小(一个 op、一条路由)。(早期草案还保留 `tests/e2e/sglang/utils/logprob_verify_generate.py`;调查发现其消费测试早已在 25399d3ff 被删——"replaced by session e2e"——故该 util 作为死代码删除,`collect_records` 随切换一并删除。)

本期非目标:

- 缩减 `generate_multi_samples=True` 的传输量(每轮 sample 各带全序列 R3 数组;仍是 GiB 级)。它作为显式请求参数暴露,代价接受并记录。
- worker 内 records 内存缩减(只保留最新一轮的 R3,每轮覆盖)。新边界解锁了它;作为独立 follow-up。
- 任何传输层工作(chunking、worker-direct listener)。见方案选择。

## 方案选择

三个候选,用约束逐一衡量:

- **A. 保持边界,修传输**——先前已决定的 records-path TODO:router 把 records GET 307 重定向到 owning worker 自己的 HTTP listener,GiB body 绕过 IPC 和 router。这只消除 IPC HOL:worker 仍在 event loop 上双重序列化,rollout driver 仍在单解释器里解析每条轨迹 3.15 GiB。动机的后半直接不满足。
- **B. 把装配搬进 worker**(选中)。parse/decode/merge 发生在 records 所在之处、解释器本就可扩展之处(`--session-server-workers`);GiB 序列化是消失而非改道;wire 上只走几十 MiB 的成品 `Sample`。每条硬约束均可满足:同样的函数、同样的 tokenizer 构造、同样的断言面。
- **C. 保持边界,Miles 侧并行化**——rollout driver 里开进程池跑 `compute_samples`。重复建设 session server 已有的解释器编队,仍要传输并解析 3.15 GiB,还多付一跳序列化(records 进池)。被 B 严格支配。

选 B 后,热路径不再传输 records,`multi-process-session-server.md` 里的 307 records-path TODO **溶解**——它的动机测量不再描述热路径。records dump 端点以 **debug-only** 存续(唯一剩余消费者:人肉排查 TITO mismatch),明确豁免性能承诺。

## 设计

### 端点与 op

router 上 `POST /sessions/{session_id}/samples`,IPC 上 `OP_SAMPLES`,与其他 session op 一样按 session hash 路由到 owning worker。用 POST(而非 GET)因为请求带 body。

注册顺序契约(milestone review 发现):router 现有 catch-all 代理路由 `@app.api_route("/sessions/{session_id}/{path:path}")`(`router.py:147`)今天会把这条路径吃进 `OP_PROXY` 转发给 backend——即它今天不是 404。新路由必须像 chat 路由(`router.py:134`)一样注册在 catch-all **之前**(Starlette 按注册顺序匹配),否则被静默遮蔽;"新增死路由"的准确说法是"遮蔽一条今天会被代理的路径"。

### Wire format —— 已决定:A,显式 codec + driver 侧 overlay(2026-07-08)

分叉:worker 返回结果时,是发**整个 pickle 的 `Sample` 对象**(B),还是**只发它算出的字段**、由 driver overlay 到本地 `deepcopy(input.sample)` 上(A)?两者产出逐字段一致的最终 `Sample`(由 parity 测试锁定);差别在什么跨越进程边界、wire 依赖什么。

A 的事实基础:`Sample` 的 26 个字段里,worker **算出**的只有这些——`tokens`、`response`、`response_length`、`loss_mask`、`rollout_log_probs`、`rollout_routed_experts`、`rollout_indexer_topk`、`status`、`weight_versions`、`prefix_cache_info`——外加每 session 的 `session_metadata` dict。其余所有字段(`prompt`、`label`、`reward`、`group_index`、`index`、`metadata`、`multimodal_*`、`train_metadata`、`session_id`、`spec_info`——本路径不触碰(simpler review:归 template 侧,不上 wire)……)都是从 `input_sample` deepcopy 而来、装配全程原样携带。

等价前置条件(milestone review 发现):旧管线对部分字段不是"覆写"而是**原地演化**——`weight_versions` append(`openai_endpoint_utils.py:200-201`)、`prefix_cache_info` 累加(`:199`)、merge 对 `spec_info` 求和(`sample_utils.py:132,141-151`,merge n 轮 = n × 输入值)、`strip_last_output_tokens` 裁剪 `teacher_log_probs`/`opd_reverse_kl`/`metadata["opd_student_top_logprobs"]`(`types.py:200-207`);而空白模板 + overlay 在这些字段上是**替换**语义。两条路径等价**当且仅当** `input_sample` 在这些字段上处于 dataclass 默认值。实践中恒真(样本由 data loader 全新构造;框架重试走 `reset_for_retry` 复位),但 parity fixture 全用 fresh 输入锁不住这一分叉类,所以由两处强制:`sample_assembly.py` 模块头记录该前置条件;driver overlay 处对这些字段加 fail-loud 默认值断言(切换 PR,几行)。

**方案 A —— 显式 codec + driver 侧 overlay**(方向层 reviewer 推荐,本文档采纳):

- 请求:`{"multi_samples": bool, "max_seq_len": int | null}` —— envelope meta 里的纯 JSON。`input_sample` 与 `agent_metadata` 从不跨进程。
- worker 用空白模板 `Sample()` 装配;回复在 IPC 上仍是**单个** envelope(meta + body 各一,`ipc.py:72-83`,`worker.handle` 只回一个 body)——per-sample 的分帧由 codec 自己打包:meta 里放 per-sample 的 JSON 元数据数组(算出的标量/列表字段)与二进制段偏移表,body = 全部二进制段拼接(`tokens`/`rollout_log_probs`/R3 数组,dtype + shape 记在对应 meta;浮点 f64),顶层附 `session_metadata` 与 `empty_reason`。
- driver:`sample = deepcopy(input.sample)`,overlay 返回的字段,再按今日顺序 apply `agent_metadata` 与 `session_metadata`。
- 等价性论证(parity 测试锁定):`merge_samples` 对模板字段的相等断言在空白模板之间平凡成立;truncation 只读 `tokens`/`response_length`。(`agent_metadata` 的可交换性论证两方案共用——见流程注记。)
- 防漂移守卫:codec 声明 `COMPUTED_FIELDS` 与 `TEMPLATE_FIELDS`,并断言其并集等于 `dataclasses.fields(Sample)`——给 `Sample` 加字段而不表态归属,会在 import 时响亮失败,而非在训练时静默出错。
- 收益:全程无 pickle;无跨进程版本耦合;multimodal `input_sample` 成为非问题;wire 可肉眼检查。
- 成本:~50–80 行显式编解码 + 字段清单维护(由上述守卫兜底)。

**方案 B —— pickle 整个 `Sample`**(已拒绝——留作备选方案的记录):

- 请求:`pickle({input_sample, multi_samples, max_seq_len})`;回复:`pickle({samples, empty_reason})`。(`agent_metadata` 两方案下都留在 driver 侧——见流程注记。)
- 收益:代码最少——回复即成品;无字段清单维护。
- 成本:两端必须跑同一代码版本(`Sample` 布局一改,跨版本 unpickle 运行时炸);对经过 HTTP 面的字节做 unpickle 是 RCE 形状的坏习惯,即使在 localhost;`input_sample` 可能携带 multimodal payload——能否 pickle 是持续悬置的问题;排查时 wire 不可读。
- 方向层 reviewer 的立场:"no pickle over HTTP——定义显式 codec;倾向返回轨迹字段、`input_sample` 在 driver 侧 overlay。"

共同简化(请求 = `{multi_samples, max_seq_len}` 标量;`agent_metadata` 两方案均不跨)之后,分叉收敛为恰好两条轴:

1. **`input_sample` 跨不跨边界?** A:不跨——worker 空白模板装配、driver overlay;multimodal payload 悬置消失。B:跨——worker 需要它作 deepcopy 模板产出成品 `Sample`;能否 pickle 持续悬置(实现时 assert-reject)。
2. **回复怎么编码?** A:显式 codec——JSON meta + 原始二进制段,~50–80 行,wire 可查,`Sample` 新字段经防漂移守卫强制表态。B:pickle——零 codec 代码,新字段自动携带,但跨版本 unpickle 运行时炸、对 HTTP 传输的字节 unpickle 是 RCE 形状的习惯;排查时 wire 不可读。

决定(2026-07-08):**A**。codec 的一次性成本被判定可接受;B 的两条轴(`input_sample` 跨界、pickle 耦合)由此被回避而非记录。这也与方向层 reviewer 的推荐一致。

### Worker 侧流程(与 `agentic_tool_call.generate` 93–129 行逐一对应)

1. `records = session.records`;为空 → 回复 `{"samples": [], "empty_reason": "no_records"}`。
2. `compute_samples_from_openai_records(args, input_sample, records, tokenizer, accumulated_token_ids, max_trim_tokens)` —— 后两个参数直接读 worker 自己的 session 状态,不再跨 wire 往返。
3. 若有 `max_seq_len`:`truncate_samples_by_total_tokens`;全截断 → 回复 `{"samples": [], "empty_reason": "all_truncated"}`。truncation 保持在 merge **之前**——它是 turn 级预算决策(哪些轮存活;超限轮在轮边界裁剪、其后各轮丢弃),而轮结构只在 merge 前存在;同一调用点还统一覆盖 `multi_samples` 模式,并在 deepcopy 密集的 merge 之前剪枝。`sample_assembly.py` 模块头记录此顺序契约。
4. 若非 `multi_samples`:`merge_samples`;`session_metadata`(与今日 `get_session` 构建的同一 dict:`tito_session_mismatch`、`accumulated_token_ids`、`max_trim_tokens`)随回复带回、由 driver 侧 apply——最终位置与今日相同(merged sample,或 `samples[-1]`)。该 dict 的构建今天内联在 `get_session` 里(`core.py:139-148`);samples op 要"同源不 fork"就必须把它抽成 helper、两个 op 共用——所以 c2 对既有 `get_session` 有一处**纯抽取**改动(行为不变),不是严格 additive(milestone review 发现)。
5. 回复装配好的 samples 及 `empty_reason`(形状见上方 wire format 决定)。

整个 op 在 worker 的 event loop 上同步执行,从读取 session 状态到装配完成之间**没有 `await`**——这正是今天无锁 `get_session` 对 chat 路径安全的同一不变量(chat 只在 session lock 内、await 间隙处改状态)。若实现把装配 offload 到 executor,必须先快照 records 或持 session lock;不要假设无锁读能在搬动后幸存。

`agent_metadata` 在两种 wire 方案下都由 driver 侧在收到回复后 apply(generate function 里的 `for s in samples: s.metadata.update(agent_metadata)`,先于 `session_metadata`):该更新对所有 sample 均匀,与 truncate(不读业务 metadata)和 merge(其 metadata 相等断言有无它都成立)可交换——因此它根本不需要跨边界。这也把 `agent_metadata` 从方案 B 的请求里移除了。

### 错误映射

- `SessionError`(session 缺失/关闭中):错误形状与状态码不变,与所有 op 一致。
- 确定性装配失败:worker 映射为 **422**,断言原文作 body。422 捕获清单是 `AssertionError`(trim/cursor、merge 前缀链)与 `ValueError`(缺 `input_ids` 的 `openai_endpoint_utils.py:172-173`;按类型 catch 也会把 R3 解码失败扫进来——base64 的 `binascii.Error` ⊂ `ValueError`、reshape 尺寸不匹配 `generate_endpoint_utils.py:100-105`——接受:存储 record 损坏同样是确定性装配失败,422 带原文就是想要的响亮面)。try 块**只包裹装配 span**,请求 body 的 JSON 解析在其外——否则坏请求的 `json.JSONDecodeError`(⊂ `ValueError`)会被误标成装配失败。`UpstreamResponseError` **从清单移除**(milestone review):其 raise 点全在 chat 路径(`core.py:239,244,254`),samples op 调用链不可达;且它是 `SessionError` 子类(`errors.py:44-51`,status 502),即便假想触发也走 worker 既有的 `except SessionError` → 502 error_response(`worker.py:93-94`),不会变 ERROR 帧拆通道——留着只是一条把 502 错改成 422 的死防御分支。mismatch 计算里的 `TokenizationError` **不是** 422:今天 `get_session` 容忍它(`core.py:140-146`,log + `mismatch=None`),samples op 复刻该容忍——收紧它是行为变更,不是 parity。客户端对 samples 端点**单次直接 POST、不重试**(绕开 `http_utils.post` 的 60×1s 盲重试循环):任何非 2xx 立即 raise 并携带 body 原文。理由(simpler review):这里的 502/503 意味着 owning worker 死了——session 的 records 随之而死,重试只能换来 session-not-found;supervisor 的 `check()` 反正会让 rollout 失败。何时真观测到瞬时 5xx,何时再引入重试。实现原语(PR2 milestone review):`http_utils.post_bytes_no_retry`——共享 httpx client + `asyncio.wait_for` 保总时长 120s;`post()` 结构上不可复用,它对一切错误重试**且**强制 json/text 解码,会把二进制 envelope 打烂。清理语义 parity(同 review):session 的 DELETE 在**所有**路径上 best-effort 执行(成功、422、5xx、超时皆是;失败仅 warning,与今日一致)——今天的等价路径都在死亡前完成或尝试过 DELETE,若只在成功路径清理,反复失败会把 GiB 级 session 累积在 worker 里。
- **计划内行为 delta —— collect 超时**:今天 collect 超时静默返回 `[], {}`、样本被 ABORT(动机里的丢数据 bug)。`collect_samples` 改为超时**直接 raise**:装配在 server 侧是秒级,120 s 还超说明真出了问题,响亮失败是已定的姿态。
- 传输错误/worker 死亡:沿用 `SessionRouter.dispatch` 现有的 503/502 映射,不变。

### 无 import 前置(实测,取代早期的 torch-free types 计划)

早期草案要求先让 `types.py` 可在无 `torch` 下 import,worker 才能 import `Sample`。测量否定了前提:今天的 worker 闭包已经携带 `torch`(经 `processing_utils.py:10` 被 `transformers` eager import)和 `sglang`(经 `chat_template_utils`),每 worker ~954 MB private,所以 `types.py` 模块级的纯标注 `torch` import 增量为零。无前置里程碑;闭包问题关闭。缩减 worker import 底座(lazy torch/sglang;可省 ~954 MB × N workers 的大部分)是已记录的、本重构范围外的 follow-up。

### 代码落位

`compute_samples_from_openai_records`、`_compute_sample_from_openai_record`、`truncate_samples_by_total_tokens` 从 `openai_endpoint_utils.py` 移入 session 包(`miles/rollout/session/sample_assembly.py`);wire codec 放在**同一模块**,不单开文件——它不能进 `ipc.py`(stdlib-only,被 torch-free 的 router import),也不能进 `worker.py`(driver 需要解码侧而不背 registry/backend 栈),而 `sample_assembly.py` 已 import `Sample` 且两端均可 import。`merge_samples` 留在 `sample_utils.py`,由 worker import(单一来源,不复制)。`agentic_tool_call.py` 删除对三者的 import。装配/merge/trim 单测随函数迁移(断言不变);tracer 测试留在瘦身后的 `test_openai_endpoint_utils.py`。

### 文档更新(`multi-process-session-server.md` 及漏网引用)

- `docs/user-guide/rollout-endpoints.md:193,208` 按名引用 `compute_samples_from_openai_records`/`_compute_sample_from_openai_record` 并链接到 `openai_endpoint_utils.py`——随迁移 commit 一并改指新路径(milestone review 发现的漏网 importer)。
- records-path accepted-risk 条目改限定到 debug 端点(测量数字留作参照;热路径不再经过它)。
- "Records are never pruned (the training data path consumes R3 from them)" 改为"训练路径消费装配好的 `Sample`;records 留存至 DELETE,可经 debug 端点 dump"。
- 307 worker-direct TODO 删除(溶解,非被取代——其动机测量不再描述热路径)。
- decomposition 一节加入 samples op;`core.py`/`worker.py` 的 doc-dev 头同步。

## 验证环

1. **Parity 测试(闸门):** 录制 fixture(records + input_sample + agent_metadata)分别驱动旧的客户端管线与新的 worker 内管线,必须产出逐字段一致的 `Sample`(R3 数组用 `numpy.array_equal` 比较),覆盖 `multi_samples` 两种模式、有/无 `max_seq_len` 截断。fixture 内容要求(否则默认值空过):records 必须携带 `spec_*` 计数、`cached_tokens`、`weight_version` 和真实 R3 payload;`input_sample.metadata`、`agent_metadata`、`session_metadata` 必须含重叠 key,使应用顺序(agent 先、session 后、session 胜)被值锁定而非碰巧通过。codec 规格:浮点列表(`rollout_log_probs` 等)以 f64 二进制过 wire。
2. **错误面:** accumulated token 链断裂的 fixture 必须一个往返就把断言原文送到 rollout 侧(422,无重试)——由测试断言;并断言 channel 存活(随后 health op 仍通)。
3. **性能证据:** 在生产 shape 下跑一次 `tests/manual/session/bench_session_server_overhead.py` 的 samples 路径;预期:session 尾部 collect 不再落在 ~103 s 区间,collect 期间 chat p99 不再崩塌(**在装配中的 owning worker 上**测——跨 worker 平均会把数字稀释成空洞的通过),collect 负载下 router `/health` 不再假 503。注意最后一条是 **bench 验收项、不是结构保证**(milestone review):samples op 按设计同步占住 owning worker 的 loop,若生产 shape 下装配 + 编码超过 `_HEALTH_TIMEOUT = 5.0`(`router.py:42`),装配期间 `/health` 依旧可能 503——预期它秒级完成,由 bench 证实。该测量作为切换 PR 的验收证据手工执行(数字进 PR 描述与 verify 日志);把场景提交进 bench 脚本推迟到 follow-up(bench 在 `tests/manual/`,没有任何东西 gate 在它入库上)。

## 风险与开放问题

- **`multi_samples=True` 传输仍是 GiB 级**(二进制而非 JSON,约小 4–6×,但仍是单帧)。接受并记录;生产 shape 走 merge。
- **Worker 瞬时内存:** 装配期间 worker 短暂同时持有 records + 产出的 samples(merge 路径 ~多一个末轮 R3 数组)。系统净内存严格改善——rollout driver 不再为每条轨迹持有 3.15 GiB 文本 + 解析对象的副本。
- ~~开放——collect 超时语义~~ 已决定(计划层 review):`collect_samples` 超时 raise——见错误映射一节的计划内行为 delta。
