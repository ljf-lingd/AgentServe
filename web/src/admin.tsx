import { useState } from 'react'
import { Alert, Button, Card, Popconfirm, Space, Spin, Statistic, Table, Typography } from 'antd'
import { errorMessage, modeLabel, post, statusLabel, type Status, type TaskError, type ViewMode } from './api'
import { useJson } from './hooks'

type Metrics = {
  time_to_first_token_ms?: number
  tokens_per_second?: number
  mean_itl_ms?: number
  num_draft_tokens?: number
  num_accepted_draft_tokens?: number
}
type Overview = {
  fetched_at: number
  cleared_at: number | null
  counts: Record<Status, number>
  jct: { mode: ViewMode; n: number; p50_seconds: number; p95_seconds: number }[]
  errors: { task_id: string; ended_at: number; mode: ViewMode; error: TaskError | null }[]
  requests: {
    task_id: string; request_id: string; ended_at: number; mode: ViewMode; phase: string
    output_tokens: number | null; metrics: Metrics | null
  }[]
}
const number = (value?: number | null) => value == null ? '—' : Number(value.toFixed(2))
const phaseLabel: Record<string, string> = {
  sql: 'SQL 生成', memory_plan: '记忆规划', memory_write: '记忆写入',
  knowledge_answer: '知识问答', answer: '文本生成',
}
const date = (seconds: number) => new Date(seconds * 1000).toLocaleString('zh-CN', { hour12: false })
const identifier = (value: string) => <Typography.Text title={value} copyable={{ text: value }}>
  {value.slice(-8)}
</Typography.Text>

export function AdminOverview() {
  const { data, error, reload } = useJson<Overview>('/admin/overview')
  const [resetError, setResetError] = useState<string>()
  async function reset() {
    setResetError(undefined)
    try {
      await post('/admin/overview/reset')
      reload()
    } catch (error) {
      setResetError(errorMessage(error))
    }
  }
  return <main className="admin">
    <div className="section-title">
      <div><Typography.Title level={2}>运行概览</Typography.Title>
        <Typography.Text type="secondary">
          当前任务状态与近 24 小时的运行记录
          {data?.cleared_at ? ' · 清空于 ' + date(data.cleared_at) : ''}
          {data?.fetched_at ? ' · 更新于 ' + date(data.fetched_at) : ''}
        </Typography.Text>
      </div>
      <Space wrap>
        <Button onClick={reload}>刷新</Button>
        <Popconfirm title="清空概览记录？" description="从现在重新统计，历史会话和任务结果会保留。"
          okText="清空" cancelText="取消" onConfirm={reset}>
          <Button danger>清空记录</Button>
        </Popconfirm>
      </Space>
    </div>
    {error && <Alert type="error" title={error} />}
    {resetError && <Alert type="error" title={resetError} />}
    {!data && !error && <Spin />}
    {data && <Space orientation="vertical" size="large" className="full">
      <div className="statistics">
        {Object.entries(data.counts).map(([status, count]) => (
          <Card key={status} size="small"><Statistic title={statusLabel[status as Status]} value={count} /></Card>
        ))}
      </div>
      <Card title="任务完成耗时（JCT）">
        <Typography.Paragraph type="secondary">成功任务从入队到持久化终态的耗时，包含排队。</Typography.Paragraph>
        <Table size="small" rowKey="mode" dataSource={data.jct} pagination={false} columns={[
          { title: '路径', render: (_, row) => modeLabel[row.mode] },
          { title: '样本数', dataIndex: 'n' },
          { title: 'P50（秒）', render: (_, row) => number(row.p50_seconds) },
          { title: 'P95（秒）', render: (_, row) => number(row.p95_seconds) },
        ]} />
      </Card>
      <Card title="推理观测">
        <Typography.Paragraph type="secondary">
          最近 30 个已记录的模型响应。模型输出速率包含 prefill。— 表示该请求没有对应指标。
        </Typography.Paragraph>
        <Table size="small" rowKey={(row) => row.task_id + ':' + row.request_id}
          dataSource={data.requests} scroll={{ x: 'max-content' }} pagination={{ pageSize: 10 }} columns={[
            { title: '时间', render: (_, row) => date(row.ended_at) },
            { title: '业务模式', render: (_, row) => modeLabel[row.mode] },
            { title: '模型阶段', render: (_, row) => phaseLabel[row.phase] ?? row.phase ?? '—' },
            { title: '输出 token', render: (_, row) => number(row.output_tokens) },
            { title: 'TTFT（ms）', render: (_, row) => number(row.metrics?.time_to_first_token_ms) },
            { title: '模型输出速率（tokens/s）', render: (_, row) => number(row.metrics?.tokens_per_second) },
            { title: '平均 ITL（ms）', render: (_, row) => number(row.metrics?.mean_itl_ms) },
            { title: 'SAM 草稿', render: (_, row) => number(row.metrics?.num_draft_tokens) },
            { title: 'SAM 接受', render: (_, row) => number(row.metrics?.num_accepted_draft_tokens) },
            { title: '任务编号', render: (_, row) => identifier(row.task_id) },
            { title: '模型请求编号', render: (_, row) => identifier(row.request_id) },
          ]} />
      </Card>
      <Card title="最近失败">
        <Table size="small" rowKey="task_id" dataSource={data.errors} scroll={{ x: 'max-content' }}
          pagination={false} columns={[
            { title: '时间', render: (_, row) => date(row.ended_at) },
            { title: '任务编号', render: (_, row) => <Typography.Text copyable>{row.task_id}</Typography.Text> },
            { title: '业务模式', render: (_, row) => modeLabel[row.mode] },
            { title: '错误码', render: (_, row) => row.error?.code ?? '—' },
            { title: '说明', render: (_, row) => row.error?.detail ?? '—' },
          ]} />
      </Card>
    </Space>}
  </main>
}
