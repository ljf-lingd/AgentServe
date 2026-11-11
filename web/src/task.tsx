import { useEffect, useState } from 'react'
import { Alert, Button, Card, Space, Table, Tag, Typography } from 'antd'
import { chatAnswer } from './chat-answer'
import { activeTask, api, errorMessage, post, shiftMonth, statusLabel, type Input, type Task } from './api'

const eventLabels: Record<string, string> = {
  TASK_ACCEPTED: '已入队', TASK_CLAIMED: '开始执行',
  MODEL_STARTED: '请求模型', MODEL_COMPLETED: '模型已响应', MODEL_RETRY_SCHEDULED: '等待重试',
  SQL_STARTED: '执行只读查询', SQL_COMPLETED: '查询已返回', TASK_COMPLETED: '处理完成',
}

const labels: Record<string, string> = {
  sales_gbp: '销售额（英镑）', cancellation_gbp: '取消金额（英镑）', country: '国家',
  invoice_count: '发票笔数', cancellation_count: '取消发票笔数',
  order_count: '订单笔数', city: '城市', status: '订单状态', category: '类别',
  order_id: '订单编号', order_date: '下单日期', month: '月份',
  revenue_cents: '成交营收（分）', revenue: '成交额（分）', total_revenue: '成交额（分）', customer_count: '客户人数',
}

function columnTitle(name: string, index: number) {
  if (labels[name]) return labels[name]
  if (!/^[\p{L}_][\p{L}\p{N}_]{0,39}$/u.test(name)) return '结果 ' + (index + 1)
  return name.endsWith('_cents') ? name + '（分）' : name
}

function Output({ task, compact }: { task: Task; compact: boolean }) {
  const { report } = task
  if (!report) return null
  const result = report.query_result
  if (compact) {
    const answer = chatAnswer(task)
    return <Space orientation="vertical" size="small" className="full">
      {answer && <Typography.Text className="answer">{answer}</Typography.Text>}
      {result?.truncated && <Alert type="warning" title="只展示已返回的部分记录，请缩小查询范围。" />}
    </Space>
  }
  if (report.answer_source === 'workbench') {
    return <Typography.Text className="answer">{report.answer}</Typography.Text>
  }
  if (report.analysis && result && !result.rows.length) {
    return <Typography.Text>所选城市或日期没有匹配数据。请重新选择城市或日期。</Typography.Text>
  }
  const window = task.input.metric ?? task.input.analysis
  const answer = report.citations.length ? report.answer?.replace(/〔[^〕]+〕/g, '') : report.answer
  const artifacts = '/tasks/' + task.task_id + '/artifacts/'
  return (
    <Space orientation="vertical" size="middle" className="full">
      {answer && !['model', 'tool_values'].includes(report.answer_source ?? '') &&
        <Typography.Paragraph className="answer">{answer}</Typography.Paragraph>}
      {window && <Typography.Text type="secondary">
        {task.input.analysis
          ? `统计月份 ${window.start.slice(0, 7)} 至 ${shiftMonth(window.end.slice(0, 7), -1)}（含）`
          : `日期范围 ${window.start} 至 ${window.end}（结束日不含）`}{window.city && ' · ' + window.city}
        {report.metric?.unit && ' · 单位 ' + report.metric.unit}
      </Typography.Text>}
      {result && result.columns.length > 0 && <Table
        aria-label="查询结果" size="small"
        columns={result.columns.map((title, index) => ({
          title: columnTitle(title, index),
          dataIndex: String(index), key: index,
        }))}
        dataSource={result.rows.map((row, index) => ({
          ...Object.fromEntries(row.map((value, column) => [String(column), value == null ? '—' : String(value)])),
          key: index,
        }))}
        scroll={{ x: 'max-content' }} pagination={result.rows.length > 10 ? { pageSize: 10 } : false}
      />}
      {result?.truncated && <Alert type="warning" title="结果已截断，请缩小查询范围查看完整结果。" />}
      {report.answer && report.answer_source === 'model' && <section aria-label="模型解释">
        <Typography.Text type="secondary">模型解释 · 数值和单位以查询结果为准</Typography.Text>
        <Typography.Paragraph className="answer">{report.answer}</Typography.Paragraph>
      </section>}
      {report.citations.length > 0 && <section aria-label="引用来源">
        <details>
          <summary>引用来源（{report.citations.length}）</summary>
          <ol className="citations">{report.citations.map((source, index) => (
            <li key={index}><strong>{source.path}#{source.anchor} · 版本 {source.version}</strong><div>{source.quote}</div></li>
          ))}</ol>
        </details>
      </section>}
      {report.analysis && task.status === 'COMPLETED' && <>
        <img className="report-chart" src={artifacts + 'chart.svg'} alt="月度营收和订单量图" />
        <Space wrap>{['monthly.csv', 'report.md', 'chart.svg'].map((name) =>
          <Button key={name} href={artifacts + name}>下载 {name}</Button>,
        )}</Space>
      </>}
    </Space>
  )
}

export function TaskPanel({ initial, onRetry, onTerminal, onSaved, compact = false }: {
  initial: Task; onRetry: (input: Input) => void; onTerminal?: () => void; onSaved?: () => void; compact?: boolean
}) {
  const [latest, setLatest] = useState<Task>()
  const [event, setEvent] = useState('')
  const [error, setError] = useState<string>()
  const [cancelling, setCancelling] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const task = activeTask(initial.status) ? (latest ?? initial) : initial
  const active = activeTask(task.status)
  const id = task.task_id

  useEffect(() => {
    if (!active && activeTask(initial.status)) onTerminal?.()
  }, [active, initial.status, onTerminal])

  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    let cursor = 0
    let timer: number
    async function poll() {
      try {
        if (!compact) {
          const events = await api<{
            events: { type: string }[]
            next_after_event_id: number
          }>('/tasks/' + id + '/events?after_event_id=' + cursor, { signal: controller.signal })
          cursor = events.next_after_event_id
          for (const item of events.events) {
            if (eventLabels[item.type]) setEvent(eventLabels[item.type])
          }
        }
        const next = await api<Task>('/tasks/' + id, { signal: controller.signal })
        if (controller.signal.aborted) return
        setLatest((previous) => previous && !activeTask(previous.status) ? previous : next)
        setError(undefined)
        if (!activeTask(next.status)) return
      } catch (error) {
        if (controller.signal.aborted) return
        setError(errorMessage(error))
      }
      timer = window.setTimeout(poll, 1000)
    }
    void poll()
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [id, active, compact])

  async function cancel() {
    setCancelling(true)
    try {
      const next = await post<{ status: Task['status'] }>('/tasks/' + id + '/cancel')
      if (!activeTask(next.status)) setLatest(await api<Task>('/tasks/' + id))
    } catch (error) {
      setError(errorMessage(error))
    } finally {
      setCancelling(false)
    }
  }

  async function save() {
    setSaving(true)
    setError(undefined)
    try {
      await post<void>('/tasks/' + id + '/save')
      setSaved(true)
      onSaved?.()
    } catch (error) {
      setError(errorMessage(error))
    } finally {
      setSaving(false)
    }
  }

  if (compact) return <Card size="small" className="result-card">
    <Space orientation="vertical" className="full" size="small">
      {active && <Space>
        <Typography.Text type="secondary">正在查询…</Typography.Text>
        <Button size="small" type="text" onClick={() => void cancel()} loading={cancelling}>取消查询</Button>
      </Space>}
      {error && <Typography.Text type="secondary">{error}</Typography.Text>}
      {task.error && <Typography.Text>{task.error.detail}</Typography.Text>}
      {task.status === 'CANCELLED' && <Typography.Text type="secondary">这次查询已取消。</Typography.Text>}
      {task.status === 'COMPLETED' && <Output task={task} compact />}
    </Space>
  </Card>

  return <Card size="small" className="result-card">
    <Space orientation="vertical" className="full" size="middle">
      {task.status !== 'COMPLETED' && <Space wrap>
        <Tag color={task.status === 'FAILED' ? 'error' : active ? 'processing' : 'default'}>
          {statusLabel[task.status]}
        </Tag>
        {active && <Button size="small" onClick={() => void cancel()} loading={cancelling}>取消任务</Button>}
        {(task.status === 'FAILED' || task.status === 'CANCELLED') &&
          <Button size="small" onClick={() => onRetry(task.input)}>重新提交</Button>}
      </Space>}
      {active && event && <Typography.Text type="secondary">{event}</Typography.Text>}
      {error && <Alert type="warning" title={error} />}
      {task.error && <Alert type="error" title={task.error.detail} />}
      <Output task={task} compact={false} />
      {task.status === 'COMPLETED' && task.in_history !== undefined && (saved || task.in_history
        ? <Typography.Text type="secondary">已保存到历史</Typography.Text>
        : <Button loading={saving} onClick={() => void save()}>保存到历史</Button>)}
    </Space>
  </Card>
}
