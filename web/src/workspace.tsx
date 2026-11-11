import { createContext, useContext, useCallback, useEffect, useRef, useState } from 'react'
import { Bubble, Conversations, Sender } from '@ant-design/x'
import { Alert, Button, Card, Checkbox, Dropdown, Empty, Form, Input as Field, Modal, Select, Space, Spin, Tabs, Typography } from 'antd'
import { activeTask, api, databaseId as defaultDatabase, errorMessage, modeLabel, monthIndex, post, shiftMonth, statusLabel } from './api'
import type { DateWindow, Input, Mode, RecentTask, Session, Task } from './api'
import { useJson, useLoad, useSubmission } from './hooks'
import { TaskPanel } from './task'

type Dataset = {
  database_id: string; label: string; first_date: string; last_date: string; currency: string
  modes: Mode[]; data_description: string; rule_description: string
  query_examples: string[]; rule_examples: string[]
}
const DatasetContext = createContext<Dataset | null>(null)
function useDataset() {
  const dataset = useContext(DatasetContext)
  if (!dataset) throw new Error('Dataset context unavailable')
  return dataset
}
const pageSize = 30
const readHash = () => location.hash.slice(1) || 'sql'
const readDataset = (hash = readHash()) => hash.includes('~') ? hash.split('~', 1)[0] : defaultDatabase
const readRoute = (hash = readHash()) => hash.slice(hash.indexOf('~') + 1)
const go = (path: string, database = readDataset()) => {
  location.hash = database === defaultDatabase ? path : database + '~' + path
}
const isGuideQuestion = (text: string) => /^(?:(?:你好|请问)[，,\s]*)?(?:(?:这里|(?:这个)?系统|我|你)?(?:能|可以)(?:问|查(?:询)?)(?:什么|哪些)(?:数据|内容|问题)?|有什么(?:数据|内容|问题)?可以(?:查(?:询)?|问)的?|(?:这个)?(?:数据(?:库)?|知识库|RAG)(?:里|里面)?有(?:什么|哪些)(?:数据|内容|资料)?)[吗呢？?。！!\s]*$/i.test(text.trim())
function OrderGuideCities() {
  const cities = useJson<{ value: string; label: string }[]>('/catalog/cities')
  return cities.data?.length ? <Typography.Text type="secondary">当前城市 · {cities.data.map((city) => city.label).join('、')}</Typography.Text> : null
}

function WorkbenchGuide() {
  const dataset = useDataset()
  return <Card className="guide-card" title="使用指南">
    <Space orientation="vertical" className="full" size="middle">
      <div><Typography.Text strong>{dataset.database_id === defaultDatabase ? '订单库可以查什么' : '当前数据集可以查什么'}</Typography.Text>
        <div>{dataset.data_description}</div>
        <div>数据覆盖 {dataset.first_date} 至 {dataset.last_date} · 币种 {dataset.currency}</div>
        {dataset.database_id === defaultDatabase && <OrderGuideCities />}
      </div>
      <div><Typography.Text strong>业务知识库有什么</Typography.Text>
        <div>{dataset.rule_description}</div>
      </div>
      <div><Typography.Text strong>使用边界</Typography.Text>
        <div>仅查库内经营数据和已收录的业务资料。植物园、景点等本地生活信息超出查询范围。</div>
      </div>
      <Space wrap>{dataset.query_examples.map((text) =>
        <Typography.Text key={text} type="secondary">{text}</Typography.Text>)}</Space>
    </Space>
  </Card>
}

function LoadError({ error, retry }: { error: string; retry: () => void }) {
  return <Alert type="error" title={error} action={<Button onClick={retry}>重试读取</Button>} />
}

function SubmissionNotice({ submission }: { submission: ReturnType<typeof useSubmission> }) {
  if (!submission.error && !submission.pending) return null
  if (submission.pending && !submission.busy) return <Alert type="warning"
    title="提交结果暂未确认"
    description="确认时沿用原提交编号，服务端只会创建一次任务。"
    action={<Button onClick={() => void submission.submit()}>确认原提交</Button>} />
  return submission.error ? <Alert type="error" title={submission.error} /> : null
}

function ChatSession({ id, identity, onChange }: { id: string; identity: string; onChange: () => void }) {
  const dataset = useDataset()
  const databaseId = dataset.database_id
  const [version, setVersion] = useState(0)
  const [message, setMessage] = useState('')
  const bottom = useRef<HTMLDivElement>(null)
  const load = useCallback(async (signal: AbortSignal) => {
    const turns: Task[] = []
    let cursor = 0
    while (true) {
      const page = await api<{ turns: Task[]; next_after_turn: number }>(
        '/sessions/' + encodeURIComponent(id) + '/history?limit=100&after_turn=' + cursor,
        { signal },
      )
      turns.push(...page.turns)
      if (page.turns.length < 100) return turns
      cursor = page.next_after_turn
    }
  }, [id])
  const { data, error, reload } = useLoad(load, version)
  const count = data?.length
  const submission = useSubmission(identity + ':sql/' + id, () => {
    setVersion((value) => value + 1)
    onChange()
  })

  useEffect(() => { if (count) bottom.current?.scrollIntoView({ block: 'end' }) }, [count])

  async function send(question: string) {
    const text = question.trim()
    if (!text || text.length > 4000) return
    if (isGuideQuestion(text)) {
      setMessage('')
      go('knowledge')
      return
    }
    if (await submission.submit({
      database_id: databaseId, session_id: id, question: text,
    })) setMessage('')
  }

  return <section className="chat-session" aria-label="多轮查数会话">
    <div className="turns">
      {error && <LoadError error={error} retry={reload} />}
      {!data && !error && <Spin />}
      {data?.map((task) => <div className="turn" key={task.task_id}>
        <Bubble placement="end" content={task.input.question} />
        <TaskPanel initial={task} compact onRetry={(input) => void submission.submit(input)} />
      </div>)}
      {data?.length === 0 && <Empty description={dataset.query_examples[0] ?? '输入查询问题'} />}
      <div ref={bottom} />
    </div>
    <div className="composer">
      <SubmissionNotice submission={submission} />
      {databaseId === defaultDatabase && <Button type="text" size="small" onClick={() => void send('有哪些城市的订单？')}
        disabled={Boolean(submission.pending)}>查看可查城市</Button>}
      <Sender value={message} onChange={setMessage} onSubmit={(value) => void send(value)}
        disabled={Boolean(submission.pending)} placeholder="输入经营问题，按 Enter 提交"
        autoSize={{ minRows: 2, maxRows: 5 }}
      />
    </div>
  </section>
}

type MetricFields = DateWindow & { name?: string; group_by?: 'city' | 'month' }
const monthFormat = /^[0-9]{4}-(?:0[1-9]|1[0-2])$/
const monthOptions = Array.from({ length: 12 }, (_, index) => ({
  value: String(index + 1).padStart(2, '0'), label: `${index + 1}月`,
}))

function MonthInput({ value = '', onChange, id, label }: {
  value?: string; onChange?: (value: string) => void; id?: string; label: string
}) {
  const [year, month] = value.split('-')
  return <div className="month-field">
    <Field type="number" min={1} max={9998} aria-label={label + '年份'} placeholder="年份" value={year}
      onChange={(event) => onChange?.(`${event.target.value}-${month ?? ''}`)} />
    <Select id={id} aria-label={label} value={month || undefined} options={monthOptions} placeholder="月份" virtual={false}
      onChange={(selected) => onChange?.(`${year}-${selected}`)} />
  </div>
}

function CityField() {
  const cities = useJson<{ value: string; label: string }[]>('/catalog/cities')
  return <>
    <Form.Item name="city" label="城市（可选）">
      <Select showSearch allowClear optionFilterProp="label" options={cities.data}
        loading={!cities.data && !cities.error} placeholder="不选则查询所有城市" />
    </Form.Item>
    {cities.error && <LoadError error="城市列表暂时无法读取。" retry={cities.reload} />}
  </>
}

function TaskForm({ mode, submission, input }: {
  mode: Exclude<Mode, 'sql'>; submission: ReturnType<typeof useSubmission>; input?: Input
}) {
  const dataset = useDataset()
  const databaseId = dataset.database_id
  const [question, setQuestion] = useState('')
  const [showGuide, setShowGuide] = useState(mode === 'knowledge')
  const monthly = mode === 'analysis'
  const initial = monthly ? (input?.analysis ? {
    ...input.analysis, start: input.analysis.start.slice(0, 7), end: shiftMonth(input.analysis.end.slice(0, 7), -1),
  } : { start: '2026-01', end: '2026-03' }) : input?.metric ?? {
    name: 'settled_revenue', start: '2026-01-01', end: '2026-04-01',
  }
  const validMonth = (value: unknown): value is string => typeof value === 'string' && monthFormat.test(value)
    && Number(value.slice(0, 4)) > 0 && Number(value.slice(0, 4)) < 9999
  return <Card title={modeLabel[mode]} className="entry-card">
    <Space orientation="vertical" className="full">
      <SubmissionNotice submission={submission} />
      {mode === 'knowledge' ? <>
        <Typography.Paragraph>用来确认当前数据集的统计口径和字段含义。</Typography.Paragraph>
        <Space wrap>{dataset.rule_examples.map((text) => <Button key={text} size="small" onClick={() => setQuestion(text)}>{text}</Button>)}</Space>
        <Button type="link" onClick={() => setShowGuide((value) => !value)}>我能查什么</Button>
        {showGuide && <WorkbenchGuide />}
        <Sender
        value={question} onChange={setQuestion} placeholder={databaseId === defaultDatabase ? '例如，成交营收的口径是什么？' : dataset.rule_examples[0] ?? '输入业务规则问题'}
        disabled={Boolean(submission.pending)}
        onSubmit={(value) => {
          const text = value.trim()
          if (isGuideQuestion(text)) {
            setQuestion('')
            setShowGuide(true)
            return
          }
          if (text && text.length <= 4000) {
            void submission.submit({ database_id: databaseId, mode, question: text })
          }
        }}
        />
      </> : <Form<MetricFields> layout="vertical"
        initialValues={initial}
        onFinish={({ name, start, end, city, group_by }) => {
          const range = {
            start: monthly ? start + '-01' : start,
            end: monthly ? shiftMonth(end, 1) + '-01' : end,
            ...(city?.trim() ? { city: city.trim() } : {}),
          }
          void submission.submit({
            database_id: databaseId, mode, save_to_history: false, ...(monthly ? { analysis: range } : {
              metric: { ...range, name: name!, ...(group_by ? { group_by } : {}) },
            }),
          })
        }}>
        {!monthly && <Form.Item name="name" label="指标" rules={[{ required: true }]}>
          <Select options={[
            { label: '成交营收', value: 'settled_revenue' },
            { label: '完成订单量', value: 'completed_orders' },
          ]} />
        </Form.Item>}
        <div className="date-fields">
          {monthly ? <>
            <Form.Item name="start" label="开始月份" rules={[{ required: true }, {
              validator: (_, value: unknown) => validMonth(value) ? Promise.resolve() : Promise.reject(new Error('请选择有效年份和月份')),
            }]}><MonthInput label="开始月份" /></Form.Item>
            <Form.Item name="end" label="结束月份（含）" dependencies={['start']} rules={[{ required: true },
              ({ getFieldValue }) => ({ validator: (_, value: unknown) => {
                const start: unknown = getFieldValue('start')
                if (!validMonth(value)) return Promise.reject(new Error('请选择有效年份和月份'))
                if (!validMonth(start)) return Promise.resolve()
                const months = monthIndex(value) - monthIndex(start) + 1
                return months >= 1 && months <= 12 ? Promise.resolve() : Promise.reject(new Error('请选择连续 1 至 12 个月'))
              } }),
            ]}><MonthInput label="结束月份" /></Form.Item>
          </> : <>
            <Form.Item name="start" label="开始日期" rules={[{ required: true }]}><Field type="date" /></Form.Item>
            <Form.Item name="end" label="结束日期（不含）" rules={[{ required: true }]}><Field type="date" /></Form.Item>
          </>}
        </div>
        <CityField />
        {!monthly && <Form.Item name="group_by" label="分组（可选）"><Select allowClear options={[
          { value: 'city', label: '按城市' }, { value: 'month', label: '按月份' },
        ]} /></Form.Item>}
        <Button type="primary" htmlType="submit" loading={submission.busy}
          disabled={Boolean(submission.pending)}>{monthly ? '生成简报' : '查询指标'}</Button>
      </Form>}
    </Space>
  </Card>
}

function Standalone({ mode, id, identity, onChange }: {
  mode: Exclude<Mode, 'sql'>; id?: string; identity: string; onChange: () => void
}) {
  const { database_id } = useDataset()
  const route = id ? mode + '/' + id : mode
  const submission = useSubmission(identity + ':' + database_id + ':' + route, (taskId) => {
    onChange()
    if (readDataset() === database_id && readRoute() === route) go(mode + '/' + taskId)
  })
  return id ? <StandaloneResult id={id} mode={mode} submission={submission} onTerminal={onChange} /> :
    <TaskForm mode={mode} submission={submission} />
}

function StandaloneResult({ id, mode, submission, onTerminal }: {
  id: string; mode: Exclude<Mode, 'sql'>; submission: ReturnType<typeof useSubmission>; onTerminal: () => void
}) {
  const { data, error, reload } = useJson<Task>('/tasks/' + encodeURIComponent(id))
  return <Space orientation="vertical" className="full" size="middle">
    {mode === 'knowledge' && <div className="section-title">
      <Typography.Title level={3}>{modeLabel[mode]}</Typography.Title>
      <Button onClick={() => go(mode)}>继续提问</Button>
    </div>}
    {mode === 'knowledge' && <SubmissionNotice submission={submission} />}
    {error && <LoadError error={error} retry={reload} />}
    {!data && !error && <Spin />}
    {mode !== 'knowledge' && data && <TaskForm mode={mode} submission={submission} input={data.input} />}
    {data?.input.question && <Bubble placement="end" content={data.input.question} />}
    {data && <TaskPanel initial={data} onRetry={(input) => void submission.submit(input)}
      onTerminal={onTerminal} onSaved={onTerminal} />}
  </Space>
}

type HistoryItem = { id: string; route: string; title: string; active: boolean; status?: RecentTask['status'] }
type HistoryAction = { kind: 'rename' | 'delete'; item: HistoryItem } | { kind: 'batch'; ids: string[] }
const historyTitleByMode: Record<Mode, string> = {
  sql: '查数会话', knowledge: '规则问答历史', metric: '指标历史', analysis: '简报历史',
}

function HistorySidebar({ mode, route, revision, onChange, onNewSession, creating, creationError }: {
  mode: Mode; route: string; revision: number; onChange: () => void; onNewSession: () => void;
  creating: boolean; creationError?: string
}) {
  const [page, setPage] = useState(0)
  const [selecting, setSelecting] = useState(false)
  const [selected, setSelected] = useState<string[]>([])
  const [historyAction, setHistoryAction] = useState<HistoryAction>()
  const [historyTitle, setHistoryTitle] = useState('')
  const [historyError, setHistoryError] = useState<string>()
  const [savingHistory, setSavingHistory] = useState(false)
  const { database_id } = useDataset()
  const sessions = mode === 'sql'
  const path = sessions ? '/sessions' : '/tasks'
  const history = useJson<(Session | RecentTask)[]>(`${path}?limit=${pageSize}&offset=${page * pageSize}&database_id=${encodeURIComponent(database_id)}`
    + (sessions ? '' : `&mode=${mode}`), revision)
  const items: HistoryItem[] = (history.data ?? []).map((item) => 'session_id' in item ? {
    id: item.session_id, route: 'sql/' + item.session_id, active: Boolean(item.active),
    title: item.title ?? '新会话',
  } : {
    id: item.task_id, route: item.mode + '/' + item.task_id, title: item.title ?? modeLabel[item.mode],
    status: item.status, active: activeTask(item.status),
  })
  const selectable = items.filter((item) => !item.active).map((item) => item.id)
  const selectedIds = selected.filter((id) => selectable.includes(id))

  function changePage(value: number) {
    setPage(value)
    setSelected([])
  }

  function historyLabel(item: HistoryItem, text = item.title) {
    return <Dropdown trigger={['contextMenu']} menu={{
      items: [{ key: 'rename', label: '重命名' }, { key: 'delete', label: '从历史删除', danger: true }],
      onClick: ({ key, domEvent }) => {
        domEvent.stopPropagation()
        setHistoryTitle(item.title.slice(0, 80))
        setHistoryError(undefined)
        setHistoryAction({ item, kind: key === 'rename' ? 'rename' : 'delete' })
      },
    }}>
      <span className="history-label" title={text}>{text}</span>
    </Dropdown>
  }

  async function saveHistory() {
    if (!historyAction || savingHistory) return
    const action = historyAction
    setSavingHistory(true)
    setHistoryError(undefined)
    try {
      if (action.kind === 'batch') {
        await post<void>(path + '/batch-delete', { ids: action.ids })
        if (action.ids.includes(readRoute().split('/')[1])) go(mode)
        changePage(0)
        setSelecting(false)
      } else {
        await api<void>(path + '/' + encodeURIComponent(action.item.id), action.kind === 'rename' ? {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: historyTitle.trim() }),
        } : { method: 'DELETE' })
        if (action.kind === 'delete') {
          if (items.length === 1) changePage(0)
          if (readRoute() === action.item.route) go(mode)
        }
      }
      onChange()
      setHistoryAction(undefined)
    } catch (error) {
      setHistoryError(errorMessage(error))
    } finally {
      setSavingHistory(false)
    }
  }

  return <aside className="sidebar" aria-label="任务历史">
      {sessions && <Button block type="primary" loading={creating} onClick={() => { changePage(0); onNewSession() }}>新建会话</Button>}
      {creationError && <Alert type="error" title={creationError} />}
      <div className="history-heading">
        <Typography.Title level={5}>{historyTitleByMode[mode]}</Typography.Title>
        {!selecting && items.length > 0 && <Button size="small" type="text" onClick={() => setSelecting(true)}>批量删除</Button>}
      </div>
      {history.error && <LoadError error={history.error} retry={history.reload} />}
      {history.data?.length === 0 && <Typography.Text type="secondary">{sessions ? '暂无会话' : '暂无保存的历史'}</Typography.Text>}
      {selecting ? <>
        <Space wrap size="small" className="history-actions">
          <Button size="small" disabled={!selectable.length} onClick={() => setSelected(selectable)}>全选本页</Button>
          <Button size="small" type="text" onClick={() => { setSelected([]); setSelecting(false) }}>退出批量</Button>
          <Button size="small" danger disabled={!selectedIds.length} onClick={() => {
            setHistoryError(undefined)
            setHistoryAction({ kind: 'batch', ids: selectedIds })
          }}>删除所选（{selectedIds.length}）</Button>
        </Space>
        <div className="history-checklist">{items.map((item) => <Checkbox key={item.id} disabled={item.active}
          checked={selectedIds.includes(item.id)} onChange={(event) => setSelected((old) => event.target.checked
            ? [...old, item.id] : old.filter((id) => id !== item.id))}>
          <span title={item.title}>{item.title}</span>{item.active && ' · 处理中'}
        </Checkbox>)}</div>
      </> : <Conversations activeKey={route} items={items.map((item) => ({
        key: item.route,
        label: historyLabel(item, item.title + (item.status && item.status !== 'COMPLETED' ? ' · ' + statusLabel[item.status] : '')),
      }))} onActiveChange={(path) => go(path)} />}
      <PageButtons page={page} setPage={changePage} hasNext={history.data?.length === pageSize} />
    {historyAction && <Modal open
      title={historyAction.kind === 'rename' ? '重命名' : historyAction.kind === 'batch' ? '批量删除历史' : '从历史删除'}
      okText={historyAction.kind === 'rename' ? '保存' : '删除'}
      cancelText="取消"
      okButtonProps={{ danger: historyAction.kind !== 'rename',
        disabled: historyAction.kind === 'rename' && !historyTitle.trim() }}
      confirmLoading={savingHistory}
      onOk={() => void saveHistory()}
      onCancel={() => { if (!savingHistory) setHistoryAction(undefined) }}>
      {historyAction.kind === 'rename'
        ? <Field aria-label="历史名称" value={historyTitle} maxLength={80}
          onChange={(event) => setHistoryTitle(event.target.value)} />
        : <Typography.Paragraph>{historyAction.kind === 'batch' ? `所选 ${historyAction.ids.length} 条记录`
          : `「${historyAction.item.title}」`}及其结果将从工作台历史移除。</Typography.Paragraph>}
      {historyError && <Alert type="error" title={historyError} />}
    </Modal>}
  </aside>
}

export function Workspace({ identity }: { identity: string }) {
  const [hash, setHash] = useState(readHash)
  const catalog = useJson<Dataset[]>('/catalog/datasets')
  useEffect(() => {
    const change = () => setHash(readHash())
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  if (catalog.error) return <LoadError error={catalog.error} retry={catalog.reload} />
  if (!catalog.data) return <Spin />
  const dataset = catalog.data.find((entry) => entry.database_id === readDataset(hash)) ?? catalog.data[0]
  if (!dataset) return <Empty description="当前没有可查的数据集" />
  const route = readDataset(hash) === dataset.database_id ? readRoute(hash) : 'sql'
  return <DatasetContext.Provider value={dataset}>
    <DatasetWorkspace key={dataset.database_id} route={route} identity={identity} catalog={catalog.data} />
  </DatasetContext.Provider>
}

function DatasetWorkspace({ route, identity, catalog }: { route: string; identity: string; catalog: Dataset[] }) {
  const dataset = useDataset()
  const databaseId = dataset.database_id
  const [revision, setRevision] = useState(0)
  const [creationError, setCreationError] = useState<string>()
  const [creating, setCreating] = useState(false)
  const requestedMode = route.split('/')[0] as Mode
  const mode = dataset.modes.includes(requestedMode) ? requestedMode : 'sql'
  const id = mode === requestedMode ? route.split('/')[1] : undefined
  const onChange = useCallback(() => setRevision((value) => value + 1), [])

  async function newSession() {
    setCreating(true)
    setCreationError(undefined)
    try {
      const session = await post<Session>('/sessions', { database_id: databaseId })
      onChange()
      if (readDataset() === databaseId || catalog.length === 1) go('sql/' + session.session_id, databaseId)
    } catch (error) {
      setCreationError(errorMessage(error))
    } finally {
      setCreating(false)
    }
  }

  return <main className="workspace">
    <HistorySidebar key={mode} mode={mode} route={route} revision={revision} onChange={onChange}
      creating={creating} creationError={mode === 'sql' ? creationError : undefined} onNewSession={() => void newSession()} />
    <section className="workspace-content">
      {catalog.length > 1 && <Select aria-label="数据集" value={databaseId}
        options={catalog.map((entry) => ({ value: entry.database_id, label: entry.label }))}
        onChange={(id) => go('sql', id)} />}
      <Tabs activeKey={mode} onChange={(key) => go(key, databaseId)}
        items={dataset.modes.map((key) => ({ key, label: modeLabel[key] }))} />
      {mode === 'sql' ? (id ? <ChatSession key={id} id={id} identity={identity} onChange={onChange} /> :
        <Empty description={dataset.query_examples[0] ?? '新建会话后开始查数'}>
          <Button type="primary" loading={creating} onClick={() => void newSession()}>开始查数</Button>
        </Empty>) :
        <Standalone key={route} mode={mode} id={id} identity={identity} onChange={onChange} />}
    </section>
  </main>
}

function PageButtons({ page, setPage, hasNext }: {
  page: number; setPage: (page: number) => void; hasNext: boolean
}) {
  if (!page && !hasNext) return null
  return <Space className="pages">
    <Button size="small" disabled={!page} onClick={() => setPage(page - 1)}>上一页</Button>
    <Button size="small" disabled={!hasNext} onClick={() => setPage(page + 1)}>下一页</Button>
  </Space>
}
