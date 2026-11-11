import { expect, test, type Page } from '@playwright/test'

async function ask(page: Page, placeholder: string, question: string) {
  const input = page.getByPlaceholder(placeholder, { exact: false })
  await input.fill(question)
  await input.press('Enter')
}

test('business workflow and runtime overview on the V6 demo server', async ({ page }) => {
  await page.goto('/app/')
  await expect(page.getByText('本地演示')).toBeVisible()
  await page.getByRole('button', { name: '新建会话' }).click()
  await ask(page, '输入经营问题', '2026年1月Shanghai的订单笔数。')
  await expect(page.locator('.turn').last().getByText(/2026年1月上海.*27 笔/)).toBeVisible()
  await expect(page.locator('.turn').last().getByRole('table')).toHaveCount(0)
  await expect(page.getByText('生成结果解释', { exact: true })).toHaveCount(0)
  await ask(page, '输入经营问题', '改成Beijing')
  await expect(page.locator('.turn')).toHaveCount(2)
  await expect(page.locator('.turn').last().getByText(/2026年1月北京.*36 笔/)).toBeVisible()
  await page.reload()
  await expect(page.locator('.turn')).toHaveCount(2)
  const captures = process.env.AGENTSERVE_TEST_CAPTURE_DIR
  if (captures) await page.screenshot({ path: captures + '/agent-v6-workspace.png' })

  await page.getByRole('tab', { name: '业务规则' }).click()
  await ask(page, '成交营收的口径', '不加额外条件，哪些订单会进入成交营收？')
  await expect(page.getByRole('region', { name: '引用来源' })).toBeVisible()
  await page.getByRole('tab', { name: '指标查询' }).click()
  await page.getByRole('button', { name: '查询指标' }).click()
  await expect(page.getByRole('button', { name: '保存到历史' })).toBeVisible()
  await expect(page.getByText('单位 分', { exact: false })).toBeVisible()

  await page.getByRole('tab', { name: '经营简报' }).click()
  await page.getByRole('button', { name: '生成简报' }).click()
  await expect(page.getByRole('img', { name: '月度营收和订单量图' })).toBeVisible()
  for (const name of ['monthly.csv', 'report.md', 'chart.svg']) {
    const download = page.waitForEvent('download')
    await page.getByRole('link', { name: '下载 ' + name }).click()
    expect((await download).suggestedFilename()).toBe(name)
  }
  await page.getByText('运行概览', { exact: true }).click()
  await expect(page.getByText('任务完成耗时（JCT）')).toBeVisible()
  await expect(page.getByRole('columnheader', { name: '模型输出速率（tokens/s）' })).toBeVisible()
  await expect(page.getByText('最近失败', { exact: true })).toBeVisible()
  if (captures) await page.screenshot({ path: captures + '/agent-v6-admin.png', fullPage: true })
  await page.getByRole('button', { name: '清空记录' }).click()
  await page.getByRole('button', { name: /^清\s*空$/ }).click()
  await expect(page.getByText(/清空于/)).toBeVisible()
})

test('city discovery, typo guidance and short answers in chat', async ({ page }) => {
  await page.goto('/app/')
  await page.getByRole('button', { name: '新建会话' }).click()
  await ask(page, '输入经营问题', '有Beijin的订单吗')
  await expect(page.locator('.turn').last().getByText(/没有「Beijin」。你是想查北京/)).toBeVisible()
  await page.getByRole('button', { name: '查看可查城市' }).click()
  await expect(page.locator('.turn').last().getByText(/可以查这些城市：.*成都.*武汉/)).toBeVisible()
  await ask(page, '输入经营问题', '那北京呢')
  await expect(page.locator('.turn').last().getByText('有，北京（Beijing）有符合条件的订单。')).toBeVisible()
  await expect(page.locator('.turn').last().getByText('已完成', { exact: true })).toHaveCount(0)
  await expect(page.locator('.turn').last().getByRole('table')).toHaveCount(0)
  await ask(page, '输入经营问题', '北京有多少订单？')
  await expect(page.locator('.turn').last().getByText(/北京（Beijing）有 173 笔/)).toBeVisible()
  await ask(page, '输入经营问题', '2026年3月有多少订单深圳和上海')
  await expect(page.locator('.turn').last().getByText(/2026年3月.*深圳.*29 笔.*上海.*35 笔/)).toBeVisible()
  await ask(page, '输入经营问题', '总共有多少个城市有订单详情？')
  await expect(page.locator('.turn').last().getByText(/共有 6 个城市有订单/)).toBeVisible()
})

test('briefs explain missing cities and empty periods without a chart', async ({ page }) => {
  await page.goto('/app/#analysis')
  await page.getByRole('combobox', { name: '城市（可选）' }).click()
  await expect(page.getByText('上海（Shanghai）', { exact: true }).first()).toBeVisible()
  for (const [city, start, end, expected] of [
    ['Chongqin', '2026-01-01', '2026-04-01', '当前数据里没有「Chongqin」'],
    ['上海', '2026-07-01', '2026-08-01', '所选时段没有已完成订单'],
  ]) {
    const response = await page.request.post('/tasks', {
      headers: { Origin: new URL(page.url()).origin },
      data: { database_id: 'business_orders_v1', mode: 'analysis', analysis: { city, start, end } },
    })
    expect(response.ok()).toBeTruthy()
    await page.goto('/app/#analysis/' + (await response.json()).task_id)
    await expect(page.getByText(expected, { exact: false })).toBeVisible()
    await expect(page.getByRole('img', { name: '月度营收和订单量图' })).toHaveCount(0)
  }
})

test('month follow-up and both history context menus', async ({ page }) => {
  await page.goto('/app/')
  await page.getByRole('button', { name: '新建会话' }).click()
  await ask(page, '输入经营问题', '深圳的订购单数呢 有吗')
  await expect(page.locator('.turn').last().getByText(/深圳（Shenzhen）有 164 笔/)).toBeVisible()
  await ask(page, '输入经营问题', '从哪月到哪月')
  await expect(page.locator('.turn').last().getByText(/实际覆盖 2026-01 至 2026-06/)).toBeVisible()
  const sidebar = page.getByRole('complementary', { name: '任务历史' })
  const session = sidebar.getByText('深圳的订购单数呢 有吗', { exact: true })
  const origin = new URL(page.url()).origin
  const resource = await page.request.post('/tasks', {
    headers: { Origin: origin },
    data: { database_id: 'business_orders_v1', mode: 'metric',
      metric: { name: '完成订单量', start: '2026-01-01', end: '2026-02-01' } },
  })
  expect(resource.ok()).toBeTruthy()
  const taskId = (await resource.json()).task_id
  for (const [item, name] of [
    [session, '深圳查数记录'],
    [sidebar.getByText(/完成订单量 ·/), '一月完成订单'],
  ] as const) {
    if (name === '一月完成订单') await page.getByRole('tab', { name: '指标查询' }).click()
    await item.click({ button: 'right' })
    await page.getByRole('menuitem', { name: '重命名' }).click()
    await page.getByRole('textbox', { name: '历史名称' }).fill(name)
    await page.getByRole('button', { name: /^保\s*存$/ }).click()
    const renamed = sidebar.getByText(name, { exact: name === '深圳查数记录' })
    await expect(renamed).toBeVisible()
    await renamed.click({ button: 'right' })
    await page.getByRole('menuitem', { name: '从历史删除' }).click()
    await page.getByRole('button', { name: /^删\s*除$/ }).click()
    await expect(renamed).toHaveCount(0)
  }
  expect((await page.request.get('/tasks/' + taskId)).status()).toBe(404)
})

test('uncertain submission, cancellation and explicit retry use the correct keys', async ({ page }) => {
  const keys: string[] = []
  let state = 'RUNNING'
  await page.route('**/tasks', async (route) => {
    keys.push(route.request().headers()['idempotency-key'])
    if (keys.length === 1) return route.abort()
    const id = keys.length === 2 ? 'cancel-me' : 'retry-of-cancelled'
    return route.fulfill({ status: 202, json: { task_id: id } })
  })
  await page.route(/\/tasks\/(cancel-me|retry-of-cancelled)$/, (route) => route.fulfill({ json: {
    task_id: route.request().url().split('/').at(-1), status: state, error: null,
    input: { database_id: 'business_orders_v1', mode: 'knowledge', question: '演示取消' }, report: null,
  } }))
  await page.route('**/tasks/cancel-me/events?*', (route) => route.fulfill({
    json: { events: [], next_after_event_id: 0 },
  }))
  await page.route('**/tasks/cancel-me/cancel', (route) => {
    state = 'CANCELLED'
    return route.fulfill({ status: 202, json: { status: state, changed: true } })
  })
  await page.goto('/app/#knowledge')
  await ask(page, '成交营收的口径', '演示取消')
  await page.getByRole('button', { name: '确认原提交' }).click()
  await page.getByRole('button', { name: '取消任务' }).click()
  await expect(page.getByText('已取消', { exact: true })).toBeVisible()
  state = 'FAILED'
  await page.getByRole('button', { name: '重新提交' }).click()
  await expect(page).toHaveURL(/#knowledge\/retry-of-cancelled$/)
  expect(keys).toHaveLength(3)
  expect(keys[0]).toBe(keys[1])
  expect(keys[2]).not.toBe(keys[1])
})

test('switching conversations ignores a late history response', async ({ page }) => {
  await page.route('**/sessions?*', (route) => route.fulfill({ json: [
    { session_id: 'old', title: '旧会话', created_at: 1 },
    { session_id: 'new', title: '新会话', created_at: 2 },
  ] }))
  let release: () => void = () => {}
  const late = new Promise<void>((resolve) => { release = resolve })
  await page.route(/\/sessions\/(old|new)\/history\?/, async (route) => {
    const old = route.request().url().includes('/old/')
    if (old) await late
    await route.fulfill({ json: {
      turns: [{
        task_id: old ? 'old-task' : 'new-task', status: 'COMPLETED', error: null, report: null,
        input: { question: old ? '过时的历史内容' : '当前的历史内容' },
      }],
      next_after_turn: 1,
    } }).catch(() => undefined)
  })
  await page.goto('/app/#sql/old')
  try {
    await page.getByText('新会话', { exact: true }).click()
    await expect(page.getByText('当前的历史内容')).toBeVisible()
  } finally {
    release()
  }
  await expect(page.getByText('过时的历史内容')).toHaveCount(0)
})

test('onboarding opens scope without a task and saved reports can be removed together', async ({ page }) => {
  let submissions = 0
  page.on('request', (request) => {
    if (request.method() === 'POST' && new URL(request.url()).pathname === '/tasks') submissions += 1
  })
  await page.goto('/app/')
  await expect(page.getByText('使用指南', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: '开始查数' }).click()
  await expect(page.getByPlaceholder('输入经营问题', { exact: false })).toBeVisible()
  await expect(page.getByText('使用指南', { exact: true })).toHaveCount(0)
  await page.getByRole('tab', { name: '业务规则' }).click()
  await expect(page.getByText('订单库可以查什么')).toBeVisible()
  await expect(page.getByText(/植物园、景点/)).toBeVisible()
  await page.getByRole('button', { name: '我能查什么' }).click()
  await ask(page, '成交营收的口径', '有什么可以查询的？')
  await expect(page.getByText('业务知识库有什么')).toBeVisible()
  expect(submissions).toBe(0)

  await page.getByRole('tab', { name: '经营简报' }).click()
  const sidebar = page.getByRole('complementary', { name: '任务历史' })
  await expect(sidebar.getByText('简报历史')).toBeVisible()
  await expect(sidebar.getByText('指标历史')).toHaveCount(0)
  const labels = ['2026-01 至 2026-04 · 全部城市', '2026-01 至 2026-05 · 全部城市']
  for (const [index, month] of [4, 5].entries()) {
    await page.getByRole('combobox', { name: '结束月份', exact: true }).click()
    await expect(page.getByRole('option', { name: '11月', exact: true })).not.toHaveAttribute('aria-selected', 'true')
    await page.getByRole('option', { name: `${month}月`, exact: true }).click()
    const queued = page.waitForRequest((request) => request.method() === 'POST' && new URL(request.url()).pathname === '/tasks')
    await page.getByRole('button', { name: '生成简报' }).click()
    const submitted = (await queued).postDataJSON()
    expect(submitted.analysis.end).toBe(`2026-0${month + 1}-01`)
    expect(submitted.save_to_history).toBe(false)
    await expect(page.getByRole('button', { name: '保存到历史' })).toBeVisible()
    await expect(sidebar.getByText(labels[index], { exact: true })).toHaveCount(0)
    await expect(page.locator('.task-id')).toHaveCount(0)
    await page.getByRole('button', { name: '保存到历史' }).click()
    await expect(sidebar.getByText(labels[index], { exact: true })).toBeVisible()
  }
  await page.getByRole('button', { name: '批量删除' }).click()
  for (const label of labels) await page.getByRole('checkbox', { name: label }).check()
  await page.getByRole('button', { name: '删除所选（2）' }).click()
  await page.getByRole('dialog', { name: '批量删除历史' }).getByRole('button', { name: /^删\s*除$/ }).click()
  for (const label of labels) await expect(sidebar.getByText(label, { exact: true })).toHaveCount(0)
})

test('revenue totals keep the inclusive month span after city and date corrections', async ({ page }) => {
  await page.goto('/app/')
  await page.getByRole('button', { name: '新建会话' }).click()
  const cases = [
    ['上海2026年1月到3月的总成交额是多少', /17,?200\.75/],
    ['深圳呢', /16,?502(?:\.00)?/],
    ['是1月到3月', /16,?502(?:\.00)?/],
    ['为什么月份增加了成交额变少了？', /16,?502(?:\.00)?/],
    ['我不是说深圳从1月到3月的总成交额吗', /16,?502(?:\.00)?/],
  ] as const
  for (const [index, [question, amount]] of cases.entries()) {
    await ask(page, '输入经营问题', question)
    await expect(page.locator('.turn')).toHaveCount(index + 1)
    const answer = page.locator('.turn').last().locator('.result-card')
    await expect(answer).toContainText(amount)
    await expect(answer).toContainText('2026年1月至3月')
  }
})

test('different calendar wording preserves city, boundaries and empty results', async ({ page }) => {
  const conversations = [
    [
      ['查一下杭州2026年四月至五月的成交额', /12,?863\.25/, '2026年4月至5月'],
      ['换成上海', /8,?577\.50/, '2026年4月至5月'],
      ['只看6-7月', /4,?846\.25/, '2026年6月至7月'],
    ],
    [
      ['深圳2025年12月至2026年2月的成交额是多少', /10,?984(?:\.00)?/, '2025年12月至2026年2月'],
      ['只看3月', /5,?518(?:\.00)?/, '2026年3月'],
      ['改看7–8月', /没有可汇总|没有已完成|(?:成交额|交易额)[^0-9]{0,8}0(?:\.00)?\s*元/, ''],
    ],
  ] as const
  for (const conversation of conversations) {
    await page.goto('/app/')
    await page.getByRole('button', { name: '新建会话' }).click()
    for (const [index, [question, amount, period]] of conversation.entries()) {
      await ask(page, '输入经营问题', question)
      await expect(page.locator('.turn')).toHaveCount(index + 1)
      const answer = page.locator('.turn').last().locator('.result-card')
      await expect(answer).toContainText(amount)
      if (period) await expect(answer).toContainText(period)
    }
  }
})
