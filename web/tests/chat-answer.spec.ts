import { expect, test } from '@playwright/test'
import type { Task } from '../src/api'

test('presents saved SQL results as Chinese business answers without tables', async ({ page }) => {
  const databaseId = 'online_retail_ii'
  const values: [string, string[], unknown[][], boolean?][] = [
    ['USA 在数据范围内共有多少个交易，销售额多少？', ['transaction_count', 'total_sales_gbp'], [[25, 8366.86]]],
    ['Japan 在日期范围内取消了多少个订单？', ['cancellation_count'], [[23]]],
    ['取消发票笔数的金额是多少？', ['cancellation_amount_gbp'], [[3361.81]]],
    ['按国家列出销售额。', ['country', 'sales_gbp'], [['USA', 8366.86], ['Japan', null]], true],
    ['这段时间有销售明细吗？', ['sales_gbp'], [[null]]],
  ]
  const turns = values.map(([question, columns, rows, truncated = false], index): Task => ({
    task_id: 'query-' + index, turn_no: index + 1, status: 'COMPLETED', error: null,
    input: { database_id: databaseId, session_id: 'history', question },
    report: { answer_source: 'tool_values', answer: '原始 SQL 结果', query_result: { columns, rows, truncated }, citations: [] },
  }))
  await page.route('**/auth/me', (route) => route.fulfill({ json: { mode: 'demo', user: null } }))
  await page.route('**/catalog/datasets', (route) => route.fulfill({ json: [{
    database_id: databaseId, label: 'UCI 零售交易', first_date: '2009-12-01', last_date: '2011-12-09',
    currency: 'GBP', modes: ['sql'], data_description: '零售交易', rule_description: '',
    query_examples: [], rule_examples: [],
  }] }))
  await page.route('**/sessions?*', (route) => route.fulfill({ json: [
    { session_id: 'history', title: '查数历史', created_at: 1 },
  ] }))
  await page.route('**/sessions/history/history?*', (route) => route.fulfill({
    json: { turns, next_after_turn: turns.length },
  }))

  await page.goto('/app/#online_retail_ii~sql/history')
  const answers = page.locator('.turn .answer')
  await expect(answers).toHaveCount(5)
  await expect(answers.nth(0)).toHaveText('共 25 笔交易，销售额为 8,366.86 英镑。')
  await expect(answers.nth(1)).toHaveText('取消发票共 23 笔。')
  await expect(answers.nth(2)).toHaveText('取消金额为 3,361.81 英镑。')
  await expect(answers.nth(3)).toContainText('国家或地区为 USA，销售额为 8,366.86 英镑。')
  await expect(answers.nth(3)).toContainText('国家或地区为 Japan，销售额暂无可汇总金额。')
  await expect(answers.nth(4)).toHaveText('没有找到符合条件的明细。')
  await expect(page.getByText('只展示已返回的部分记录，请缩小查询范围。')).toBeVisible()
  await expect(page.locator('.turn').getByRole('table')).toHaveCount(0)
  await expect(page.getByText(/transaction_count|total_sales_gbp|cancellation_amount_gbp|原始 SQL 结果/)).toHaveCount(0)
})

