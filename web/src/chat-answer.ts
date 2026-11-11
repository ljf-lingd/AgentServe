import type { Task } from './api'

const businessLabels: Record<string, string> = {
  country: '国家或地区', city: '城市', month: '月份', year: '年份', status: '订单状态',
  category: '类别', stock_code: '商品编号', sku: '商品编号', description: '商品描述',
  product: '商品', product_name: '商品', customer: '客户', customer_id: '客户编号',
  order_id: '订单编号', order_date: '下单日期', invoice_date: '开票日期', invoice: '发票编号',
  quantity: '数量', total_quantity: '总数量', customer_count: '客户数', country_count: '国家或地区数',
  sales: '销售额', cancellation: '取消金额', cancellation_amount: '取消金额',
  revenue: '成交额', total_revenue: '成交额', amount: '金额', price: '单价',
  total_amount: '总金额', average_amount: '平均金额', net_sales: '净销售额',
  unit_price: '单价', total_sales: '销售额', sales_amount: '销售额',
  transaction_count: '交易笔数', order_count: '订单笔数', invoice_count: '发票笔数',
  cancellation_count: '取消发票笔数', cancellation_invoice_count: '取消发票笔数',
  cancelled_invoice_count: '取消发票笔数', cancelled_order_count: '取消订单笔数',
  count: '计数',
}

const counts: Record<string, [string, string]> = {
  order_count: ['共', '笔订单'],
  transaction_count: ['共', '笔交易'],
  invoice_count: ['共', '笔发票'],
  cancellation_count: ['取消发票共', '笔'],
  cancellation_invoice_count: ['取消发票共', '笔'],
  cancelled_invoice_count: ['取消发票共', '笔'],
  cancelled_order_count: ['取消订单共', '笔'],
  customer_count: ['共有', '位客户'],
  country_count: ['涉及', '个国家或地区'],
}

const currency = { gbp: '英镑', cny: '元', cents: '分' }
const numberFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 20 })
const moneyFormat = new Intl.NumberFormat('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 20 })

function describe(name: string, value: unknown, index: number) {
  const column = name.toLowerCase().trim()
  const suffix = /_(gbp|cny|cents)$/.exec(column)?.[1] as keyof typeof currency | undefined
  const base = suffix ? column.slice(0, -(suffix.length + 1)) : column
  const distinct = /^count\s*\(\s*distinct\s+(?:\w+\.)?(country|invoice|customer_id)\s*\)$/.exec(base)?.[1]
  const key = distinct === 'country' ? 'country_count' : distinct === 'invoice' ? 'invoice_count'
    : distinct === 'customer_id' ? 'customer_count' : base
  const label = Object.hasOwn(businessLabels, key) ? businessLabels[key]
    : suffix ? '金额' : /^count\s*\(/.test(key) ? '计数' : '第 ' + (index + 1) + ' 项'
  if (value == null) return label + (suffix ? '暂无可汇总金额' : '暂无数据')
  const formatted = typeof value === 'number'
    ? (suffix === 'gbp' || suffix === 'cny' ? moneyFormat : numberFormat).format(value)
    : typeof value === 'string' ? value : '无法显示的值'
  const count = Object.hasOwn(counts, key) ? counts[key] : undefined
  if (count) return count[0] + ' ' + formatted + ' ' + count[1]
  return label + '为 ' + formatted + (suffix ? ' ' + currency[suffix] : '')
}

export function chatAnswer(task: Task): string | undefined {
  const report = task.report
  const result = report?.query_result
  if (report?.answer_source === 'workbench') return report.answer
  if (!result || !result.columns.length) {
    return report?.answer_source === 'model' ? report.answer : undefined
  }
  if (!result.rows.length) return '没有找到符合条件的记录。'
  if (result.rows.length === 1 && !result.truncated) {
    if (report.answer_source === 'model' && report.answer && !task.input.answer_mode && /\p{Script=Han}/u.test(report.answer)) return report.answer
    const [name] = result.columns
    const [value] = result.rows[0]
    if (result.columns.length === 1) {
      // Historic existence results were sometimes stored with the expression as their column name.
      if (/^EXISTS\s*\(/i.test(name) && (value === 0 || value === 1)) {
        return value === 1 ? '有符合条件的订单。' : '没有找到符合条件的订单。'
      }
      if (name === '1' && value === 1 && /^(?:有|有没有).+订单(?:吗|么)[？?]?$/.test(task.input.question ?? '')) {
        return '有符合条件的订单。'
      }
      if (name.endsWith('_gbp') && value == null) return '没有找到符合条件的明细。'
    }
  }
  const sentences = result.rows.map((row) => result.columns.map(
    (name, index) => describe(name, row[index], index),
  ).join('，') + '。')
  return sentences.length === 1 ? sentences[0] : sentences.map(
    (sentence, index) => (index + 1) + '、' + sentence,
  ).join('\n')
}

