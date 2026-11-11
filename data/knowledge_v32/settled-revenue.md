# 成交营收

成交营收仅计入 completed 订单。以订单日期筛选统计窗口；按明细数量乘历史成交单价后求和（SUM(order_items.quantity * order_items.unit_price_cents)），单位分。支持按客户城市或订单月份分组。无匹配明细时，未分组金额为 NULL。
